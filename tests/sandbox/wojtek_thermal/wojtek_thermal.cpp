/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */
#include <libmpdata++/concurr/threads.hpp>
#include <libmpdata++/output/hdf5_xdmf.hpp>
#include "wojtek_thermal.hpp"

using namespace libmpdataxx;
using T = double;

int main() 
{
  // compile-time parameters
  struct ct_params_t : ct_params_default_t
  {
    using real_t = T;
    enum { opts = opts::nug | opts::abs | opts::fct};
    enum { n_dims = 2 };
    enum { n_eqns = 5 };
    enum { rhs_scheme = solvers::trapez };
    enum { prs_scheme = solvers::cr };
    struct ix { enum {
      u, w, tht, qv, qc,
      vip_i=u, vip_j=w, vip_den=-1
    }; };
  }; 
  using ix = typename ct_params_t::ix;
  using real_t = typename ct_params_t::real_t;

  const int nx = 181, nz = 121, nt = 10 * 30;

  // conjugate residual
  using slv_out_t = output::hdf5_xdmf<wojtek_thermal<ct_params_t>>;
  // run-time parameters
  slv_out_t::rt_params_t p;

  T dx = 20;
  T dz = 20;

  p.dt = 2.0;
  p.di = dx;
  p.dj = dz; 
  p.prs_tol = 1e-7;
  p.grid_size = {nx, nz};
  p.n_iters = 1;

  p.outfreq = 10; //12;
  p.outvars = {
    {ix::qv, {"qv", "?"  }},
    {ix::tht, {"tht",  "?"  }},
    {ix::qc, {"qc", "?"  }},
    {ix::u, {"u", "?"  }},
    {ix::w, {"w", "?"  }}
  };
  p.outdir = "out";
  
  // constants init

  T gg = 9.72;
  T rg = 287;
  T rv = 461;
  T cp = 1005;
  T hlatv = 2.53e6;
  T hlats = 2.84e6;
  T tup = 168;
  T tdn = 153;
  T tt0 = 273.16;
  T ee0 = 611;

  p.gg    = gg   ; 
  p.rg    = rg   ; 
  p.rv    = rv   ; 
  p.cp    = cp   ; 
  p.hlatv = hlatv;
  p.hlats = hlats;
  p.tup   = tup  ; 
  p.tdn   = tdn  ; 
  p.tt0   = tt0  ; 
  p.ee0   = ee0  ; 

  libmpdataxx::concurr::threads<
    slv_out_t, 
    bcond::cyclic, bcond::cyclic,
    bcond::rigid, bcond::rigid
  > slv(p);

  rng_t i_r(0, nx - 1);
  rng_t k_r(0, nz - 1);


  // init
  {
    const auto& tht_e = slv.sclr_array("tht_e");
    const auto& tht_0 = slv.sclr_array("tht_0");
    const auto& thm_e = slv.sclr_array("thm_e");
    const auto& rho_0 = slv.sclr_array("rho_0");
    const auto& qv_e = slv.sclr_array("qv_e");

    auto alim01 = [](T x){return std::max(0.0, std::min(1.0, x));};
    auto comb = [&alim01](T tm, T td, T tu, T ad, T au){return alim01((tm-td)/(tu-td))*au + alim01((tu-tm)/(tu-td))*ad;};

    auto a=rg/rv;
    auto c=hlatv/cp;
    auto b=hlats/rv;
    auto d=hlatv/rv;
    auto e=-cp/rg;

    auto st = 1.3E-05;
    auto cap=rg/cp;
    auto capi=1./cap;

    thm_e(i_r, 0)=283.;
    auto pres=0.850e3;
    rho_0(i_r, 0)=pres*100./rg/thm_e(i_r, 0);
    auto relhum=.2;
    tht_e(i_r, 0) =thm_e(i_r, 0)*pow(1.e3/pres, (rg/cp));
    tht_0(i_r, 0) =tht_e(i_r, 0);
    auto pres0= pres;

    auto coe_l=comb(thm_e(0, 0),tdn,tup,0.,1.);
    auto tt=thm_e(0, 0);
    auto delt=(tt-tt0)/(tt*tt0);
    auto esw=ee0*exp(d * delt);
    auto esi=ee0*exp(b * delt);
    auto qvsw=a * esw /(pres*1.e2-esw);
    auto qvsi=a * esi /(pres*1.e2-esi);
    auto qvs=coe_l*qvsw + (1.-coe_l)*qvsi;
    qv_e(i_r, 0)=relhum*qvs;

    auto cs=gg/(cp*thm_e(0, 0)*st);
    for (int k = 1; k < nz; ++k)
    {
      auto exs=exp(-st*k*dz);
      tht_0(i_r, k)=tht_0(0, 0)/exs;
      tht_e(i_r, k)=tht_0(0, k);
      rho_0(i_r, k)=rho_0(0, 0)*exs*pow((1.-cs*(1.-exs)),(capi-1.));
      thm_e(i_r, k)=thm_e(0, 0)/exs*(1.-cs*(1.-exs));
      auto pres=pres0*pow((tht_e(0, k)/thm_e(0, k)), (-cp/rg));

      auto coe_l=comb(thm_e(0, k),tdn,tup,0.,1.);
      auto tt=thm_e(0, k);
      auto delt=(tt-tt0)/(tt*tt0);
      auto esw=ee0*exp(d * delt);
      auto esi=ee0*exp(b * delt);
      auto qvsw=a * esw /(pres*1.e2-esw);
      auto qvsi=a * esi /(pres*1.e2-esi);
      auto qvs=coe_l*qvsw + (1.-coe_l)*qvsi;
      qv_e(i_r, k)=relhum*qvs;
      
    }

    slv.advectee(ix::u) = 0; 
    slv.advectee(ix::w) = 0; 
    slv.advectee(ix::tht)(i_r, k_r) = tht_e(i_r, k_r);
    slv.advectee(ix::qc)(i_r, k_r) = 0;
    slv.g_factor()(i_r, k_r) = rho_0(i_r, k_r); 

    for (int k = 0; k < nz; ++k)
    {
      for (int i = 0; i < nx; ++i)
      {

         auto pi=4.*atan(1.);
         auto x1=i * dx;
         auto z1=k * dz;
         auto xc = (nx-1)*dx/2.;
         auto zc=800.;

         auto rad=sqrt(pow(x1-xc, 2) + pow(z1-zc, 2));
         auto del=1.0;
         auto coe=pi/2. * (rad-200.)/100.;
         coe=cos(coe)*cos(coe);
         if(rad>=200.) del=0.2+0.8*coe;
         if(rad>=300.) del=0.;

         auto relhum=0.2+del*0.8;

         auto thetme=tht_e(0, k)/thm_e(0, k);
         auto pre=1.e5*pow(thetme, e);
         auto tt=slv.advectee(ix::tht)(i,k)/thetme;
         auto delt=(tt-tt0)/(tt*tt0);
         auto esw=ee0*exp(d * delt);
         auto qvsw=a * esw /(pre-esw);

         slv.advectee(ix::qv)(i, k) =qvsw*relhum;
      }
    }

  }

  // integration
  slv.advance(nt);  

  std::cout.precision(18);
  std::cout << "MIN U   " << min(slv.advectee(ix::u)) << std::endl; 
  std::cout << "MIN W   " << min(slv.advectee(ix::w)) << std::endl; 
  std::cout << "MIN THT " << min(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "MIN QV  " << min(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "MIN QC  " << min(slv.advectee(ix::qc)) << std::endl; 

  std::cout << "MAX U   " << max(slv.advectee(ix::u)) << std::endl; 
  std::cout << "MAX W   " << max(slv.advectee(ix::w)) << std::endl; 
  std::cout << "MAX THT " << max(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "MAX QV  " << max(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "MAX QC  " << max(slv.advectee(ix::qc)) << std::endl; 
  
  std::cout << "SUM U   " << sum(slv.advectee(ix::u)) << std::endl; 
  std::cout << "SUM W   " << sum(slv.advectee(ix::w)) << std::endl; 
  std::cout << "SUM THT " << sum(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "SUM QV  " << sum(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "SUM QC  " << sum(slv.advectee(ix::qc)) << std::endl; 
}
