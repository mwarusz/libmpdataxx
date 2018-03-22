/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */
#include <libmpdata++/concurr/threads.hpp>
#include <libmpdata++/output/hdf5_xdmf.hpp>
#include "supercell.hpp"

using namespace libmpdataxx;
using T = double;

const T pi = std::acos(-1.0);

int main() 
{
  // compile-time parameters
  struct ct_params_t : ct_params_default_t
  {
    using real_t = T;
    enum { opts = opts::nug | opts::iga | opts::fct};
    enum { n_dims = 3 };
    enum { n_eqns = 6 };
    enum { rhs_scheme = solvers::trapez };
    enum { prs_scheme = solvers::cr };
    struct ix { enum {
      u, v, w, tht, qv, qc,
      vip_i=u, vip_j=v, vip_k=w, vip_den=-1
    }; };
  }; 
  using ix = typename ct_params_t::ix;
  using real_t = typename ct_params_t::real_t;

  const int nx = 65, ny = 65, nz = 52, nt = 12 * 60 * 2;

  // conjugate residual
  using slv_out_t = output::hdf5_xdmf<supercell<ct_params_t>>;
  // run-time parameters
  slv_out_t::rt_params_t p;

  T dx = 2000;
  T dy = dx;
  T dz = 350;

  p.dt = 5.0;
  p.di = dx;
  p.dj = dy;
  p.dk = dz; 
  p.prs_tol = 1e-6;
  p.grid_size = {nx, ny, nz};
  p.n_iters = 2;

  p.outfreq = 30; //12;
  p.outvars = {
    {ix::qv, {"qv", "?"  }},
    {ix::tht, {"tht",  "?"  }},
    {ix::qc, {"qc", "?"  }},
    {ix::u, {"u", "?"  }},
    {ix::v, {"v", "?"  }},
    {ix::w, {"w", "?"  }}
  };
  p.outdir = "out";
  
  // constants init

  T gg = 9.8016;
  T rg = 287.04;
  T rv = 461;
  T cp = 1005;
  T hlatv = 2.53e6;
  T hlats = 2.84e6;
  T tup = 168;
  T tdn = 153;
  T tt0 = 300;
  T ee0 = 611;
  
  T pr00 = 1e5;
  T pref00 = 1e5;
  
  T tht_tr = 343;
  T T_tr = 213;
  T z_tr = 12 * 1000;

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

  auto rc = rg / cp;
  auto rci = 1./rc;

  libmpdataxx::concurr::threads<
    slv_out_t, 
    bcond::cyclic, bcond::cyclic,
    bcond::cyclic, bcond::cyclic,
    bcond::rigid, bcond::rigid
  > slv(p);

  rng_t i_r(0, nx - 1);
  rng_t j_r(0, ny - 1);
  rng_t k_r(0, nz - 1);

  // init
  {
    const auto& tht_e = slv.sclr_array("tht_e");
    const auto& tht_0 = slv.sclr_array("tht_0");
    const auto& thm_e = slv.sclr_array("thm_e");
    const auto& rho_0 = slv.sclr_array("rho_0");
    const auto& qv_e = slv.sclr_array("qv_e");
    decltype(slv.advectee()) pres(slv.advectee().shape());
    decltype(slv.advectee()) rh_e(slv.advectee().shape());

    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      rho_0(i_r, j_r, k) = 1.11;
      tht_0(i_r, j_r, k) = tt0;

      T z = k * dz;
      if (z < z_tr)
      {
        tht_e(i_r, j_r, k) = tt0 + (tht_tr - tt0) * std::pow(z / z_tr, 5./4);
        thm_e(i_r, j_r, k) = tht_e(i_r, j_r, k);

        rh_e(i_r, j_r, k) = 1 - 0.75 * std::pow(z / z_tr, 5./4); 
      }
      else
      {
        tht_e(i_r, j_r, k) = tht_tr * std::exp(gg / (cp * T_tr) * (z - z_tr));
        thm_e(i_r, j_r, k) = tht_e(i_r, j_r, k);
        
        rh_e(i_r, j_r, k) = 0.25;
      }
    }

    pres(i_r, j_r, 0) = pr00 / 100.;
    for (int k = k_r.first() + 1; k <= k_r.last(); ++k)
    {
      auto delt = tht_e(0, 0, k) - tht_e(0, 0, k - 1);
      T tavi;
      if (delt > 1e-4)
      {
        tavi = std::log(tht_e(0, 0, k) / tht_e(0, 0, k - 1)) / delt;
      }
      else
      {
        tavi = 1./ tht_e(0, 0, k);
      }
      pres(i_r, j_r, k) = pow(pow(pres(i_r, j_r, k - 1), rc) - gg * std::pow(1e3, rc) * tavi * dz / cp, rci);
    }

    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      pres(i_r, j_r, k) *= 100;
      //tht_e(i_r, j_r, k) *= pow(pres(i_r, j_r, k) / pref00, rc);
      thm_e(i_r, j_r, k) *= pow(pres(i_r, j_r, k) / pref00, rc);
    }

    auto alim01 = [](T x){return std::max(0.0, std::min(1.0, x));};
    auto comb = [&alim01](T tm, T td, T tu, T ad, T au){return alim01((tm-td)/(tu-td))*au + alim01((tu-tm)/(tu-td))*ad;};

    auto a=rg/rv;
    auto c=hlatv/cp;
    auto b=hlats/rv;
    auto d=hlatv/rv;
    auto e=-cp/rg;

    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      //auto prs = pres(0, 0, k);
      //auto coe_l=comb(thm_e(0, 0, k),tdn,tup,0.,1.);
      //auto tt=thm_e(0, 0, k);
      //auto delt=(tt-tt0)/(tt*tt0);
      //auto esw=ee0*exp(d * delt);
      //auto esi=ee0*exp(b * delt);
      //auto qvsw=a * esw /(prs-esw);
      //auto qvsi=a * esi /(prs-esi);
      //auto qvs=coe_l*qvsw + (1.-coe_l)*qvsi;
      auto relhum = rh_e(0, 0, k);
      auto qvs = 380. / pres(0, 0, k) * std::exp(17.27 * (thm_e(0, 0, k) - 273) / (thm_e(0, 0, k) - 36.));
      qv_e(i_r, j_r, k)=std::min(relhum*qvs, 0.014);
    
      
      T z = k * dz;
      T u00 = 15.;
      T zs = 3e3;
      slv.advectee(ix::u)(i_r, j_r, k) = u00 * std::tanh(z / zs); 
      slv.advectee(ix::u)(i_r, j_r, k) -= 0.61 * u00; 
    }
    
    slv.advectee(ix::v) = 0; 
    slv.advectee(ix::w) = 0; 

    slv.advectee(ix::tht)(i_r, j_r, k_r) = tht_e(i_r, k_r);
    slv.advectee(ix::qv)(i_r, j_r, k_r) = qv_e(i_r, j_r, k_r);
    slv.advectee(ix::qc)(i_r, j_r, k_r) = 0;
    slv.g_factor()(i_r, j_r, k_r) = rho_0(i_r, j_r, k_r); 
   
    T r0x = 10e3;
    T r0z = 1.4e3;
    T x0 = 0.5 * (nx - 1) * dx;
    T y0 = 0.5 * (ny - 1) * dy;
    // introduce tht perturbation
    for (int i = i_r.first(); i <= i_r.last(); ++i)
    {
      for (int j = j_r.first(); j <= j_r.last(); ++j)
      {
        for (int k = k_r.first(); k <= k_r.last(); ++k)
        {
          auto x = i * dx;
          auto y = j * dy;
          auto z = k * dz;
          auto rad = std::sqrt(std::pow((x - x0) / r0x, 2) + std::pow((y - y0) / r0x, 2) + std::pow((z - r0z) / r0z, 2));
          auto del=2.*std::pow((std::cos(0.5*pi*rad)), 2);
          del *= tht_e(i, j, k) / thm_e(i, j, k);

          if (rad <= 1.)
          {
            slv.advectee(ix::tht)(i, j, k) += del;
          }
        }
      }
    }


    //auto st = 1.3E-05;
    //auto cap=rg/cp;
    //auto capi=1./cap;

    //thm_e(i_r, 0)=283.;
    //auto pres=0.850e3;
    //rho_0(i_r, 0)=pres*100./rg/thm_e(i_r, 0);
    //auto relhum=.2;
    //tht_e(i_r, 0) =thm_e(i_r, 0)*pow(1.e3/pres, (rg/cp));
    //tht_0(i_r, 0) =tht_e(i_r, 0);
    //auto pres0= pres;

    //auto coe_l=comb(thm_e(0, 0),tdn,tup,0.,1.);
    //auto tt=thm_e(0, 0);
    //auto delt=(tt-tt0)/(tt*tt0);
    //auto esw=ee0*exp(d * delt);
    //auto esi=ee0*exp(b * delt);
    //auto qvsw=a * esw /(pres*1.e2-esw);
    //auto qvsi=a * esi /(pres*1.e2-esi);
    //auto qvs=coe_l*qvsw + (1.-coe_l)*qvsi;
    //qv_e(i_r, 0)=relhum*qvs;

    //auto cs=gg/(cp*thm_e(0, 0)*st);
    //for (int k = 1; k < nz; ++k)
    //{
    //  auto exs=exp(-st*k*dz);
    //  tht_0(i_r, k)=tht_0(0, 0)/exs;
    //  tht_e(i_r, k)=tht_0(0, k);
    //  rho_0(i_r, k)=rho_0(0, 0)*exs*pow((1.-cs*(1.-exs)),(capi-1.));
    //  thm_e(i_r, k)=thm_e(0, 0)/exs*(1.-cs*(1.-exs));
    //  auto pres=pres0*pow((tht_e(0, k)/thm_e(0, k)), (-cp/rg));

    //  auto coe_l=comb(thm_e(0, k),tdn,tup,0.,1.);
    //  auto tt=thm_e(0, k);
    //  auto delt=(tt-tt0)/(tt*tt0);
    //  auto esw=ee0*exp(d * delt);
    //  auto esi=ee0*exp(b * delt);
    //  auto qvsw=a * esw /(pres*1.e2-esw);
    //  auto qvsi=a * esi /(pres*1.e2-esi);
    //  auto qvs=coe_l*qvsw + (1.-coe_l)*qvsi;
    //  qv_e(i_r, k)=relhum*qvs;
    //  
    //}


    //for (int k = 0; k < nz; ++k)
    //{
    //  for (int i = 0; i < nx; ++i)
    //  {

    //     auto pi=4.*atan(1.);
    //     auto x1=i * dx;
    //     auto z1=k * dz;
    //     auto xc = (nx-1)*dx/2.;
    //     auto zc=800.;

    //     auto rad=sqrt(pow(x1-xc, 2) + pow(z1-zc, 2));
    //     auto del=1.0;
    //     auto coe=pi/2. * (rad-200.)/100.;
    //     coe=cos(coe)*cos(coe);
    //     if(rad>=200.) del=0.2+0.8*coe;
    //     if(rad>=300.) del=0.;

    //     auto relhum=0.2+del*0.8;

    //     auto thetme=tht_e(0, k)/thm_e(0, k);
    //     auto pre=1.e5*pow(thetme, e);
    //     auto tt=slv.advectee(ix::tht)(i,k)/thetme;
    //     auto delt=(tt-tt0)/(tt*tt0);
    //     auto esw=ee0*exp(d * delt);
    //     auto qvsw=a * esw /(pre-esw);

    //     slv.advectee(ix::qv)(i, k) =qvsw*relhum;
    //  }
    //}

  }
  std::cout.precision(18);
  std::cout << "MIN U   " << min(slv.advectee(ix::u)) << std::endl; 
  std::cout << "MIN V   " << min(slv.advectee(ix::v)) << std::endl; 
  std::cout << "MIN W   " << min(slv.advectee(ix::w)) << std::endl; 
  std::cout << "MIN THT " << min(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "MIN QV  " << min(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "MIN QC  " << min(slv.advectee(ix::qc)) << std::endl; 

  std::cout << "MAX U   " << max(slv.advectee(ix::u)) << std::endl; 
  std::cout << "MAX V   " << max(slv.advectee(ix::v)) << std::endl; 
  std::cout << "MAX W   " << max(slv.advectee(ix::w)) << std::endl; 
  std::cout << "MAX THT " << max(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "MAX QV  " << max(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "MAX QC  " << max(slv.advectee(ix::qc)) << std::endl; 

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
  
  //std::cout << "SUM U   " << sum(slv.advectee(ix::u)) << std::endl; 
  //std::cout << "SUM W   " << sum(slv.advectee(ix::w)) << std::endl; 
  //std::cout << "SUM THT " << sum(slv.advectee(ix::tht)) << std::endl; 
  //std::cout << "SUM QV  " << sum(slv.advectee(ix::qv)) << std::endl; 
  //std::cout << "SUM QC  " << sum(slv.advectee(ix::qc)) << std::endl; 
}
