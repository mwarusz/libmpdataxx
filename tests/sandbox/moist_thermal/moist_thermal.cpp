/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */
#include <libmpdata++/concurr/threads.hpp>
#include <libmpdata++/output/hdf5_xdmf.hpp>

#include <libcloudph++/common/moist_air.hpp>
#include <libcloudph++/common/const_cp.hpp>
#include <libcloudph++/common/theta_std.hpp>
#include <libcloudph++/common/earth.hpp>


using libcloudphxx::common::theta_std::p_1000;
using libcloudphxx::common::moist_air::R_d_over_c_pd;
using libcloudphxx::common::moist_air::c_pd;
using libcloudphxx::common::moist_air::R_d;
using libcloudphxx::common::moist_air::R_v;
using libcloudphxx::common::const_cp::l_tri;
using libcloudphxx::common::const_cp::p_vs;
using libcloudphxx::common::const_cp::r_vs;
using libcloudphxx::common::theta_std::p_1000;


#include <iostream>
#include "moist_thermal.hpp"

using namespace libmpdataxx;
using real_t = double;

//const T pi = std::acos(-1.0);

struct profiles_t
{
  blitz::Array<real_t, 1> th_e, p_e, rv_e, rl_e, th_ref, rhod;

  profiles_t(int nz) :
    th_e(nz), p_e(nz), rv_e(nz), rl_e(nz), th_ref(nz), rhod(nz)
  {}
};

quantity<si::dimensionless, real_t> RH_T_p_to_rv(const real_t &RH,
                                                  const quantity<si::temperature, real_t> &T,
                                                  const quantity<si::pressure, real_t> &p)
{
  return RH * r_vs<real_t>(T, p);
}


const quantity<si::temperature, real_t> T_0(283. * si::kelvins);  // surface temperature
const quantity<si::pressure, real_t> p_0(85000 * si::pascals); // total surface temperature
const real_t stab = 1.3e-5; // stability, 1/m
const real_t env_RH = 0.2;
const real_t prtrb_RH = 1. - 1e-10;
const quantity<si::temperature, real_t> th_std_0 = T_0 / pow(p_0 / p_1000<real_t>(),  R_d_over_c_pd<real_t>());
const quantity<si::dimensionless, real_t> rv_0 = RH_T_p_to_rv(env_RH, T_0, p_0);
const quantity<si::dimensionless, real_t> qv_0 = rv_0 / (1. + rv_0); // specific humidity at surface
const quantity<si::length, real_t> 
 Z    ( 2400 * si::metres), 
 X    ( 3600 * si::metres), 
 Y    ( 3600 * si::metres), 
 z_prtrb ( 800 * si::metres);
const real_t rhod_surf = libcloudphxx::common::theta_std::rhod(p_0, th_std_0, rv_0) * si::cubic_metres / si::kilograms;
const real_t cs = (libcloudphxx::common::earth::g<real_t>() / si::metres_per_second_squared) / (c_pd<real_t>() / si::joules * si::kilograms * si::kelvins) / stab / (T_0 / si::kelvins);
const real_t z_abs = 125000; // [m] height above which absorber works, no absorber

struct th_std_fctr
{
  const real_t th_surf;
  th_std_fctr(const real_t th) :
    th_surf(th) {}

  real_t operator()(const real_t &z) const
  {
    return (th_surf) * real_t(exp(stab * z));
  }
  BZ_DECLARE_FUNCTOR(th_std_fctr);
};


struct rho_fctr
{
  const real_t rh_surf;
  rho_fctr(const real_t &rho) :
    rh_surf(rho) {}

  real_t operator()(const real_t &z) const
  {
    return rh_surf * exp(- stab * z) * pow(
             1. - cs * (1 - exp(- stab * z)), (1. / R_d_over_c_pd<real_t>()) - 1);
  }
  BZ_DECLARE_FUNCTOR(rho_fctr);
};

struct RH
{
  quantity<si::dimensionless, real_t> operator()(const real_t &r) const // r - distance from the center of the perturbation
  {
    if(r <= 250.)
      return prtrb_RH;
    else if(r >= 350.)
      return env_RH;
    else // transition layer
      return env_RH + (prtrb_RH - env_RH) * pow( cos(boost::math::constants::pi<real_t>() / 2. * (r - 250) / 100.), 2);
  }
BZ_DECLARE_FUNCTOR(RH);
};

struct prtrb_rv 
{
  blitz::Array<real_t, 1> &_T, &_p;
  real_t dz;
  prtrb_rv(blitz::Array<real_t, 1> &_T, blitz::Array<real_t, 1> &_p, real_t dz): _T(_T), _p(_p), dz(dz) {}

  quantity<si::dimensionless, real_t> operator()(const real_t &r, const real_t &z) const
  {
    return RH_T_p_to_rv(RH()(r), this->_T(z/this->dz) * si::kelvins , this->_p(z/this->dz) * si::pascals);
  }
  BZ_DECLARE_FUNCTOR2(prtrb_rv);
};

blitz::Array<real_t, 1> env_prof(profiles_t &profs, int nz, real_t dz)
{

  blitz::firstIndex k;
  // temperature and total pressure profiles
  blitz::Array<real_t, 1> T(nz), pre_ref(nz);

  real_t tt0 = 273.17;
  real_t rv = 461;
  real_t ee0 = 611.;
  const real_t gg = 9.81;
  real_t rg = R_d<real_t>() / si::joules * si::kelvins * si::kilograms;
  real_t a = R_d<real_t>() / rv / si::joules * si::kelvins * si::kilograms;
  real_t b = l_tri<real_t>() / si::joules * si::kilograms / rv / tt0;
  real_t c = l_tri<real_t>() / c_pd<real_t>() / si::kelvins;
  real_t d = l_tri<real_t>() / si::joules * si::kilograms / rv;
  real_t cap = R_d_over_c_pd<real_t>(); 
  real_t capi = 1./cap;

  // surface data
  
  real_t tt = T_0 / si::kelvins; // T(0)
  real_t delt = (tt - tt0) / (tt * tt0); 
  real_t esw = ee0*exp(d * delt);
  real_t qvs = a * esw / ((p_0 / si::pascals) -esw);
  //rv_e(0) = env_RH * qvs;
  profs.rv_e(0) = rv_0;// env_RH * qvs;
  profs.rl_e = 0.;
  real_t th_e_surf = th_std_0 / si::kelvins * (1 + a * profs.rv_e(0)); // virtual potential temp

  profs.th_e = th_std_fctr(th_e_surf)(k * dz);
  
  pre_ref(0.) = p_0 / si::pascals;
  profs.p_e(0) = pre_ref(0);
  T(0.) = T_0 / si::kelvins;
  
  for(int k=1; k<nz; ++k)
  {
    real_t zz = k * dz;  
    // predictor
     real_t rhob=pre_ref(k-1) / rg / (T(k-1)*(1.+a*profs.rv_e(k-1))); // density of air at k-1
     pre_ref(k)=pre_ref(k-1) - gg*rhob*dz; // estimate of pre at k (dp = -g * rho * dz)
 //iteration for T and qv:
     profs.rv_e(k)=profs.rv_e(k-1);
     T(k)=profs.th_e(k)* pow(pre_ref(k)/1.e5, cap); 
     T(k)=T(k)/(1.+a*profs.rv_e(k));
    
    for(int iter=0; iter<4; ++iter)
    {
      tt=T(k);
      delt=(tt-tt0)/(tt*tt0);
      esw=ee0*exp(d * delt);
      qvs=a * esw /(pre_ref(k)-esw);
      profs.rv_e(k)=env_RH*qvs;
     T(k)=profs.th_e(k)* pow(pre_ref(k)/1.e5, cap);
      T(k)=T(k)/(1.+a*profs.rv_e(k));
    }

    // corrector
     real_t rhon=pre_ref(k) / rg / (T(k)*(1.+a*profs.rv_e(k)));
     pre_ref(k)=pre_ref(k-1) - gg*(rhob+rhon) / 2. *dz;
 //iteration for T and qv:
     T(k)=profs.th_e(k)* pow(pre_ref(k)/1.e5, cap);
     T(k)=T(k)/(1.+a*profs.rv_e(k));
    
    for(int iter=0; iter<4; ++iter)
    {
      tt=T(k);
      delt=(tt-tt0)/(tt*tt0);
      esw=ee0*exp(d * delt);
      qvs=a * esw /(pre_ref(k)-esw);
      profs.rv_e(k)=env_RH*qvs;
      T(k)=profs.th_e(k)* pow(pre_ref(k)/1.e5, cap);
      T(k)=T(k)/(1.+a*profs.rv_e(k));
    }
    //rv_e(k) =  RH_T_p_to_rv(env_RH, T(k) * si::kelvins, pre_ref(k) * si::pascals); // cheating!
    profs.p_e(k) = pre_ref(k);

  }

  //th_ref = th_std_fctr(th_std_0 / si::kelvins)(k * dz);
  profs.rhod = rho_fctr(rhod_surf)(k * dz); // rhod is dry density profsile?

  // turn virtual potential temperature env profsile into env profsile of standard potential temp
  profs.th_e = profs.th_e / (1. + a * profs.rv_e);

  profs.th_ref = profs.th_e;//th_std_fctr(th_std_0 / si::kelvins)(k * dz);
  return T;
}


template <int opts_arg>
void test(const int scale, std::string name)
{
  // compile-time parameters
  struct ct_params_t : ct_params_default_t
  {
    using real_t = ::real_t;
    enum { var_dt = false};
    enum { opts = opts_arg};
    //enum { opts = opts::nug | opts::abs };
    enum { sptl_intrp = solvers::aver4};
    enum { vip_vab = solvers::impl};
    enum { n_dims = 2 };
    enum { n_eqns = 7 };
    enum { rhs_scheme = solvers::mixed };
    enum { prs_scheme = solvers::cr };
    struct ix { enum {
      u, w, tht, qv, qc, qr, thf,
      vip_i=u, vip_j=w, vip_den=-1
    }; };
  }; 
  using ix = typename ct_params_t::ix;
  using real_t = typename ct_params_t::real_t;

  const real_t sim_time = 7 * 60;

  const int nx = scale * 180 + 1, nz = scale * 120 + 1;

  using slv_out_t = typename output::hdf5_xdmf<moist_thermal<ct_params_t>>;
  // run-time parameters
  typename slv_out_t::rt_params_t p;

  real_t length_x = X / si::metres;
  real_t length_z = Z / si::metres;

  real_t dx = length_x / (nx - 1);
  real_t dz = length_z / (nz - 1);

  if (ct_params_t::var_dt)
  {
    p.dt = 3.0 * scale;
  }
  else
  {
    p.dt = 2. / scale;
  }

  int nt = sim_time / p.dt;

  p.di = dx;
  p.dj = dz;
  p.prs_tol = 1e-5;
  p.grid_size = {nx, nz};
  p.n_iters = 2;
  
  if (ct_params_t::var_dt)
  {
    p.max_courant = 0.8;
  }

  if (ct_params_t::var_dt)
  {
    p.outfreq = 10 * 60;
  }
  else
  {
    p.outfreq = nt / 7;
  }
  p.outvars = {
    {ix::qv, {"qv", "?"  }},
    {ix::qc, {"qc", "?"  }},
    {ix::qr, {"qr", "?"  }},
    {ix::tht, {"tht",  "?"  }},
    {ix::u, {"u", "?"  }},
    {ix::w, {"w", "?"  }}
  };
  p.outdir = name + "_" + std::to_string(nx - 1);
  p.name = "stats_" + name.erase(0, 4) + "_" + std::to_string(nx - 1);

  const real_t cp = 1004.5;
  const real_t Rd = 287.;
  const real_t R_d_over_c_pd_v = Rd / cp;
  const real_t T0 = 273.16;
  const real_t L = 2.53e6;
  const real_t e0 = 611;
  const real_t Rv = 461.;
  const real_t epsa = Rd/Rv;
  const real_t epsb = Rv/Rd-1;
  
  p.g = 9.81;
  p.cp = cp;
  p.Rd = Rd;
  p.Rv = Rv;
  p.L = L;
  p.e0 = e0;
  p.epsa = epsa;
  p.buoy_eps = epsb;
  p.T0 = T0;

  libmpdataxx::concurr::threads<
    slv_out_t, 
    bcond::cyclic, bcond::cyclic,
    bcond::rigid, bcond::rigid
  > slv(p);



  rng_t i_r(0, nx - 1);
  rng_t k_r(0, nz - 1);

  // init profiles
  profiles_t profs(nz);
  auto T = env_prof(profs, nz, dz);
  const auto& tht_e = slv.sclr_array("tht_e");
  const auto& pk_e = slv.sclr_array("pk_e");
  const auto& qv_e = slv.sclr_array("qv_e");
  const auto& tht_b = slv.sclr_array("tht_b");
  const auto& rho_b = slv.g_factor();

  for (int k = 0; k < nz; ++k)
  {
    tht_e(i_r, k) = profs.th_e(k);
    qv_e(i_r, k) = profs.rv_e(k);
    pk_e(i_r, k) = std::pow(profs.p_e(k) / 1e5, R_d_over_c_pd_v);
    
    tht_b(i_r, k) = profs.th_ref(k);
    rho_b(i_r, k) = profs.rhod(k);
  }

  // init state
  slv.advectee(ix::u) = 0;
  slv.advectee(ix::w) = 0;  
  
  // absorbers
  //slv.vab_coefficient() = where(blitz::tensor::j * dz >= z_abs,
  //                                 1. / 100 * pow(sin(3.1419 / 2. * (blitz::tensor::j * dz - z_abs)/ (length_z - z_abs)), 2), 0);
  
  slv.vab_coefficient() = 0;
  slv.vab_relaxed_state(0) = 0;
  slv.vab_relaxed_state(1) = 0; // vertical relaxed state


  slv.advectee(ix::tht) = 0;
  slv.advectee(ix::thf) = tht_e(i_r, k_r);

  slv.advectee(ix::qv) = prtrb_rv(T, profs.p_e, dz)(
    sqrt(
      pow(blitz::tensor::i * dx - (X / si::metres / 2.), 2) + 
      pow(blitz::tensor::j * dz - (z_prtrb / si::metres), 2)
    ),
    blitz::tensor::j * dz
  );

  slv.advectee(ix::qc) = 0;
  slv.advectee(ix::qr) = 0;

  // init
  //{
  //  slv.advectee(ix::w) = 0;
  //  slv.advectee(ix::qc) = 0;
  //  slv.advectee(ix::qr) = 0;
  //  
  //  slv.advectee(ix::qv)(i_r, k_r) = qv_e(i_r, k_r);

  //  slv.advectee(ix::tht)(i_r, k_r) = 0;

  //  slv.advectee(ix::u)(i_r, k_r) = 0;

  //  const T r0x = 10e3;
  //  const T r0z = 10e3;

  //  const T x0 = 0.5 * length_x;
  //  const T z0 = r0z;
  //  
  //  decltype(slv.advectee()) rr(slv.advectee().shape()), delta(slv.advectee().shape());
  //  {
  //    blitz::firstIndex i;
  //    blitz::secondIndex k;

  //    rr = sqrt(pow2((i * dx - x0) / r0x) + pow2((k * dz - z0) / r0z)); 

  //    delta = where(rr(i, k) <= 1.0,
  //                  3 * pow2(cos(0.5 * pi * rr(i, k))),
  //                  0.0);
  //      
  //    slv.advectee(ix::tht) += delta(i, k);
  //  
  //    const T z_abs = 15000;
  //    slv.vab_coefficient() = where(k * dz >= z_abs,
  //                                   1. / 100 * (k * dz - z_abs) / (length_z - z_abs),
  //                                   0);
  //    //slv.vab_coefficient() = 0.;
  //  }
  //  
  //  slv.advectee(ix::thf)(i_r, k_r) =   slv.advectee(ix::tht)(i_r, k_r)
  //                                      + slv.sclr_array("tht_e")(i_r, k_r);
  //  
  //  slv.vab_relaxed_state(0)(i_r, k_r) = 0;
  //  slv.vab_relaxed_state(1) = 0;
  //  
  //  std::cout.precision(18);
  //  for (int k = k_r.first(); k <= k_r.last(); ++k)
  //  {
  //    std::cout << k * dz << ' '
  //      << qv_e(0, k) << ' '
  //      << wk_T(k) << ' '
  //      << slv.advectee(ix::u)(0, k) << ' '
  //      << wk_tht(k) << std::endl;
  //  }
  //}

  std::cout << "Calculating: " << p.outdir << std::endl; 

  // integration
  if (ct_params_t::var_dt)
  {
    slv.advance(sim_time);
  }
  else
  {
    slv.advance(nt);
  }
}

int main() 
{
  std::vector<int> scales = {4}; 
  for (const auto scale : scales)
  {
    {
      enum { opts = opts::nug | opts::iga | opts::fct};
      test<opts>(scale, "out_thermal_cdt_Mg2No");
    }
    
    //{
    //  enum { opts = opts::nug | opts::abs | opts::fct};
    //  test<opts>(scale, "out_thermal_cdt_Mp2No");
    //}
    //{
    //  enum { opts = opts::nug | opts::iga | opts::tot | opts::fct};
    //  test<false, opts>(500, np, "out_phd_vdt80_Mg3ccNo_iles");
    //}
    //{
    //  enum { opts = opts::nug | opts::iga | opts::div_2nd | opts::div_3rd | opts::fct};
    //  test<false, opts>(v_mult, np, "out_obrona_vdt80_Mg3No" + suffix);
    //}
  }
}
