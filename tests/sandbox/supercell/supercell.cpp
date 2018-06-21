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

template <bool simple>
void test(T eta, const int np, std::string name)
{
  // compile-time parameters
  struct ct_params_t : ct_params_default_t
  {
    using real_t = T;
    enum { impl_tht = false};
    enum { var_dt = false};
    enum { sgs_scheme = solvers::iles};
    enum { stress_diff = simple ? solvers::normal : solvers::compact};
    //enum { opts = opts::nug | opts::abs | opts::fct};
    enum { opts = opts::nug | opts::iga | opts::fct};
    enum { vip_vab = solvers::impl};
    enum { n_dims = 3 };
    enum { n_eqns = 7 };
    enum { rhs_scheme = solvers::mixed };
    enum { prs_scheme = solvers::cr };
    struct ix { enum {
      u, v, w, tht, qv, qc, qr,
      vip_i=u, vip_j=v, vip_k=w, vip_den=-1
    }; };
  }; 
  using ix = typename ct_params_t::ix;
  using real_t = typename ct_params_t::real_t;

  const int scale = 336 / np;

  const T sim_time = 7200;

  const int nx = np, ny = np, nz = 41;

  using slv_out_t = typename output::hdf5_xdmf<supercell<ct_params_t>>;
  // run-time parameters
  typename slv_out_t::rt_params_t p;

  T length_x = 167e3;
  T length_y = 167e3;
  T length_z = 20e3;

  T dx = length_x / (nx - 1);
  T dy = length_y / (ny - 1);
  T dz = length_z / (nz - 1);

  if (ct_params_t::var_dt)
  {
    p.dt = 3.0 * scale;
  }
  else
  {
    p.dt = 2.5 * scale;
  }

  int nt = sim_time / p.dt;
  //nt = 100;

  p.di = dx;
  p.dj = dy;
  p.dk = dz; 
  p.prs_tol = 1e-5;
  p.grid_size = {nx, ny, nz};
  p.n_iters = 2;
  
  p.buoy_filter = true;
  //p.eta = eta;
  
  if (ct_params_t::var_dt)
  {
    p.max_courant = 0.97;
  }

  if (ct_params_t::var_dt)
  {
    p.outfreq = 10 * 60;
  }
  else
  {
    p.outfreq = nt / 24;
  }
  p.outvars = {
    {ix::qv, {"qv", "?"  }},
    {ix::qc, {"qc", "?"  }},
    {ix::qr, {"qr", "?"  }},
    {ix::tht, {"tht",  "?"  }},
    {ix::u, {"u", "?"  }},
    {ix::v, {"v", "?"  }},
    {ix::w, {"w", "?"  }}
  };
  p.outdir = name + "_" + std::to_string(np);
  p.name = "stats_" + name.erase(0, 4) + "_" + std::to_string(np);

  const T cp = 1004.5;
  const T Rd = 287.;
  const T R_d_over_c_pd_v = Rd / cp;
  const T T0 = 273.16;
  const T L = 2.53e6;
  const T e0 = 611;
  const T Rv = 461.;
  const T epsa = Rd/Rv;
  const T epsb = Rv/Rd-1;
  
  p.g = 9.81;
  p.cp = cp;
  p.Rd = Rd;
  p.Rv = Rv;
  p.L = L;
  p.e0 = e0;
  p.epsa = epsa;
  p.buoy_eps = epsb;
  p.T0 = T0;

  std::cout << "epsb: " << epsb << std::endl;



  libmpdataxx::concurr::threads<
    slv_out_t, 
    bcond::cyclic, bcond::cyclic,
    bcond::cyclic, bcond::cyclic,
    bcond::gndsky, bcond::gndsky
  > slv(p);

  rng_t i_r(0, nx - 1);
  rng_t j_r(0, ny - 1);
  rng_t k_r(0, nz - 1);

  // init
  {

    blitz::Array<T, 1> wk_tht(nz), wk_RH(nz), wk_p(nz), wk_T(nz),  wk_thd(nz), wk_pi(nz);
   

    const T z_tr = 12e3;
    const T tht_0 = 300;
    const T tht_tr = 343;
    const T T_tr = 213;
    
    
    const auto& tht_e = slv.sclr_array("tht_e");
    const auto& pk_e = slv.sclr_array("pk_e");
    const auto& qv_e = slv.sclr_array("qv_e");

    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      T z = k * dz;
      if (z < z_tr)
      {
        wk_tht(k) = tht_0 + (tht_tr - tht_0) * std::pow(z / z_tr, 5./4);
        wk_RH(k) = 1. - 0.75 * std::pow(z / z_tr, 5./4); 
      }
      else
      {
        wk_tht(k) = tht_tr * std::exp(p.g / (cp * T_tr) * (z - z_tr));
        wk_RH(k) = 0.25;
      }
      
      wk_thd(k) = wk_tht(k);
    }
    
    wk_pi(0) = 1;
    for (int iter = 0; iter < 10; ++iter)
    {
      for (int k = k_r.first() + 1; k <= k_r.last(); ++k)
      {
        wk_pi(k) = wk_pi(k - 1) - 2. * p.g / cp * dz / (wk_thd(k) + wk_thd(k - 1));
      }
      
      for (int k = k_r.first(); k <= k_r.last(); ++k)
      {
        wk_p(k) = 1e5 * std::pow(wk_pi(k), 1./ R_d_over_c_pd_v);
        wk_T(k) = wk_tht(k) * wk_pi(k);

        T p = wk_p(k);
        T Temp = wk_T(k);
        T es = e0 * std::exp(L / Rv * ((Temp - T0) / (T0 * Temp)));
        T qvs = epsa * es / (p - es);
        
        qv_e(i_r, j_r, k) = std::min(0.014, wk_RH(k) * qvs);
        
        wk_thd(k) = wk_tht(k) * (1 + epsb * qv_e(0, 0, k));
        
        pk_e(i_r, j_r, k) = wk_pi(k);
        tht_e(i_r, j_r, k) = wk_tht(k);
      }
    }
   
    //wk_p(0) = 1e5;
    //for (int k = k_r.first() + 1; k <= k_r.last(); ++k)
    //{
    //  auto del_th = wk_tht(k) - wk_tht(k - 1);
    //  // copied from eulag, why ?
    //  auto inv_th = del_th > 1e-4 ? std::log(wk_tht(k) / wk_tht(k - 1)) / del_th : 1. / wk_tht(k);
    //  wk_p(k) = std::pow(std::pow(wk_p(k - 1), R_d_over_c_pd_v) - p.g * std::pow(1e5, R_d_over_c_pd_v) * inv_th * dz / cp
    //               ,
    //               1./ R_d_over_c_pd_v);
    //}
    
    //for (int k = k_r.first(); k <= k_r.last(); ++k)
    //{
    //  wk_T(k) = wk_tht(k) * std::pow(wk_p(k) / 1e5, R_d_over_c_pd_v);

    //}
  

    //for (int k = k_r.first(); k <= k_r.last(); ++k)
    //{
    //  tht_e(i_r, j_r, k) = wk_tht(k);
    //  pk_e(i_r, j_r, k) = std::pow(wk_p(k) / 1e5, R_d_over_c_pd_v);

    //  // this was inconsistent !
    //  //      const real_t f2x = 17.27;
    //  //      const T xk = 0.2875;
    //  //      const real_t psl = 1000.0;
    //  //      const T pc = 3.8 / (std::pow(pk_e(0, 0, k), 1. / xk) * psl);
    //  //      const real_t qvs = pc * std::exp(f2x * (pk_e(0, 0, k) * tht_e(0, 0, k) - 273.)
    //  //                                            / (pk_e(0, 0, k) * tht_e(0, 0, k)- 36.));
    //  //

    //  T pk = pk_e(0, 0, k);
    //  T p = wk_p(k);
    //  T th = tht_e(0, 0, k);
    //  T Temp = th * pk;


    //  T es = e0 * std::exp(L / Rv * ((Temp - T0) / (T0 * Temp)));
    //  T qvs = eps * es / (p - es);
    //  qv_e(i_r, j_r, k) = std::min(0.014, wk_RH(k) * qvs);
    //}
    
    const auto& tht_b = slv.sclr_array("tht_b");
    const auto& rho_b = slv.g_factor();

    
    T stab = 0.0;
    for (int k = k_r.first() + 1; k <= k_r.last() - 1; ++k)
    {
      stab += (wk_tht(k + 1) - wk_tht(k - 1)) / (2 * dz * wk_tht(k));
    }
    stab /= (nz - 2);

    stab = 1.235e-5;
    std::cout << "stability: " << stab << std::endl;

    const T cs_v = p.g / (cp * 300. * stab);

    const T rho0 = 1e5 / (Rd * 300.);

    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      T z = k * dz;
      tht_b(i_r, j_r, k) = 300 * std::exp(stab * z);
      rho_b(i_r, j_r, k) = rho0 * std::exp(-stab * z) * std::pow(1 - cs_v * (1 - std::exp(-stab * z)), 1. / R_d_over_c_pd_v - 1);
    }
    
    //for (int k = k_r.first(); k <= k_r.last(); ++k)
    //{
    //  T z = k * dz;
    //  tht_b(i_r, j_r, k) = 300;
    //  rho_b(i_r, j_r, k) = 1.11 * std::pow(1. - R_d_over_c_pd_v * z / 7.e3, 1. / R_d_over_c_pd_v - 1);
    //}

    slv.advectee(ix::v) = 0;
    slv.advectee(ix::w) = 0;
    slv.advectee(ix::qc) = 0;
    slv.advectee(ix::qr) = 0;
    
    slv.advectee(ix::qv)(i_r, j_r, k_r) = qv_e(i_r, j_r, k_r);

    if (ct_params_t::impl_tht)
    {
      slv.advectee(ix::tht)(i_r, j_r, k_r) = 0;
    }
    else
    {
      slv.advectee(ix::tht)(i_r, j_r, k_r) = tht_e(i_r, j_r, k_r);
    }

    T U_s = 30, U_c = 15;
    T z_s = 5e3, dz_u = 1e3;

    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      T z = k * dz;
      T u_e = -10000;
      if (z < z_s - dz_u)
      {
        u_e = U_s * z / z_s - U_c;
      }
      else if (std::abs(z - z_s) <= dz_u)
      {
        u_e = (-4. / 5 + 3 * z / z_s - 5. / 4 * std::pow(z / z_s, 2)) * U_s - U_c;
      }
      else if (z > z_s + dz_u)
      {
        u_e = U_s - U_c;
      }
      //u_e = 0.;
      slv.advectee(ix::u)(i_r, j_r, k) = u_e;
      slv.sclr_array("u_e")(i_r, j_r, k) = u_e;
    }

  const T r0x = 10e3;
  const T r0y = r0x;
  const T r0z = 1.5e3;

  const T x0 = 0.5 * length_x;
  const T y0 = 0.5 * length_y;
  const T z0 = r0z;
  
  decltype(slv.advectee()) rr(slv.advectee().shape()), delta(slv.advectee().shape());
    {
      blitz::firstIndex i;
      blitz::secondIndex j;
      blitz::thirdIndex k;

      rr = sqrt(pow2((i * dx - x0) / r0x) + pow2((j * dy - y0) / r0y) + pow2((k * dz - z0) / r0z)); 

      delta = where(rr(i, j, k) <= 1.0,
                    3 * pow2(cos(0.5 * pi * rr(i, j, k))),
                    0.0);
        
      slv.advectee(ix::tht) += delta(i, j, k);
    
      //const T z_abs = 15000;
      //slv.vab_coefficient() = where(k * dz >= z_abs,
      //                               1. / 100 * (k * dz - z_abs) / (length_z - z_abs),
      //                               0);
      slv.vab_coefficient() = 0.;
    }
    
    slv.vab_relaxed_state(0)(i_r, j_r, k_r) = slv.advectee(ix::u)(i_r, j_r, k_r);
    slv.vab_relaxed_state(1) = 0;
    slv.vab_relaxed_state(2) = 0;
  
    //for (int k = k_r.first(); k <= k_r.last(); ++k)
    //{
    //  std::cout
    //    << k * dz << ' '
    //    << wk_p(k) << ' '
    //    << wk_tht(k) << ' '
    //    << wk_T(k) << ' '
    //    << wk_RH(k) << ' '
    //    << qv_e(0, 0, k) << ' '
    //    << slv.advectee(ix::u)(0, 0, k) << ' '
    //    << tht_b(0, 0, k) << ' '
    //    << rho_b(0, 0, k)
    //    << std::endl;
    //}
    
    std::cout.precision(18);
    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      std::cout << k * dz << ' '
        << qv_e(0, 0, k) << ' '
        << wk_T(k) << ' '
        << slv.advectee(ix::u)(0, 0, k) << ' '
        << wk_tht(k) << std::endl;
    }
    
    std::cout << "sum qv_e: " << sum(qv_e(0, 0, k_r)) << std::endl;
    rng_t ki_r(1, nz - 2);
    std::cout << "sum test: " <<   0.5 * qv_e(0, 0, 0) * rho_b(0, 0, 0)
                                 + sum(qv_e(0, 0, ki_r) * rho_b(0, 0, ki_r))
                                 + 0.5 * qv_e(0, 0, nz - 1) * rho_b(0, 0, nz - 1)
                                       << std::endl;
  }
  

  std::cout << "Calculating: " << p.outdir << std::endl; 

  std::cout << "MIN U   " << min(slv.advectee(ix::u)) << std::endl; 
  std::cout << "MIN V   " << min(slv.advectee(ix::v)) << std::endl; 
  std::cout << "MIN W   " << min(slv.advectee(ix::w)) << std::endl; 
  std::cout << "MIN THT " << min(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "MIN QV  " << min(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "MIN QC  " << min(slv.advectee(ix::qc)) << std::endl; 
  std::cout << "MIN QR  " << min(slv.advectee(ix::qr)) << std::endl; 

  std::cout << "MAX U   " << max(slv.advectee(ix::u)) << std::endl; 
  std::cout << "MAX V   " << max(slv.advectee(ix::v)) << std::endl; 
  std::cout << "MAX W   " << max(slv.advectee(ix::w)) << std::endl; 
  std::cout << "MAX THT " << max(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "MAX QV  " << max(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "MAX QC  " << max(slv.advectee(ix::qc)) << std::endl; 
  std::cout << "MAX QR  " << max(slv.advectee(ix::qr)) << std::endl; 

  // integration
  if (ct_params_t::var_dt)
  {
    slv.advance(sim_time);
  }
  else
  {
    slv.advance(nt);
  }

  std::cout.precision(18);
  std::cout << "MIN U   " << min(slv.advectee(ix::u)) << std::endl; 
  std::cout << "MIN W   " << min(slv.advectee(ix::w)) << std::endl; 
  std::cout << "MIN THT " << min(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "MIN QV  " << min(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "MIN QC  " << min(slv.advectee(ix::qc)) << std::endl; 
  std::cout << "MIN QR  " << min(slv.advectee(ix::qr)) << std::endl; 

  std::cout << "MAX U   " << max(slv.advectee(ix::u)) << std::endl; 
  std::cout << "MAX W   " << max(slv.advectee(ix::w)) << std::endl; 
  std::cout << "MAX THT " << max(slv.advectee(ix::tht)) << std::endl; 
  std::cout << "MAX QV  " << max(slv.advectee(ix::qv)) << std::endl; 
  std::cout << "MAX QC  " << max(slv.advectee(ix::qc)) << std::endl; 
  std::cout << "MAX QR  " << max(slv.advectee(ix::qr)) << std::endl; 
}

int main() 
{
  std::vector<int> nps = {168 / 2}; 
  //std::vector<int> nps = {168 / 4, 168 / 2, 168, 168 * 2};
  for (const auto np : nps)
  {
    //test(500, np, "out_mdiff_cdt");
    //test(0, np, "out_iles_cdt");
    //test(500, np, "out_pdiff_cdt5");
    //test(500, np, "out_pdiff_vdt9");
    //test(0, np, "out_iles_cdt10_wojtek");
    //test<false>(500, np, "out_pdiff_vdt8_cmpct");
    //test<true>(500, np, "out_pdiff_cdt5_simple");
    //test<true>(500, np, "out_pdiff_vdt97_simple");
    //test<true>(500, np, "out_pdiff_cdt10_simple_rfrc");
    test<false>(500, np, "out_piotr_cdt5_noprec_upw");
    //test<false>(500, np, "out_piotr_cdt5_fixes");
  }
}
