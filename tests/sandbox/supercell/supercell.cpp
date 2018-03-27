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
    enum { n_eqns = 7 };
    enum { rhs_scheme = solvers::trapez };
    enum { prs_scheme = solvers::cr };
    struct ix { enum {
      u, v, w, qv, qc, qr, tht,
      vip_i=u, vip_j=v, vip_k=w, vip_den=-1
    }; };
  }; 
  using ix = typename ct_params_t::ix;
  using real_t = typename ct_params_t::real_t;

  const int nx = 257, ny = 257, nz = 41, nt = 2400;

  using slv_out_t = output::hdf5_xdmf<supercell<ct_params_t>>;
  // run-time parameters
  slv_out_t::rt_params_t p;

  T dx = 500;
  T dy = dx;
  T dz = 500;

  p.dt = 3.0;
  p.di = dx;
  p.dj = dy;
  p.dk = dz; 
  p.prs_tol = 1e-6;
  p.grid_size = {nx, ny, nz};
  p.n_iters = 2;
  
  p.buoy_filter = true;

  p.outfreq = 100;
  p.outvars = {
    {ix::qv, {"qv", "?"  }},
    {ix::qc, {"qc", "?"  }},
    {ix::qr, {"qr", "?"  }},
    {ix::tht, {"tht",  "?"  }},
    {ix::u, {"u", "?"  }},
    {ix::v, {"v", "?"  }},
    {ix::w, {"w", "?"  }}
  };
  p.outdir = "out";

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

    blitz::Array<T, 1> wk_tht(nz), wk_RH(nz), wk_p(nz), wk_T(nz);
   
    const T cp = 1004.5;

    const T z_tr = 12e3;
    const T tht_0 = 300;
    const T tht_tr = 343;
    const T T_tr = 213;

    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      T z = k * dz;
      if (z < z_tr)
      {
        wk_tht(k) = tht_0 + (tht_tr - tht_0) * std::pow(z / z_tr, 5./4);
        wk_RH(k) = 1 - 0.75 * std::pow(z / z_tr, 5./4); 
      }
      else
      {
        wk_tht(k) = tht_tr * std::exp(p.g / (cp * T_tr) * (z - z_tr));
        wk_RH(k) = 0.25;
      }
    }
   
    T R_d_over_c_pd_v = 287. / cp;
    wk_p(0) = 1e5;
    for (int k = k_r.first() + 1; k <= k_r.last(); ++k)
    {
      auto del_th = wk_tht(k) - wk_tht(k - 1);
      // copied from eulag, why ?
      auto inv_th = del_th > 1e-4 ? std::log(wk_tht(k) / wk_tht(k - 1)) / del_th : 1. / wk_tht(k);
      wk_p(k) = std::pow(std::pow(wk_p(k - 1), R_d_over_c_pd_v) - p.g * std::pow(1e5, R_d_over_c_pd_v) * inv_th * dz / cp
                   ,
                   1./ R_d_over_c_pd_v);
    }
    
    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      wk_T(k) = wk_tht(k) * std::pow(wk_p(k) / 1e5, R_d_over_c_pd_v);
    }
  
    const auto& tht_e = slv.sclr_array("tht_e");
    const auto& pk_e = slv.sclr_array("pk_e");
    const auto& qv_e = slv.sclr_array("qv_e");

    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      tht_e(i_r, j_r, k) = wk_tht(k);
      pk_e(i_r, j_r, k) = std::pow(wk_p(k) / 1e5, R_d_over_c_pd_v);

      const real_t f2x = 17.27;
      const T xk = 0.2875;
      const real_t psl = 1000.0;
      const T pc = 3.8 / (std::pow(pk_e(0, 0, k), 1. / xk) * psl);
      const real_t qvs = pc * std::exp(f2x * (pk_e(0, 0, k) * tht_e(0, 0, k) - 273.)
                                            / (pk_e(0, 0, k) * tht_e(0, 0, k)- 36.));

      qv_e(i_r, j_r, k) = std::min(0.014, wk_RH(k) * qvs);
    }
    
    const auto& tht_b = slv.sclr_array("tht_b");
    const auto& rho_b = slv.g_factor();

    const T stab = 1.020e-5;
    const T cs_v = p.g / (cp * 300. * stab);
    for (int k = k_r.first(); k <= k_r.last(); ++k)
    {
      T z = k * dz;
      tht_b(i_r, j_r, k) = 300 * std::exp(stab * z);
      rho_b(i_r, j_r, k) = 1.11 * std::exp(-stab * z) * std::pow(1 - cs_v * (1 - std::exp(-stab * z)), 1. / R_d_over_c_pd_v - 1);
    }

    slv.advectee(ix::v) = 0;
    slv.advectee(ix::w) = 0;
    slv.advectee(ix::qc) = 0;
    slv.advectee(ix::qr) = 0;
    
    slv.advectee(ix::qv)(i_r, j_r, k_r) = qv_e(i_r, j_r, k_r);
    slv.advectee(ix::tht)(i_r, j_r, k_r) = tht_e(i_r, j_r, k_r);

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
      slv.advectee(ix::u)(i_r, j_r, k) = u_e;
    }

  const T r0x = 10e3;
  const T r0y = r0x;
  const T r0z = 1.5e3;

  const T x0 = 64e3;
  const T y0 = 64e3;
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
    }
  
  }

  std::cout.precision(18);
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
  slv.advance(nt);  

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
