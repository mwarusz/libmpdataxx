/* 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */

#include <cmath>

#include "transforms.hpp"

#include <libmpdata++/solvers/mpdata.hpp>
#include <libmpdata++/concurr/threads.hpp>
#include <libmpdata++/output/hdf5_xdmf.hpp>

using namespace libmpdataxx;
template <int opts_arg, int opts_iters>
void test(const std::string filename, int nlat)
{
  std::cout << "Calculating: " << filename << std::endl;

  struct ct_params_t : ct_params_default_t
  {
    using real_t = double;
    enum { n_dims = 2 };
    enum { n_eqns = 1 };
    enum { opts = opts_arg };
  };
  
  int nlon = 2 * nlat + 1;
  T dx = 2 * pi / (nlon - 1), dy = pi / nlat;

  T time = 12;
  T dt = dx / (48 * 2 * pi);
  int nt = time / dt;

  using slv_out_t = 
      output::hdf5_xdmf<
        solvers::mpdata<ct_params_t>
    >;
  typename slv_out_t::rt_params_t p;

  p.n_iters = opts_iters; 
  p.grid_size = {nlon, nlat};

  p.outfreq = nt / 12; 
  p.outvars[0].name = "psi";
  p.outdir = filename;

  concurr::threads<
    slv_out_t, 
    bcond::cyclic, bcond::cyclic,
    bcond::polar, bcond::polar
  > run(p); 
  
  T
    x0 = 3 * pi / 2,
    y0 = 0,
    u0 = 2 * pi / 12,
    a = pi / 2;

  xpf_t xpf{.x0 = x0, .y0 = y0};
  ypf_t ypf{.x0 = x0, .y0 = y0};

  blitz::firstIndex i;
  blitz::secondIndex j;
  
  decltype(run.advectee()) solution(run.advectee().extent());

  {
    decltype(run.advectee()) X(run.advectee().extent()), Y(run.advectee().extent());
    X = i * dx;
    Y = (j + 0.5) * dy - pi / 2;

    decltype(run.advectee()) r(run.advectee().extent());

    r = acos(sin(y0) * sin(Y) + cos(y0) * cos(Y) * cos(X - x0));

    run.advectee() = where(r <= 1.0 / 3, 500 * (1 + cos(3 * pi * r)), 0.);
    run.g_factor() = dx * dy * blitz::cos(Y);
    
    solution = run.advectee();

    decltype(run.advectee()) C(run.advectee().extent());
    C =   abs(u0 * (cos(a) + tan(Y) * sin(a) * cos(X)) * dt / dx) 
        + abs(- u0 * sin(X) * sin(a) * dt / dy );
    std::cout << "max Courant number: " << max(C) << std::endl;

    run.advector(0) =   u0 * (cos(a) + tan(Y) * sin(a) * cos(X - 0.5 * dx)) * dt / dx * dx * dy * cos(Y);

    run.advector(1) = - u0 * sin(X) * sin(a) * dt / dy * dx * dy * cos(Y + 0.5 * dy);
  }

  run.advance(nt);

  T Linf = max(abs(run.advectee() - solution)) / max(abs(solution));
  T L1 = sum(run.g_factor() * abs(run.advectee() - solution)) / sum(run.g_factor() * abs(solution));
  T L2 = sqrt(sum(run.g_factor() * pow2(run.advectee() - solution)) / sum(run.g_factor() * pow2(solution)));

  std::cout << "real time:\t" << nt * dt << std::endl;
  std::cout << "Linf:\t" << Linf << std::endl;
  std::cout << "L1:\t" << L1 << std::endl;
  std::cout << "L2:\t" << L2 << std::endl;
}

int main()
{
  {
    enum { opts = opts::nug};
    const int opts_iters = 2;
    test<opts, opts_iters>("sb_zro_024_2", 24);
    test<opts, opts_iters>("sb_zro_048_2", 48);
    test<opts, opts_iters>("sb_zro_096_2", 96);
    test<opts, opts_iters>("sb_zro_192_2", 192);
    test<opts, opts_iters>("sb_zro_384_2", 384);
    test<opts, opts_iters>("sb_zro_768_2", 768);
  }
}
