/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */

#include <libmpdata++/solvers/mpdata_rhs_vip_prs_sgs.hpp>
#include <libmpdata++/concurr/threads.hpp>
#include <libmpdata++/output/hdf5_xdmf.hpp>
#include <boost/math/constants/constants.hpp>

using namespace libmpdataxx;
using boost::math::constants::pi;

template <int opt_arg, int iters_arg, int prs_ord_arg>
void test(const std::string &dirname)
{
  const int nx = 65, ny = 65, nz = 65;

  struct ct_params_t : ct_params_default_t
  {
    using real_t = double;
    enum { prs_order = prs_ord_arg };
    enum { var_dt = false };
    enum { n_dims = 3 };
    enum { opts = opt_arg};
    enum { n_eqns = 3 };
    enum { rhs_scheme = solvers::trapez };
    enum { prs_scheme = solvers::cr };
    struct ix { enum {
      u, v, w,
      vip_i=u, vip_j=v, vip_k=w, vip_den=-1
    }; };
  
    enum { hint_norhs = opts::bit(ix::u) | opts::bit(ix::v) | opts::bit(ix::w)}; 
  }; 

  using ix = typename ct_params_t::ix;

  using solver_t = output::hdf5_xdmf<libmpdataxx::solvers::mpdata_rhs_vip_prs<ct_params_t>>;

  typename solver_t::rt_params_t p;

  p.n_iters = iters_arg;

  double time = 10.0;
  p.dt = 0.02;
  p.max_courant = 1.0;
  int nt = time / p.dt;

  p.di = 2 * pi<typename ct_params_t::real_t>() / (nx - 1);
  p.dj = 2 * pi<typename ct_params_t::real_t>() / (ny - 1);
  p.dk = 2 * pi<typename ct_params_t::real_t>() / (nz - 1);

  p.outfreq = nt; 
  p.outwindow = 1;
  p.outvars = {
    {ix::u,   {.name = "u",   .unit = "m/s"}}, 
    {ix::v,   {.name = "v",   .unit = "m/s"}}, 
    {ix::w,   {.name = "w",   .unit = "m/s"}}, 
  };
  p.outdir = dirname;
  p.prs_tol = 1e-6;
  p.grid_size = {nx, ny, nz};

  libmpdataxx::concurr::threads<
    solver_t, 
    bcond::cyclic, bcond::cyclic,
    bcond::cyclic, bcond::cyclic,
    bcond::cyclic, bcond::cyclic
  > slv(p);

  {
    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;

    slv.advectee(ix::u) =  sin(p.di * i) * cos(p.dj * j) * cos(p.dk * k);
    slv.advectee(ix::v) = -cos(p.di * i) * sin(p.dj * j) * cos(p.dk * k); 
    slv.advectee(ix::w) = 0.0; 
  }

  slv.advance(nt); 
};

int main()
{

  //{
  //  const int prs_order = 2;
  //  test<opts::iga | opts::fct, 2, prs_order>("out_iga_fct_prs2");
  //  test<opts::iga | opts::tot | opts::fct, 2, prs_order>("out_iga_tot_fct_prs2");
  //  test<opts::iga | opts::div_2nd | opts::div_3rd | opts::fct, 2, prs_order>("out_iga_div3_fct_prs2");

  //  test<opts::iga, 2, prs_order>("out_iga_prs2");
  //  test<opts::iga | opts::tot, 2, prs_order>("out_iga_tot_prs2");
  //  test<opts::iga | opts::div_2nd | opts::div_3rd, 2, prs_order>("out_iga_div3_prs2");
  //  
  //  test<opts::abs, 2, prs_order>("out_abs_prs2");
  //  test<opts::abs | opts::tot, 3, prs_order>("out_abs_tot_prs2");
  //  test<opts::abs | opts::div_2nd | opts::div_3rd, 2, prs_order>("out_abs_div3_prs2");
  //}
  
  {
    const int prs_order = 4;
    //test<opts::iga | opts::fct, 2, prs_order>("out_iga_fct_prs4");
    //test<opts::iga | opts::tot | opts::fct, 2, prs_order>("out_iga_tot_fct_prs4");
    //test<opts::iga | opts::div_2nd | opts::div_3rd | opts::fct, 2, prs_order>("out_iga_div3_fct_prs4");

    test<opts::iga, 2, prs_order>("out_iga_prs4");
    //test<opts::iga | opts::tot, 2, prs_order>("out_iga_tot_prs4");
    //test<opts::iga | opts::div_2nd | opts::div_3rd, 2, prs_order>("out_iga_div3_prs4");
    //
    //test<opts::abs, 2, prs_order>("out_abs_prs4");
    //test<opts::abs | opts::tot, 3, prs_order>("out_abs_tot_prs4");
    //test<opts::abs | opts::div_2nd | opts::div_3rd, 2, prs_order>("out_abs_div3_prs4");
  }
}
