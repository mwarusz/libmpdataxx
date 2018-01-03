/**
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */
#include <libmpdata++/solvers/shallow_water_sphere.hpp>
#include <libmpdata++/concurr/threads.hpp>
#include <libmpdata++/output/hdf5_xdmf.hpp>
#include <boost/math/constants/constants.hpp>
using namespace libmpdataxx; 

using real_t = double;
constexpr auto pi = boost::math::constants::pi<real_t>();

template <int opts_arg>
void test(const std::string &outdir) 
{
  struct ct_params_t : ct_params_default_t
  {
    using real_t = ::real_t;
    enum { n_dims = 2 };
    enum { n_eqns = 3 };
    
    enum { opts = opts_arg};
    enum { rhs_scheme = solvers::trapez };

    struct ix { 
      enum { h = 0, qx, qy };
      enum { vip_i=qx, vip_j=qy, vip_den=h };
    };  
    
    // hints
    enum { hint_norhs = opts::bit(ix::h) }; 
  };

  const real_t dt = 20;
  const real_t day = 24 * 3600;
  const int 
    nt = (15 * day) / dt,
    outfreq = nt / 15;

  const real_t R = 6371.22e3;
  const real_t F0 = 1.4584e-4;
  //const real_t F0 = 0.;

  using ix = typename ct_params_t::ix;

  // solver choice
  using slv_out_t = 
    output::hdf5_xdmf<
      solvers::shallow_water_sphere<ct_params_t>
    >;

  // run-time parameters
  typename slv_out_t::rt_params_t p; 

  p.grid_size = {129, 64};
  p.dt = dt;
  p.di = p.dj = pi / p.grid_size[1];
  p.outfreq = outfreq;
  p.outdir = outdir;
  p.outvars[ix::h].name = "h";
  p.outvars[ix::qx].name = "qx";
  p.outvars[ix::qy].name = "qy";
  p.vip_eps = 1e-8;
  p.n_iters = 2;

  // instantiation
  concurr::threads<
    slv_out_t, 
    bcond::cyclic, bcond::cyclic,
    bcond::polar, bcond::polar
  > run(p); 

  {
    blitz::firstIndex i;
    blitz::secondIndex j;

    // coordinates
    decltype(run.advectee()) X(run.advectee().extent()), Y(run.advectee().extent());
    X = i * p.di;
    Y = (j + 0.5) * p.dj - pi / 2;

    // geometry
    run.sclr_array("hx") = R * cos(Y(i, j));
    run.sclr_array("hy") = R;

    const auto& hx = run.sclr_array("hx");
    const auto& hy = run.sclr_array("hy");

    run.sclr_array("dhx2y") = - 0.5 * R * R * sin(2 * Y(i, j)) / (hx(i, j) * hx(i, j) * hy(i, j));
    run.g_factor() = hx(i, j) * hy(i, j);

    constexpr bool rhw = false;
 
    if (rhw)
    {
      decltype(run.advectee()) ath(run.advectee().extent()),
                               bth(run.advectee().extent()),
                               cth(run.advectee().extent());

      const real_t A = R;
      const real_t om = 7.848e-6;
      const real_t K = 7.848e-6;
      const real_t RR = 4;
      const real_t PH0 = 78.4e3;

      ath = om * 0.5 * (F0 + om) * pow(cos(Y(i, j)), 2)
                  + 0.25 * K * K * pow(cos(Y(i, j)), 2 * RR) * ( (RR + 1) * pow(cos(Y(i, j)), 2)
                  + (2 * RR * RR - RR - 2) - 2 * RR * RR / pow(cos(Y(i, j)), 2)
                  );
      
      bth = (F0 + 2 * om) * K / ((RR + 1) * (RR + 2)) * pow(cos(Y(i, j)), RR) *
            ((RR * RR + 2 * RR + 2) - pow((RR + 1) * cos(Y(i, j)), 2));

      cth = 0.25 * K * K * pow(cos(Y(i, j)), 2 * RR) * ((RR + 1) * pow(cos(Y(i, j)), 2) - RR - 2);

      // initial condition
      run.advectee(ix::h) = (
                              PH0 + A * A * (ath(i, j) + bth(i, j) * cos(RR * X(i, j)) + cth(i, j) * cos(2 * RR * X(i, j)))
                            ) / p.g;

      run.advectee(ix::qx) = (
                                A * om * cos(Y(i, j)) + A * K * cos(RR * X(i, j)) * 
                                pow(cos(Y(i, j)), RR - 1) * (RR * pow(sin(Y(i, j)), 2) - pow(cos(Y(i, j)), 2))
                             ) * run.advectee(ix::h)(i, j);

      run.advectee(ix::qy) = (
                                - A * K * RR * pow(cos(Y(i, j)), RR - 1) * sin(Y(i, j)) * sin(RR * X(i, j))
                             ) * run.advectee(ix::h)(i, j);
      
      run.sclr_array("cor") = F0 * sin(Y(i,j));
    }
    else
    {
      const auto h00 = 8.0e3 * p.g;
      const auto beta = 0.0;
      const auto Q = 20./ R;

      decltype(run.advectee()) dist(run.advectee().extent());

      const auto x0 = 3 * pi / 2;
      const auto y0 = pi / 6;
      const auto rad = pi / 9;
      const auto hscale = 2.e3;

      dist = 2. * sqrt(pow(cos(Y(i,j)) * sin(0.5 * (X(i,j) - x0)), 2) + pow(sin(0.5 * (Y(i,j) - y0)), 2));

      run.sclr_array("h0") = hscale * where(dist(i,j) < rad, 1 - dist(i,j) / rad, 0.0);

      //std::cout << run.sclr_array("h0")  << std::endl;

      run.advectee(ix::h) = (
                              h00 - R * R * (F0 + Q) * 0.5 * Q * pow(
                              -cos(X(i, j)) * cos(Y(i, j)) * sin(beta) + sin(Y(i, j)) * cos(beta)
                              ,
                              2
                              )
                            ) / p.g - run.sclr_array("h0")(i, j);
      
      run.advectee(ix::qx) = (
                                Q * (cos(beta) + tan(Y(i, j)) * cos(X(i, j)) * sin(beta)) * R * cos(Y(i, j))
                             ) * run.advectee(ix::h)(i, j);
      
      run.advectee(ix::qy) = (
                               -Q * sin(X(i, j)) * sin(beta) * R
                             ) * run.advectee(ix::h)(i, j);
      
      run.sclr_array("cor") = F0 * (-cos(X(i, j)) * cos(Y(i, j)) * sin(beta) + sin(Y(i, j)) * cos(beta));
    }

    std::cout << "initial state" << std::endl;
    std::cout << "h:  " << min(run.advectee(ix::h)) << ' ' << max(run.advectee(ix::h)) << std::endl;
    std::cout << "qx: " << min(run.advectee(ix::qx)) << ' ' << max(run.advectee(ix::qx)) << std::endl;
    std::cout << "qy: " << min(run.advectee(ix::qy)) << ' ' << max(run.advectee(ix::qy)) << std::endl;
    std::cout << "umax: " << max(run.advectee(ix::qx) / run.advectee(ix::h)) << std::endl;
  }
 

  auto i_r = blitz::Range(0, 127);
  auto j_r = blitz::Range(0, 63);
  const auto &gf = run.g_factor()(i_r, j_r)      ;
  const auto &h  = run.advectee(ix::h)(i_r, j_r) ;
  const auto &qx = run.advectee(ix::qx)(i_r, j_r);
  const auto &qy = run.advectee(ix::qy)(i_r, j_r);

  auto init_en = sum(gf * ((pow(qx, 2) + pow(qy, 2)) / h + p.g * h * h));
  
  run.advance(nt);

  std::cout << "umax: " << max(run.advectee(ix::qx) / run.advectee(ix::h)) << std::endl;

  auto en = sum(gf * ((pow(qx, 2) + pow(qy, 2)) / h + p.g * h * h));

  std::cout << "en error: " << (en - init_en) / init_en << std::endl;
};

int main()
{
  //test<opts::nug | opts::iga | opts::div_2nd | opts::fct>("out_ifc");
  test<opts::nug | opts::iga | opts::div_2nd | opts::div_3rd | opts::fct>("out_ifc_div3");
}
