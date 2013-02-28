/** @file
  * @copyright University of Warsaw
  * @section LICENSE
  * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
  */
/** @mainpage
  *
  * @section INTRODUCTION
  *
  * libmpdata++ is C++ library of parallel, object-oriented implementations
  * of the MPDATA family solvers of generalised transport equations 
  * of the form:
  *
  * \f$ \partial_t \psi + \nabla \cdot (\vec{v} \psi) = R \f$
  *
  * The theory behind MPDATA solvers was developed by Piotr Smolarkiewicz et al.
  * (see e.g. @copydetails Smolarkiewicz_2006, for a review and list of references).
  * Development of libmpdata++ is carried out by Sylwester Arabas, Anna Jaruga and
  * co-workers at the [Institute of Geophysics](http://www.igf.fuw.edu.pl/),
  * [Faculty of Physics](http://www.fuw.edu.pl/),
  * [University of Warsaw](http://www.uw.edu.pl/) (the copyright holder)
  * with funding from the Polish [National Science Centre](http://www.ncn.gov.pl/).
  *
  * libmpdata++ is based on the Blitz++ library for high-performance array handling.
  * libmpdata++ (and Blitz++) is a header-only library.
  * 
  * @section sec_concepts KEY CONCEPTS
  *
  * Shortest example:
  * TODO
  * 
  * Library "layers":
  * - solvers
  * - boundary conditions
  * - concurency schemes
  * - output mechanisms
  *
  * At present libmpdata++ handles 1D, 2D and 3D computations in cartesian coordinates on an Arakawa-C grid.
  *
  * @section sec_solvers SUPPORTED SOLVER TYPES
  * 
  * @subsection sec_solvers_homo HOMOGENEOUS TRANSPORT EQUATIONS WITH PRESCRIBED VELOCITY
  *
  * \f$ \partial_t \psi + \nabla (\vec{u} \psi) = 0 \f$
  *
  * See:
  * - solver classes:
  *   - donorcell_1d.hpp
  *   - donorcell_2d.hpp
  *   - donorcell_3d.hpp
  *   - mpdata_1d.hpp
  *   - mpdata_2d.hpp
  * - examples:
  *   - test_gnuplot-iostream_1d.cpp 
  *   - test_gnuplot-iostream_2d.cpp 
  *   - test_var_sign_2d.cpp 
  *
  * @subsection sec_solvers_inhomo INHOMOGENEOUS TRANSPORT EQUATIONS WITH PRESCRIBED VELOCITY
  *
  * a pair of coupled 1-dimensional harmonic oscillators:
  *
  * \f$ \partial_t \psi + \nabla (\vec{u} \psi) =  \omega \phi \f$
  *
  * \f$ \partial_t \phi + \nabla (\vec{u} \phi) = -\omega \psi \f$
  *
  * See:
  * - examples:
  *   - test_harmosc.cpp (system defined in coupled_harmosc.hpp)
  * 
  * @subsection sec_solvers_inhomo_vel INHOMOGENEOUS TRANSPORT EQUATIONS 
  *
  * 1-dimensional shallow-water equations:
  * 
  * \f$ \partial_t u + u \cdot \nabla_z u = - \frac{1}{\rho} \nabla_z p \f$ 
  * 
  * \f$ \partial_t h + \nabla_z ( \vec{u} h ) = 0 \f$
  *
  * @subsection sec_solvers_inhomo_dyn_pres INHOMOGENEOUS TRANSPORT EQUATIONS COUPLED WITH PRESSURE SOLVER
  *
  * 2-dimensional Navier-Stokes coupled assuming adiabatic transport with constant density:
  *
  * \f$ \partial_t \theta + \nabla (\vec{u} \theta) = 0 \f$
  * 
  * \f$ \partial_t u + \nabla (\vec{u} u) = -\partial_x \frac{p-p_e}{\rho} \f$
  *
  * \f$ \partial_t w + \nabla (\vec{u} w) = -\partial_z \frac{p-p_e}{\rho} + g \frac{\theta - \theta_e}{\theta_e} \f$
  *
  * See:
  * - pressure solver classes:
  *   - Richardson solver 
  *   - Minimum-Residual solver (solver_pressure_mr.hpp)
  *   - Conjugate-Residual solver (solver_pressure_cr.hpp)
  *   - Conjugate-Residual with preconditioner (solver_pressure_pc.hpp)
  * - examples:
  *   - test_bombel.cpp (system defined in bombel.hpp)
  *
  * @section sec_concurr SUPPORTES CONCURENCY SCHEMES
  * 
  * Shared-memory parallelisation:
  * - OpenMP (openmp.hpp) 
  * - Boost.Thread (boost_thread.hpp)
  * - Threads (threads.hpp - OpenMP if supported, Boost.Thread otherwise)
  *
  * @section sec_output SUPPORTES OUTPUT MECHANISMS
  * - gnuplot-iostream
  */

  /*
  *  suggested compiler options (by compiler): -march=native, -Ofast, -DNDEBUG, -lblitz (opt), -DBZDEBUG (opt), -std=c++11
  */
