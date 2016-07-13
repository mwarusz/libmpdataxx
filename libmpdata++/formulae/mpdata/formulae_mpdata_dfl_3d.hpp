/** @file
* @copyright University of Warsaw
* @section LICENSE
* GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
*/

#pragma once

#include <libmpdata++/formulae/mpdata/formulae_mpdata_common.hpp>
#include <libmpdata++/formulae/mpdata/formulae_mpdata_common_3d.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>

namespace libmpdataxx
{ 
  namespace formulae 
  { 
    namespace mpdata 
    {

      //divergent flow correction see eq. (30) from @copybrief Smolarkiewicz_and_Margolin_1998)
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto DFL(
        const arr_3d_t &psi,
        const arrvec_t<arr_3d_t> &GC,
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::dfl)>::type* = 0 
      ) -> decltype(0)
      { 
        return 0;  
      }

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto DFL(
        const arr_3d_t &psi,    //to have the same arguments as in iga option
        const arrvec_t<arr_3d_t> &GC,      
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::dfl) && !opts::isset(opts, opts::iga)>::type* = 0 
      ) return_macro(,
        - 0.25 * GC[dim](pi<dim>(i+h, j, k)) 
        /
        G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k)
        * 
        (
          (
            GC[dim](pi<dim>((i+1)+h, j, k)) - 
            GC[dim](pi<dim>(i-h    , j, k))
          )
          +
          (
            GC[dim+1](pi<dim>(i+1, j+h, k)) + 
            GC[dim+1](pi<dim>(i,   j+h, k)) -
            GC[dim+1](pi<dim>(i+1, j-h, k)) - 
            GC[dim+1](pi<dim>(i,   j-h, k))
          )
          +
          (
            GC[dim-1](pi<dim>(i+1, j, k+h)) + 
            GC[dim-1](pi<dim>(i,   j, k+h)) -
            GC[dim-1](pi<dim>(i+1, j, k-h)) - 
            GC[dim-1](pi<dim>(i,   j, k-h))
          )
        )
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto DFL(
        const arr_3d_t &psi,
        const arrvec_t<arr_3d_t> &GC,
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::dfl) && opts::isset(opts, opts::iga)>::type* = 0 
      ) return_macro(,
        - 0.25 * GC[dim](pi<dim>(i+h, j, k)) 
        /
        G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k)
        * 
        (
          (
            GC[dim](pi<dim>((i+1)+h, j, k)) - 
            GC[dim](pi<dim>(i-h    , j, k))
          )
          +
          (
            GC[dim+1](pi<dim>(i+1, j+h, k)) + 
            GC[dim+1](pi<dim>(i,   j+h, k)) -
            GC[dim+1](pi<dim>(i+1, j-h, k)) - 
            GC[dim+1](pi<dim>(i,   j-h, k))
          )
          +
          (
            GC[dim-1](pi<dim>(i+1, j, k+h)) + 
            GC[dim-1](pi<dim>(i,   j, k+h)) -
            GC[dim-1](pi<dim>(i+1, j, k-h)) - 
            GC[dim-1](pi<dim>(i,   j, k-h))
          )
        )
        *
        0.5 *  (psi(pi<dim>(i+1, j, k)) + psi(pi<dim>(i, j, k)))  //to be compatible with iga formulation
      )
      
    } // namespace mpdata
  } // namespace formulae
} // namespcae libmpdataxx
