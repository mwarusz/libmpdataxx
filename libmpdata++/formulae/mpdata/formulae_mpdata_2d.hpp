/** @file
* @copyright University of Warsaw
* @section LICENSE
* GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
*/

#pragma once

#include <libmpdata++/formulae/mpdata/formulae_mpdata_common.hpp>
#include <libmpdata++/formulae/mpdata/formulae_mpdata_common_2d.hpp>
#include <libmpdata++/formulae/mpdata/formulae_mpdata_hot_2d.hpp>
#include <libmpdata++/formulae/mpdata/formulae_mpdata_dfl_2d.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>

namespace libmpdataxx 
{ 
  namespace formulae 
  { 
    namespace mpdata 
    {
      //eq. (29a) from @copybrief Smolarkiewicz_and_Margolin_1998
      template <opts_t opts, int dim, class arr_2d_t>
      inline auto antidiff(
        const arr_2d_t &psi, 
        const arrvec_t<arr_2d_t> &GC,
        const arr_2d_t &G, 
        const rng_t &i, 
        const rng_t &j,
        const double mu
      ) return_macro(,
        // second order terms
        abs(GC[dim](pi<dim>(i+h, j))) 
        * (1 - abs(GC[dim](pi<dim>(i+h, j))) / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j)) / 2
        * dpsi_dx<opts BOOST_PP_COMMA() dim>(psi, i, j) 
        - 
        GC[dim](pi<dim>(i+h, j)) 
        * GC1_bar_xy<dim>(GC[dim-1], i, j)
        / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j) / 2
        * dpsi_dy<opts BOOST_PP_COMMA() dim>(psi, i, j)
        // third order terms
        + HOT<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        // divergent flow correction
        + DFL<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        - mu * dpsi_dx<opts BOOST_PP_COMMA() dim>(psi, i, j) 
      ) 

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto antidiff_corr( // antidiffusive velocity correction
        const arr_2d_t &psi, 
        const arrvec_t<arr_2d_t> &GC,
        const arrvec_t<arr_2d_t> &dGC_dt,
        const arrvec_t<arr_2d_t> &dGC_dtt,
        const arrvec_t<arr_2d_t> &GC_corr,
        const arr_2d_t &G, 
        const rng_t &i, 
        const rng_t &j,
        double mu
      ) return_macro(,
        GC_corr[dim](pi<dim>(i+h, j))

        + aux<opts BOOST_PP_COMMA() dim>(psi, GC_corr[dim], i, j)

        //+ abs(GC_corr[dim](pi<dim>(i+h, j))) / 2 * dpsi_dx<opts BOOST_PP_COMMA() dim>(psi, i, j) 
        //+ HOT<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j) // ordinary higher order term

         //dimensional terms
        - 1.0 / 24 *
        (
            4 * GC[dim](pi<dim>(i+h, j)) * dpsi_dxx<opts BOOST_PP_COMMA() dim>(psi, i, j)
          + 2 * dpsi_dx<opts BOOST_PP_COMMA() dim>(psi, i, j) * dGC0_dx<dim>(GC[dim], i, j)
          + 4 * dGC0_dxx<opts BOOST_PP_COMMA() dim>(psi, GC[dim], i, j)
        )
        
        // mixed terms
        + 0.5 * abs(GC[dim](pi<dim>(i+h, j))) * grad_gdiv<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        
        // temporal terms
        + 1.0 / 24 *
        (
            - 8 * GC[dim](pi<dim>(i+h, j)) *  gdiv_gdiv<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
            + 10 * dGC0_dtt<opts BOOST_PP_COMMA() dim>(psi, dGC_dtt[dim], i, j)
            + 2 * GC[dim](pi<dim>(i+h, j)) *  gdiv<opts BOOST_PP_COMMA() dim>(psi, dGC_dt, G, i, j)
            - 2 * dGC_dt[dim](pi<dim>(i+h, j)) * gdiv<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        )
      )
    } // namespace mpdata
  } // namespace formulae
} // namespcae libmpdataxx 
