/** @file
* @copyright University of Warsaw
* @section LICENSE
* GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
*/

#pragma once

#include <libmpdata++/formulae/mpdata/formulae_mpdata_common.hpp>
#include <libmpdata++/formulae/mpdata/formulae_mpdata_common_3d.hpp>
#include <libmpdata++/formulae/mpdata/formulae_mpdata_hot_3d.hpp>
#include <libmpdata++/formulae/mpdata/formulae_mpdata_dfl_3d.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>

namespace libmpdataxx
{
  namespace formulae
  {
    namespace mpdata
    {
      // for definition of A and B see eq. 17 in @copybrief Smolarkiewicz_and_Margolin_1998
      template<opts_t opts, int d, class arr_3d_t>
      inline auto A(  // positive sign scalar version
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	  frac<opts>(
	    psi(pi<d>(i+1, j, k)) - psi(pi<d>(i, j, k))
	  , //-----------------------------------------
	    psi(pi<d>(i+1, j, k)) + psi(pi<d>(i, j, k))
	  )
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto A(  // variable-sign scalar version
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	  frac<opts>(
	    abs(psi(pi<d>(i+1, j, k))) - abs(psi(pi<d>(i, j, k)))
	  , //---------------------------------------------------
	    abs(psi(pi<d>(i+1, j, k))) + abs(psi(pi<d>(i, j, k)))
	  )
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto A(  // inf. gauge option
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "iga & abs options are mutually exclusive");
        ,
	(
	  psi(pi<d>(i+1, j, k)) - psi(pi<d>(i, j, k))
	)
	/ //-----------------------------------------
	(
	  1 + 1
	)
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto B1( // positive sign signal
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	  frac<opts>(
	    psi(pi<d>(i+1, j+1, k)) + psi(pi<d>(i, j+1, k)) - psi(pi<d>(i+1, j-1, k)) - psi(pi<d>(i, j-1, k))
	  , //-----------------------------------------------------------------------------------------------
	    psi(pi<d>(i+1, j+1, k)) + psi(pi<d>(i, j+1, k)) + psi(pi<d>(i+1, j-1, k)) + psi(pi<d>(i, j-1, k))
	  ) / 2
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto B1( // variable-sign signal
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	  frac<opts>(
	    abs(psi(pi<d>(i+1, j+1, k)))
          + abs(psi(pi<d>(i, j+1, k)))
          - abs(psi(pi<d>(i+1, j-1, k)))
          - abs(psi(pi<d>(i, j-1, k)))
	  , //--------------------------
	    abs(psi(pi<d>(i+1, j+1, k)))
          + abs(psi(pi<d>(i, j+1, k)))
          + abs(psi(pi<d>(i+1, j-1, k)))
          + abs(psi(pi<d>(i, j-1, k)))
	  ) / 2
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto B1( // inf. gauge
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "iga & abs options are mutually exclusive");
        ,
	(
	  psi(pi<d>(i+1, j+1, k)) + psi(pi<d>(i, j+1, k)) - psi(pi<d>(i+1, j-1, k)) - psi(pi<d>(i, j-1, k))
	)
	/ //-----------------------------------------------------------------------------------------------
	(
	  1 + 1 + 1 +1
	) / 2
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto B2( // positive sign signal
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	  frac<opts>(
	    psi(pi<d>(i+1, j, k+1)) + psi(pi<d>(i, j, k+1)) - psi(pi<d>(i+1, j, k-1)) - psi(pi<d>(i, j, k-1))
	  , //-----------------------------------------------------------------------------------------------
	    psi(pi<d>(i+1, j, k+1)) + psi(pi<d>(i, j, k+1)) + psi(pi<d>(i+1, j, k-1)) + psi(pi<d>(i, j, k-1))
	  ) / 2
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto B2( // variable-sign signal
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	  frac<opts>(
	    abs(psi(pi<d>(i+1, j, k+1)))
          + abs(psi(pi<d>(i, j, k+1)))
          - abs(psi(pi<d>(i+1, j, k-1)))
          - abs(psi(pi<d>(i, j, k-1)))
	  , //--------------------------
	    abs(psi(pi<d>(i+1, j, k+1)))
          + abs(psi(pi<d>(i, j, k+1)))
          + abs(psi(pi<d>(i+1, j, k-1)))
          + abs(psi(pi<d>(i, j, k-1)))
	  ) / 2
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto B2( // inf. gauge
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "iga & abs options are mutually exclusive");
        ,
	(
	  psi(pi<d>(i+1, j, k+1)) + psi(pi<d>(i, j, k+1)) - psi(pi<d>(i+1, j, k-1)) - psi(pi<d>(i, j, k-1))
	)
	/ //-----------------------------------------------------------------------------------------------
	(
	  1 + 1 + 1 + 1
	) / 2
      )
     
      template <opts_t opts, int dim, class arr_3d_t>
      inline auto antidiff(
        const arr_3d_t &psi,
        const arrvec_t<arr_3d_t> &GC,
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
          // second order terms
          abs(GC[dim](pi<dim>(i+h, j, k)))
        * (1 - abs(GC[dim](pi<dim>(i+h, j, k))) / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k))
        * A<opts BOOST_PP_COMMA() dim>(psi, i, j, k)
        - GC[dim](pi<dim>(i+h, j, k))
        * (
          GC1_bar_xy<dim>(GC[dim+1], i, j, k)
          / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k)
          * B1<opts BOOST_PP_COMMA() dim>(psi, i, j, k)
          + GC2_bar_xz<dim>(GC[dim-1], i, j, k)
          / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k)
          * B2<opts BOOST_PP_COMMA() dim>(psi, i, j, k)
          )
          // third order terms
        + HOT<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k)
        + DFL<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k)
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto antidiff_corr( // antidiffusive velocity correction
        const arr_3d_t &psi, 
        const arrvec_t<arr_3d_t> &GC,
        const arrvec_t<arr_3d_t> &dGC_dt,
        const arrvec_t<arr_3d_t> &dGC_dtt,
        const arrvec_t<arr_3d_t> &GC_corr,
        const arr_3d_t &G, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        GC_corr[dim](pi<dim>(i+h, j, k))

        + aux<opts BOOST_PP_COMMA() dim>(psi, GC_corr[dim], i, j, k)
        
        //dimensional terms
        - 1.0 / 24 *
        (
            4 * GC[dim](pi<dim>(i+h, j, k)) * dpsi_dxx<opts BOOST_PP_COMMA() dim>(psi, i, j, k)
          + 2 * dpsi_dx<opts BOOST_PP_COMMA() dim>(psi, i, j, k) * dGC0_dx<dim>(GC[dim], i, j, k)
          + 1 * dGC0_dxx<opts BOOST_PP_COMMA() dim>(psi, GC[dim], i, j, k)
        )
        
        // mixed terms
        + 0.5 * abs(GC[dim](pi<dim>(i+h, j, k))) * grad_gdiv<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k)
        
        // temporal terms
        + 1.0 / 24 *
        (
            - 8 * GC[dim](pi<dim>(i+h, j, k)) *  gdiv_gdiv<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k)
            //+ 1 * dGC0_dtt<opts BOOST_PP_COMMA() dim>(psi, dGC_dtt[dim], i, j, k)
            //+ 2 * GC[dim](pi<dim>(i+h, j, k)) *  gdiv<opts BOOST_PP_COMMA() dim>(psi, dGC_dt, G, i, j, k)
            //- 2 * dGC_dt[dim](pi<dim>(i+h, j,k )) * gdiv<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k)
        )
      )
    } // namespace mpdata
  } // namespace formulae
} // namespcae libmpdataxx
