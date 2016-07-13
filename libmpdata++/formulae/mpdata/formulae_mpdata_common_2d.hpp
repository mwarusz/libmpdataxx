/** @file
* @copyright University of Warsaw
* @section LICENSE
* GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
*/

#pragma once

#include <libmpdata++/formulae/mpdata/formulae_mpdata_common.hpp>

namespace libmpdataxx 
{ 
  namespace formulae 
  { 
    namespace mpdata 
    {
      //G at (i+h, j)
      template<opts_t opts, int dim, class arr_2d_t>
      inline typename arr_2d_t::T_numtype G_bar_x(
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == false
      ) {
        return 1;
      }

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto G_bar_x( 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == true
      ) return_macro(,
        (
          formulae::G<opts, dim>(G, i+1, j) 
          +
          formulae::G<opts, dim>(G, i  , j)
        ) / 2
      )
      
      template<opts_t opts, int dim, class arr_2d_t>
      inline typename arr_2d_t::T_numtype G_bar_xy(
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == false
      ) {
        return 1;
      }
      
      template<opts_t opts, int dim, class arr_2d_t>
      inline auto G_bar_xy(
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == true
      ) return_macro(,
        (
          formulae::G<opts, dim>(G, i  , j) +
          formulae::G<opts, dim>(G, i  , j+1) +
          formulae::G<opts, dim>(G, i+1, j) +
          formulae::G<opts, dim>(G, i+1, j+1)
        ) / 4
      )
      
      
      template<opts_t opts, int d, class arr_2d_t>
      inline auto dpsi_dx(  // positive sign scalar version
        const arr_2d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        2 *
	frac<opts>(
	  psi(pi<d>(i+1, j)) - psi(pi<d>(i, j))
	  ,// --------------------------------
	  psi(pi<d>(i+1, j)) + psi(pi<d>(i, j))
	)
      )

      template<opts_t opts, int d, class arr_2d_t>
      inline auto dpsi_dx(  // variable-sign scalar version
        const arr_2d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        2 *
	frac<opts>(
	  abs(psi(pi<d>(i+1, j))) - abs(psi(pi<d>(i, j)))
	  ,// -------------------------------------------
	  abs(psi(pi<d>(i+1, j))) + abs(psi(pi<d>(i, j)))
	) 
      ) 

      template<opts_t opts, int d, class arr_2d_t>
      inline auto dpsi_dx(  // inf. gauge option
        const arr_2d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "abs & iga options are mutually exclusive");
        ,
        2 *
        (
	  psi(pi<d>(i+1, j)) - psi(pi<d>(i, j))
        ) / ( //---------------------
	  1 + 1
        )
      )
      
      template<opts_t opts, int d, class arr_2d_t>
      inline auto dpsi_dy( // positive sign signal
        const arr_2d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	frac<opts>( 
	  psi(pi<d>(i+1, j+1)) + psi(pi<d>(i, j+1)) - psi(pi<d>(i+1, j-1)) - psi(pi<d>(i, j-1))
	  ,// --------------------------------------------------------------------------------
	  psi(pi<d>(i+1, j+1)) + psi(pi<d>(i, j+1)) + psi(pi<d>(i+1, j-1)) + psi(pi<d>(i, j-1))
	)
      )

      template<opts_t opts, int d, class arr_2d_t>
      inline auto dpsi_dy( // variable-sign signal
        const arr_2d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	frac<opts>( 
	  abs(psi(pi<d>(i+1, j+1))) + abs(psi(pi<d>(i, j+1))) - abs(psi(pi<d>(i+1, j-1))) - abs(psi(pi<d>(i, j-1)))
	  ,// ----------------------------------------------------------------------------------------------------
	  abs(psi(pi<d>(i+1, j+1))) + abs(psi(pi<d>(i, j+1))) + abs(psi(pi<d>(i+1, j-1))) + abs(psi(pi<d>(i, j-1)))
	)
      )

      template<opts_t opts, int d, class arr_2d_t>
      inline auto dpsi_dy( // inf. gauge
        const arr_2d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "abs & iga options are mutually exclusive");
        ,
        (
	  psi(pi<d>(i+1, j+1)) + psi(pi<d>(i, j+1)) - psi(pi<d>(i+1, j-1)) - psi(pi<d>(i, j-1))
	) / (  // --------------------------------------------------------------------------------
	  1 + 1 + 1 + 1
        )
      )
      
      template<opts_t opts, int dim, class arr_2d_t>
      inline auto dpsi_dxx( // positive sign signal
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
          2 *
          frac<opts>(
            psi(pi<dim>(i+2, j)) - psi(pi<dim>(i+1, j)) - psi(pi<dim>(i, j)) + psi(pi<dim>(i-1, j))
            ,//-----------------------------------------------------------------------------------
            psi(pi<dim>(i+2, j)) + psi(pi<dim>(i+1, j)) + psi(pi<dim>(i, j)) + psi(pi<dim>(i-1, j))
        )
      )

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto dpsi_dxx( // variable sign signal
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
          2 *
          frac<opts>(
            abs(psi(pi<dim>(i+2, j))) - abs(psi(pi<dim>(i+1, j))) - abs(psi(pi<dim>(i, j))) + abs(psi(pi<dim>(i-1, j)))
            ,//-------------------------------------------------------------------------------------------------------
            abs(psi(pi<dim>(i+2, j))) + abs(psi(pi<dim>(i+1, j))) + abs(psi(pi<dim>(i, j))) + abs(psi(pi<dim>(i-1, j)))
        )
      )

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto dpsi_dxx( // inf. gauge option
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "iga & abs are mutually exclusive");
        ,
        2 *
	( psi(pi<dim>(i+2, j)) - psi(pi<dim>(i+1, j)) - psi(pi<dim>(i, j)) + psi(pi<dim>(i-1, j)) )
	/ //--------------------------------------------------------------------------------------
	(1 + 1 + 1 + 1)
      )

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto dpsi_dxy( // positive sign signal
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
          2 *
          frac<opts>(
            psi(pi<dim>(i+1, j+1)) - psi(pi<dim>(i, j+1)) - psi(pi<dim>(i+1, j-1)) + psi(pi<dim>(i, j-1))            
            ,//-----------------------------------------------------------------------------------------
            psi(pi<dim>(i+1, j+1)) + psi(pi<dim>(i, j+1)) + psi(pi<dim>(i+1, j-1)) + psi(pi<dim>(i, j-1))
        )
      )

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto dpsi_dxy( // variable sign signal
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
          2 *
          frac<opts>(
            abs(psi(pi<dim>(i+1, j+1))) - abs(psi(pi<dim>(i, j+1))) - abs(psi(pi<dim>(i+1, j-1))) + abs(psi(pi<dim>(i, j-1)))            
            ,//-------------------------------------------------------------------------------------------------------------
            abs(psi(pi<dim>(i+1, j+1))) + abs(psi(pi<dim>(i, j+1))) + abs(psi(pi<dim>(i+1, j-1))) + abs(psi(pi<dim>(i, j-1)))
        )
      )

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto dpsi_dxy( // inf. gauge option
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "iga & abs options are mutually exclusive");
        ,
        2 *
	(psi(pi<dim>(i+1, j+1)) - psi(pi<dim>(i, j+1)) - psi(pi<dim>(i+1, j-1)) + psi(pi<dim>(i, j-1)))            
	/ //-------------------------------------------------------------------------------------------
	(1 + 1 + 1 + 1)
      )
      
      template<opts_t opts, int dim, class arr_2d_t>
      inline auto psi_bar_x( 
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i+1, j)) + 
          psi(pi<dim>(i, j))
        ) / 2
      )
      
      template<opts_t opts, int dim, class arr_2d_t>
      inline auto psi_bar_x( 
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i+1, j))) + 
          abs(psi(pi<dim>(i, j)))
        ) / 2
      )
      
      // psi at (i, j+1/2)
      template<opts_t opts, int dim, class arr_2d_t>
      inline auto psi_bar_y( 
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i, j)) + 
          psi(pi<dim>(i, j+1))
        ) / 2
      )

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto psi_bar_y( 
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i, j))) + 
          abs(psi(pi<dim>(i, j+1)))
        ) / 2
      )
      
      // psi at (i+1/2, j+1/2)
      template<opts_t opts, int dim, class arr_2d_t>
      inline auto psi_bar_xy( 
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i+1, j)) + 
          psi(pi<dim>(i, j))   +
          psi(pi<dim>(i, j+1)) +
          psi(pi<dim>(i+1, j+1))
        ) / 4
      )

      template<opts_t opts, int dim, class arr_2d_t>
      inline auto psi_bar_xy( 
        const arr_2d_t &psi,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i+1, j))) + 
          abs(psi(pi<dim>(i, j)))   +
          abs(psi(pi<dim>(i, j+1))) +
          abs(psi(pi<dim>(i+1, j+1)))
        ) / 4
      )
      
      // GC[1] at (i+h, j)
      template<int dim, class arr_2d_t>
      inline auto GC1_bar_xy( // caution proper call looks like GC1_bar<dim>(GC[dim+1], i, j) - note dim vs dim+1
        const arr_2d_t &GC, 
        const rng_t &i, 
        const rng_t &j
      ) return_macro(,
        (
          GC(pi<dim>(i+1, j+h)) + 
          GC(pi<dim>(i,   j+h)) +
          GC(pi<dim>(i+1, j-h)) + 
          GC(pi<dim>(i,   j-h)) 
        ) / 4
      )
      
      // GC[0] at (i, j+h)
      template<int dim, class arr_2d_t>
      inline auto GC0_bar_xy(
        const arr_2d_t &GC, 
        const rng_t &i, 
        const rng_t &j
      ) return_macro(,
        (
          GC(pi<dim>(i+h, j)) + 
          GC(pi<dim>(i-h, j)) +
          GC(pi<dim>(i+h, j+1)) + 
          GC(pi<dim>(i-h, j+1)) 
        ) / 4
      )
      
      // GC[0] at (i, j)
      template<int dim, class arr_2d_t>
      inline auto GC0_bar_x( 
        const arr_2d_t &GC,
        const rng_t &i,
        const rng_t &j
      ) return_macro(,
        (
          GC(pi<dim>(i+h, j)) + 
          GC(pi<dim>(i-h, j))
        ) / 2
      )
      
      // GC[1] at (i+h, j+h)
      template<int dim, class arr_2d_t>
      inline auto GC1_bar_x( 
        const arr_2d_t &GC,
        const rng_t &i,
        const rng_t &j
      ) return_macro(,
        (
          GC(pi<dim>(i+1, j+h)) + 
          GC(pi<dim>(i, j+h))
        ) / 2
      )

      template <int dim, class arr_2d_t>
      inline auto dGC0_dx(
        const arr_2d_t &GC, 
        const rng_t &i,
        const rng_t &j
      ) return_macro(,
        (GC(pi<dim>(i+h+1, j)) - GC(pi<dim>(i+h-1, j))) / 2
      )
      
      template <opts_t opts, int dim, class arr_2d_t>
      inline auto dGC0_dxx(
        const arr_2d_t &psi, 
        const arr_2d_t &GC, 
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        GC(pi<dim>(i+h+1, j)) + GC(pi<dim>(i+h-1, j)) - 2 * GC(pi<dim>(i+h, j))
      )
      
      template <opts_t opts, int dim, class arr_2d_t>
      inline auto dGC0_dxx(
        const arr_2d_t &psi, 
        const arr_2d_t &GC, 
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        (GC(pi<dim>(i+h+1, j)) + GC(pi<dim>(i+h-1, j)) - 2 * GC(pi<dim>(i+h, j))) * 
         psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j)
      )
      
      template <opts_t opts, int dim, class arr_2d_t>
      inline auto dGC0_dtt(
        const arr_2d_t &psi, 
        const arr_2d_t &dGC_dtt, 
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        dGC_dtt(pi<dim>(i+h, j)) + 0
      )
      
      template <opts_t opts, int dim, class arr_2d_t>
      inline auto dGC0_dtt(
        const arr_2d_t &psi, 
        const arr_2d_t &dGC_dtt, 
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        dGC_dtt(pi<dim>(i+h, j)) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j)
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto gdiv1(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j
      ) return_macro(,
        (
          GC[dim](pi<dim>(i+h, j)) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j)
        - GC[dim](pi<dim>(i-h, j)) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i-1, j)
        + GC[dim+1](pi<dim>(i, j+h)) * psi_bar_y<opts BOOST_PP_COMMA() dim>(psi, i, j)
        - GC[dim+1](pi<dim>(i, j-h)) * psi_bar_y<opts BOOST_PP_COMMA() dim>(psi, i, j-1)
        ) / formulae::G<opts BOOST_PP_COMMA() dim>(G, i, j)
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto gdiv2(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j
      ) return_macro(,
        (
            GC0_bar_xy<dim>(GC[dim], i+1, j) * psi_bar_y<opts BOOST_PP_COMMA() dim>(psi, i+1, j)
          - GC0_bar_xy<dim>(GC[dim], i  , j) * psi_bar_y<opts BOOST_PP_COMMA() dim>(psi, i, j)
            
          + GC1_bar_xy<dim>(GC[dim+1], i, j+1) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j+1)
          - GC1_bar_xy<dim>(GC[dim+1], i, j  ) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j)
        ) / G_bar_xy<opts BOOST_PP_COMMA() dim>(G, i, j)
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        6 *
        frac<opts>(
          GC0_bar_x<dim>(GC[dim], i+1, j) * psi(pi<dim>(i+1, j))
        - GC0_bar_x<dim>(GC[dim], i  , j) * psi(pi<dim>(i, j))
        + GC1_bar_x<dim>(GC[dim+1], i, j  ) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j - 1)
        ,
          G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j) * (
            psi(pi<dim>(i+1, j  ))
          + psi(pi<dim>(i  , j  ))
          + psi(pi<dim>(i  , j+1))
          + psi(pi<dim>(i+1, j+1))
          + psi(pi<dim>(i  , j-1))
          + psi(pi<dim>(i+1, j-1))
          )
        )
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        6 *
        frac<opts>(
          GC0_bar_x<dim>(GC[dim], i+1, j) * abs(psi(pi<dim>(i+1, j)))
        - GC0_bar_x<dim>(GC[dim], i  , j) * abs(psi(pi<dim>(i, j)))
        + GC1_bar_x<dim>(GC[dim+1], i, j  ) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j - 1)
        ,
          G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j) * (
            abs(psi(pi<dim>(i+1, j  )))
          + abs(psi(pi<dim>(i  , j  )))
          + abs(psi(pi<dim>(i  , j+1)))
          + abs(psi(pi<dim>(i+1, j+1)))
          + abs(psi(pi<dim>(i  , j-1)))
          + abs(psi(pi<dim>(i+1, j-1)))
          )
        )
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        (
          GC0_bar_x<dim>(GC[dim], i+1, j) * psi(pi<dim>(i+1, j))
        - GC0_bar_x<dim>(GC[dim], i  , j) * psi(pi<dim>(i, j))
        + GC1_bar_x<dim>(GC[dim+1], i, j  ) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j - 1)
        ) / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j)
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto gdiv_gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        8 *
        frac<opts>(
          GC0_bar_x<dim>(GC[dim], i+1, j) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j)
        - GC0_bar_x<dim>(GC[dim], i, j) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        + GC1_bar_x<dim>(GC[dim+1], i, j) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j-1)
        ,
          G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j) * (
            psi(pi<dim>(i+2,   j)) +
            psi(pi<dim>(i+1,   j)) +
            psi(pi<dim>(i  ,   j)) +
            psi(pi<dim>(i-1,   j)) +
            psi(pi<dim>(i  , j+1)) +
            psi(pi<dim>(i  , j-1)) +
            psi(pi<dim>(i+1, j+1)) +
            psi(pi<dim>(i+1, j-1))
          )
        )
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto gdiv_gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        8 *
        frac<opts>(
          GC0_bar_x<dim>(GC[dim], i+1, j) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j)
        - GC0_bar_x<dim>(GC[dim], i, j) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        + GC1_bar_x<dim>(GC[dim+1], i, j) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j-1)
        ,
          G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j) * (
            abs(psi(pi<dim>(i+2,   j))) +
            abs(psi(pi<dim>(i+1,   j))) +
            abs(psi(pi<dim>(i  ,   j))) +
            abs(psi(pi<dim>(i-1,   j))) +
            abs(psi(pi<dim>(i  , j+1))) +
            abs(psi(pi<dim>(i  , j-1))) +
            abs(psi(pi<dim>(i+1, j+1))) +
            abs(psi(pi<dim>(i+1, j-1)))
          )
        )
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto gdiv_gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        (
          GC0_bar_x<dim>(GC[dim], i+1, j) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j)
        - GC0_bar_x<dim>(GC[dim], i, j) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        + GC1_bar_x<dim>(GC[dim+1], i, j) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j-1)
        ) / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j)
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto grad_gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        8 * 
        frac<opts>(
          gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j)
        - gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        ,
          psi(pi<dim>(i+2,   j)) +
          psi(pi<dim>(i+1,   j)) +
          psi(pi<dim>(i  ,   j)) +
          psi(pi<dim>(i-1,   j)) +
          psi(pi<dim>(i  , j+1)) +
          psi(pi<dim>(i  , j-1)) +
          psi(pi<dim>(i+1, j+1)) +
          psi(pi<dim>(i+1, j-1))
        )
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto grad_gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        8 * 
        frac<opts>(
          gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j)
        - gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
        ,
          abs(psi(pi<dim>(i+2, j  ))) +
          abs(psi(pi<dim>(i+1, j  ))) +
          abs(psi(pi<dim>(i  , j  ))) +
          abs(psi(pi<dim>(i-1, j  ))) +
          abs(psi(pi<dim>(i  , j+1))) +
          abs(psi(pi<dim>(i  , j-1))) +
          abs(psi(pi<dim>(i+1, j+1))) +
          abs(psi(pi<dim>(i+1, j-1)))
        )
      )
      
      template <opts_t opts, int dim, class arr_2d_t, class arrvec_t>
      inline auto grad_gdiv(
        const arr_2d_t &psi,
        const arrvec_t &GC, 
        const arr_2d_t &G,
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
          gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j)
        - gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j)
      )

      template <opts_t opts, int dim, class arr_2d_t>
      inline auto aux(
        const arr_2d_t &psi,
        const arr_2d_t &GC_corr, 
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<!opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
          abs(GC_corr(pi<dim>(i+h, j))) / 2 * dpsi_dx<opts BOOST_PP_COMMA() dim>(psi, i, j) 
      )
      
      template <opts_t opts, int dim, class arr_2d_t>
      inline typename arr_2d_t::T_numtype aux(
        const arr_2d_t &psi,
        const arr_2d_t &GC_corr, 
        const rng_t &i,
        const rng_t &j,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) {
        return 0;
      }

    } // namespace mpdata
  } // namespace formulae
} // namespcae libmpdataxx 
