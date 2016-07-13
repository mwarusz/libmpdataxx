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
      template<opts_t opts, int dim, class arr_3d_t>
      inline typename arr_3d_t::T_numtype G_bar_x( 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == false
      ) {
          return 1;
      }

      //G evaluated at (i+1/2, j, k)
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto G_bar_x(
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == true
      ) return_macro(,
          (
            formulae::G<opts, dim>(G, i+1, j, k)
          + formulae::G<opts, dim>(G, i  , j, k)
          ) / 2
      )
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline typename arr_3d_t::T_numtype G_bar_xy(
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == false
      ) {
        return 1;
      }
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto G_bar_xy(
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == true
      ) return_macro(,
        (
          formulae::G<opts, dim>(G, i  , j  , k) +
          formulae::G<opts, dim>(G, i  , j+1, k) +
          formulae::G<opts, dim>(G, i+1, j  , k) +
          formulae::G<opts, dim>(G, i+1, j+1, k)
        ) / 4
      )
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline typename arr_3d_t::T_numtype G_bar_xz(
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == false
      ) {
        return 1;
      }
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto G_bar_xz(
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::nug)>::type* = 0 // enabled if nug == true
      ) return_macro(,
        (
          formulae::G<opts, dim>(G, i  , j, k  ) +
          formulae::G<opts, dim>(G, i  , j, k+1) +
          formulae::G<opts, dim>(G, i+1, j, k  ) +
          formulae::G<opts, dim>(G, i+1, j, k+1)
        ) / 4
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dx(  // positive sign scalar version
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        2 *
	frac<opts>(
	  psi(pi<d>(i+1, j, k)) - psi(pi<d>(i, j, k))
	  ,
	  psi(pi<d>(i+1, j, k)) + psi(pi<d>(i, j, k))
	)
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dx(  // variable-sign scalar version
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        2 *
	frac<opts>(
	  abs(psi(pi<d>(i+1, j, k))) - abs(psi(pi<d>(i, j, k)))
	  ,
	  abs(psi(pi<d>(i+1, j, k))) + abs(psi(pi<d>(i, j, k)))
	) 
      ) 

      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dx(  // inf. gauge option
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "abs & iga options are mutually exclusive");
        ,
	psi(pi<d>(i+1, j, k)) - psi(pi<d>(i, j, k))
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dy( // positive sign signal
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	frac<opts>( 
	    psi(pi<d>(i+1, j+1, k))
          + psi(pi<d>(i  , j+1, k))
          - psi(pi<d>(i+1, j-1, k))
          - psi(pi<d>(i  , j-1, k))
	  ,
	    psi(pi<d>(i+1, j+1, k))
          + psi(pi<d>(i  , j+1, k))
          + psi(pi<d>(i+1, j-1, k))
          + psi(pi<d>(i  , j-1, k))
	)
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dy( // variable-sign signal
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	frac<opts>( 
	    abs(psi(pi<d>(i+1, j+1, k)))
          + abs(psi(pi<d>(i  , j+1, k)))
          - abs(psi(pi<d>(i+1, j-1, k)))
          - abs(psi(pi<d>(i  , j-1, k)))
	  ,
	    abs(psi(pi<d>(i+1, j+1, k)))
          + abs(psi(pi<d>(i  , j+1, k)))
          + abs(psi(pi<d>(i+1, j-1, k)))
          + abs(psi(pi<d>(i  , j-1, k)))
	)
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dy( // inf. gauge
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "abs & iga options are mutually exclusive");
        ,
        (
	  psi(pi<d>(i+1, j+1, k))
        + psi(pi<d>(i  , j+1, k))
        - psi(pi<d>(i+1, j-1, k))
        - psi(pi<d>(i  , j-1, k))
        ) / 4
      )
      
      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dz( // positive sign signal
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	frac<opts>( 
	    psi(pi<d>(i+1, j, k+1))
          + psi(pi<d>(i  , j, k+1))
          - psi(pi<d>(i+1, j, k-1))
          - psi(pi<d>(i  , j, k-1))
	  ,
	    psi(pi<d>(i+1, j, k+1))
          + psi(pi<d>(i  , j, k+1))
          + psi(pi<d>(i+1, j, k-1))
          + psi(pi<d>(i  , j, k-1))
	)
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dz( // variable-sign signal
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
	frac<opts>( 
	    abs(psi(pi<d>(i+1, j, k+1)))
          + abs(psi(pi<d>(i  , j, k+1)))
          - abs(psi(pi<d>(i+1, j, k-1)))
          - abs(psi(pi<d>(i  , j, k-1)))
	  ,
	    abs(psi(pi<d>(i+1, j, k+1)))
          + abs(psi(pi<d>(i  , j, k+1)))
          + abs(psi(pi<d>(i+1, j, k-1)))
          + abs(psi(pi<d>(i  , j, k-1)))
	)
      )

      template<opts_t opts, int d, class arr_3d_t>
      inline auto dpsi_dz( // inf. gauge
        const arr_3d_t &psi, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "abs & iga options are mutually exclusive");
        ,
        (
	  psi(pi<d>(i+1, j, k+1))
        + psi(pi<d>(i  , j, k+1))
        - psi(pi<d>(i+1, j, k-1))
        - psi(pi<d>(i  , j, k-1))
        ) / 4
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto dpsi_dxx( // positive sign signal
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
          2 *
          frac<opts>(
              psi(pi<dim>(i+2, j, k))
            - psi(pi<dim>(i+1, j, k))
            - psi(pi<dim>(i  , j, k))
            + psi(pi<dim>(i-1, j, k))
            ,
              psi(pi<dim>(i+2, j, k))
            + psi(pi<dim>(i+1, j, k))
            + psi(pi<dim>(i  , j, k))
            + psi(pi<dim>(i-1, j, k))
        )
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto dpsi_dxx( // variable sign signal
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
          2 *
          frac<opts>(
              abs(psi(pi<dim>(i+2, j, k)))
            - abs(psi(pi<dim>(i+1, j, k)))
            - abs(psi(pi<dim>(i  , j, k)))
            + abs(psi(pi<dim>(i-1, j, k)))
            ,
              abs(psi(pi<dim>(i+2, j, k)))
            + abs(psi(pi<dim>(i+1, j, k)))
            + abs(psi(pi<dim>(i  , j, k)))
            + abs(psi(pi<dim>(i-1, j, k)))
        )
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto dpsi_dxx( // inf. gauge option
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "iga & abs are mutually exclusive");
        ,
        2 * (
              psi(pi<dim>(i+2, j, k))
            - psi(pi<dim>(i+1, j, k))
            - psi(pi<dim>(i  , j, k))
            + psi(pi<dim>(i-1, j, k)) )
        / 4
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto dpsi_dxy( // positive sign signal
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
          2 *
          frac<opts>(
              psi(pi<dim>(i+1, j+1, k))
            - psi(pi<dim>(i  , j+1, k))
            - psi(pi<dim>(i+1, j-1, k))
            + psi(pi<dim>(i  , j-1, k))            
            ,
              psi(pi<dim>(i+1, j+1, k))
            + psi(pi<dim>(i  , j+1, k))
            + psi(pi<dim>(i+1, j-1, k))
            + psi(pi<dim>(i  , j-1, k))
        )
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto dpsi_dxy( // variable sign signal
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
          2 *
          frac<opts>(
              abs(psi(pi<dim>(i+1, j+1, k)))
            - abs(psi(pi<dim>(i  , j+1, k)))
            - abs(psi(pi<dim>(i+1, j-1, k)))
            + abs(psi(pi<dim>(i  , j-1, k)))            
            ,
              abs(psi(pi<dim>(i+1, j+1, k)))
            + abs(psi(pi<dim>(i  , j+1, k)))
            + abs(psi(pi<dim>(i+1, j-1, k)))
            + abs(psi(pi<dim>(i  , j-1, k)))
        )
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto dpsi_dxy( // inf. gauge option
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(
        static_assert(!opts::isset(opts, opts::abs), "iga & abs options are mutually exclusive");
        ,
        2 * (
              psi(pi<dim>(i+1, j+1, k))
            - psi(pi<dim>(i  , j+1, k))
            - psi(pi<dim>(i+1, j-1, k))
            + psi(pi<dim>(i  , j-1, k))            
        ) / 4
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_x( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i+1, j, k)) + 
          psi(pi<dim>(i  , j, k))
        ) / 2
      )
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_x( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i+1, j, k))) + 
          abs(psi(pi<dim>(i  , j, k)))
        ) / 2
      )
      
      // psi at (i, j+1/2)
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_y( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i, j  , k)) + 
          psi(pi<dim>(i, j+1, k))
        ) / 2
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_y( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i, j  , k))) + 
          abs(psi(pi<dim>(i, j+1, k)))
        ) / 2
      )
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_z( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i, j, k  )) + 
          psi(pi<dim>(i, j, k+1))
        ) / 2
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_z( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i, j, k  ))) + 
          abs(psi(pi<dim>(i, j, k+1)))
        ) / 2
      )
      
      // psi at (i+1/2, j+1/2)
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_xy( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i+1, j  , k)) + 
          psi(pi<dim>(i  , j  , k)) +
          psi(pi<dim>(i  , j+1, k)) +
          psi(pi<dim>(i+1, j+1, k))
        ) / 4
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_xy( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i+1, j  , k))) + 
          abs(psi(pi<dim>(i  , j  , k))) +
          abs(psi(pi<dim>(i  , j+1, k))) +
          abs(psi(pi<dim>(i+1, j+1, k)))
        ) / 4
      )
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_xz( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i+1, j  , k)) + 
          psi(pi<dim>(i  , j  , k)) +
          psi(pi<dim>(i  , j, k+1)) +
          psi(pi<dim>(i+1, j, k+1))
        ) / 4
      )

      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_xz( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i+1, j  , k))) + 
          abs(psi(pi<dim>(i  , j  , k))) +
          abs(psi(pi<dim>(i  , j, k+1))) +
          abs(psi(pi<dim>(i+1, j, k+1)))
        ) / 4
      )
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_xyz( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          psi(pi<dim>(i  , j  , k  )) +
          psi(pi<dim>(i+1, j  , k  )) + 
          psi(pi<dim>(i  , j+1, k  )) +
          psi(pi<dim>(i  , j  , k+1)) +
          psi(pi<dim>(i+1, j+1, k  )) + 
          psi(pi<dim>(i+1, j  , k+1)) + 
          psi(pi<dim>(i  , j+1, k+1)) + 
          psi(pi<dim>(i+1, j+1, k+1))
        ) / 8
      )
      
      template<opts_t opts, int dim, class arr_3d_t>
      inline auto psi_bar_xyz( 
        const arr_3d_t &psi,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        (
          abs(psi(pi<dim>(i  , j  , k  ))) +
          abs(psi(pi<dim>(i+1, j  , k  ))) + 
          abs(psi(pi<dim>(i  , j+1, k  ))) +
          abs(psi(pi<dim>(i  , j  , k+1))) +
          abs(psi(pi<dim>(i+1, j+1, k  ))) + 
          abs(psi(pi<dim>(i+1, j  , k+1))) + 
          abs(psi(pi<dim>(i  , j+1, k+1))) + 
          abs(psi(pi<dim>(i+1, j+1, k+1)))
        ) / 8
      )

      template<int dim, class arr_3d_t>
      inline auto GC1_bar_xy(
        const arr_3d_t &GC, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i+1, j+h, k)) + 
          GC(pi<dim>(i,   j+h, k)) +
          GC(pi<dim>(i+1, j-h, k)) + 
          GC(pi<dim>(i,   j-h, k)) 
        ) / 4
      )
      
      template<int dim, class arr_3d_t>
      inline auto GC1_bar_xz(
        const arr_3d_t &GC, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i, j+h, k  )) + 
          GC(pi<dim>(i, j+h, k+1)) +
          GC(pi<dim>(i, j-h, k  )) + 
          GC(pi<dim>(i, j-h, k+1)) 
        ) / 4
      )
      
      template<int dim, class arr_3d_t>
      inline auto GC2_bar_x(
        const arr_3d_t &GC, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i+1, j, k+h)) + 
          GC(pi<dim>(i,   j, k+h))
        ) / 2
      )

      template<int dim, class arr_3d_t>
      inline auto GC2_bar_xz(
        const arr_3d_t &GC, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i+1, j, k+h)) + 
          GC(pi<dim>(i,   j, k+h)) +
          GC(pi<dim>(i+1, j, k-h)) + 
          GC(pi<dim>(i,   j, k-h)) 
        ) / 4
      )
      
      template<int dim, class arr_3d_t>
      inline auto GC2_bar_xy(
        const arr_3d_t &GC, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i+1, j, k+h)) + 
          GC(pi<dim>(i,   j, k+h)) +
          GC(pi<dim>(i, j+1, k+h)) + 
          GC(pi<dim>(i, j  , k+h))
        ) / 4
      )
      
      template<int dim, class arr_3d_t>
      inline auto GC0_bar_xy(
        const arr_3d_t &GC, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i+h, j  , k)) + 
          GC(pi<dim>(i-h, j  , k)) +
          GC(pi<dim>(i+h, j+1, k)) + 
          GC(pi<dim>(i-h, j+1, k)) 
        ) / 4
      )

      template<int dim, class arr_3d_t>
      inline auto GC0_bar_xz(
        const arr_3d_t &GC, 
        const rng_t &i, 
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i+h, j, k  )) + 
          GC(pi<dim>(i-h, j, k  )) +
          GC(pi<dim>(i+h, j, k+1)) + 
          GC(pi<dim>(i-h, j, k+1)) 
        ) / 4
      )
      
      template<int dim, class arr_3d_t>
      inline auto GC0_bar_x( 
        const arr_3d_t &GC,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i+h, j, k)) + 
          GC(pi<dim>(i-h, j, k))
        ) / 2
      )
      
      template<int dim, class arr_3d_t>
      inline auto GC1_bar_x( 
        const arr_3d_t &GC,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC(pi<dim>(i+1, j+h, k)) + 
          GC(pi<dim>(i  , j+h, k))
        ) / 2
      )

      template <int dim, class arr_3d_t>
      inline auto dGC0_dx(
        const arr_3d_t &GC, 
        const rng_t &i,
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (GC(pi<dim>(i+h+1, j, k)) - GC(pi<dim>(i+h-1, j, k))) / 2
      )
      
      template <opts_t opts, int dim, class arr_3d_t>
      inline auto dGC0_dxx(
        const arr_3d_t &psi, 
        const arr_3d_t &GC, 
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        GC(pi<dim>(i+h+1, j, k)) + GC(pi<dim>(i+h-1, j, k)) - 2 * GC(pi<dim>(i+h, j, k))
      )
      
      template <opts_t opts, int dim, class arr_3d_t>
      inline auto dGC0_dxx(
        const arr_3d_t &psi, 
        const arr_3d_t &GC, 
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        (GC(pi<dim>(i+h+1, j, k)) + GC(pi<dim>(i+h-1, j, k)) - 2 * GC(pi<dim>(i+h, j, k))) * 
         psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j, k)
      )
      
      template <opts_t opts, int dim, class arr_3d_t>
      inline auto dGC0_dtt(
        const arr_3d_t &psi, 
        const arr_3d_t &dGC_dtt, 
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        dGC_dtt(pi<dim>(i+h, j, k)) + 0
      )
      
      template <opts_t opts, int dim, class arr_3d_t>
      inline auto dGC0_dtt(
        const arr_3d_t &psi, 
        const arr_3d_t &dGC_dtt, 
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        dGC_dtt(pi<dim>(i+h, j, k)) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j, k)
      )

      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv1(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
          GC[dim  ](pi<dim>(i+h, j, k  )) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i  , j  , k  )
        - GC[dim  ](pi<dim>(i-h, j, k  )) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i-1, j  , k  )

        + GC[dim+1](pi<dim>(i, j+h, k  )) * psi_bar_y<opts BOOST_PP_COMMA() dim>(psi, i  , j  , k  )
        - GC[dim+1](pi<dim>(i, j-h, k  )) * psi_bar_y<opts BOOST_PP_COMMA() dim>(psi, i  , j-1, k  )

        + GC[dim-1](pi<dim>(i, j  , k+h)) * psi_bar_z<opts BOOST_PP_COMMA() dim>(psi, i  , j  , k  )
        - GC[dim-1](pi<dim>(i, j  , k-h)) * psi_bar_z<opts BOOST_PP_COMMA() dim>(psi, i  , j  , k-1)
        ) / formulae::G<opts BOOST_PP_COMMA() dim>(G, i, j, k)
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv2(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
            GC0_bar_xy<dim>(GC[dim  ], i+1, j, k) * psi_bar_y<opts BOOST_PP_COMMA() dim>(psi, i+1, j, k)
          - GC0_bar_xy<dim>(GC[dim  ], i  , j, k) * psi_bar_y<opts BOOST_PP_COMMA() dim>(psi, i  , j, k)
            
          + GC1_bar_xy<dim>(GC[dim+1], i, j+1, k) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j+1, k)
          - GC1_bar_xy<dim>(GC[dim+1], i, j  , k) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j  , k)
          
          + GC2_bar_xy<dim>(GC[dim-1], i, j, k  ) * psi_bar_xyz<opts BOOST_PP_COMMA() dim>(psi, i, j , k  )
          - GC2_bar_xy<dim>(GC[dim-1], i, j, k-1) * psi_bar_xyz<opts BOOST_PP_COMMA() dim>(psi, i, j , k-1)
        ) / G_bar_xy<opts BOOST_PP_COMMA() dim>(G, i, j, k)
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv3(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k
      ) return_macro(,
        (
            GC0_bar_xz<dim>(GC[dim  ], i+1, j, k) * psi_bar_z<opts BOOST_PP_COMMA() dim>(psi, i+1, j, k)
          - GC0_bar_xz<dim>(GC[dim  ], i  , j, k) * psi_bar_z<opts BOOST_PP_COMMA() dim>(psi, i  , j, k)
            
          + GC1_bar_xz<dim>(GC[dim+1], i, j  , k) * psi_bar_xyz<opts BOOST_PP_COMMA() dim>(psi, i, j  , k)
          - GC1_bar_xz<dim>(GC[dim+1], i, j-1, k) * psi_bar_xyz<opts BOOST_PP_COMMA() dim>(psi, i, j-1, k)
          
          + GC2_bar_xz<dim>(GC[dim-1], i, j, k+1) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j , k+1)
          - GC2_bar_xz<dim>(GC[dim-1], i, j, k  ) * psi_bar_x<opts BOOST_PP_COMMA() dim>(psi, i, j , k  )
        ) / G_bar_xz<opts BOOST_PP_COMMA() dim>(G, i, j, k)
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        10 *
        frac<opts>(
          GC0_bar_x<dim>(GC[dim  ], i+1, j, k) * psi(pi<dim>(i+1, j, k))
        - GC0_bar_x<dim>(GC[dim  ], i  , j, k) * psi(pi<dim>(i  , j, k))

        + GC1_bar_x<dim>(GC[dim+1], i, j  , k) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j  , k)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1, k) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j-1, k)
        
        + GC2_bar_x<dim>(GC[dim-1], i, j, k  ) * psi_bar_xz<opts BOOST_PP_COMMA() dim>(psi, i, j, k  )
        - GC2_bar_x<dim>(GC[dim-1], i, j, k-1) * psi_bar_xz<opts BOOST_PP_COMMA() dim>(psi, i, j, k-1)
        ,
          G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k) * (
            psi(pi<dim>(i+1, j  , k  ))
          + psi(pi<dim>(i  , j  , k  ))
          + psi(pi<dim>(i  , j+1, k  ))
          + psi(pi<dim>(i+1, j+1, k  ))
          + psi(pi<dim>(i  , j-1, k  ))
          + psi(pi<dim>(i+1, j-1, k  ))
          + psi(pi<dim>(i+1, j  , k+1))
          + psi(pi<dim>(i+1, j  , k-1))
          + psi(pi<dim>(i  , j  , k-1))
          + psi(pi<dim>(i  , j  , k+1))
          )
        )
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        10 *
        frac<opts>(
          GC0_bar_x<dim>(GC[dim  ], i+1, j, k) * abs(psi(pi<dim>(i+1, j, k)))
        - GC0_bar_x<dim>(GC[dim  ], i  , j, k) * abs(psi(pi<dim>(i  , j, k)))

        + GC1_bar_x<dim>(GC[dim+1], i, j  , k) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j  , k)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1, k) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j-1, k)
        
        + GC2_bar_x<dim>(GC[dim-1], i, j, k  ) * psi_bar_xz<opts BOOST_PP_COMMA() dim>(psi, i, j, k  )
        - GC2_bar_x<dim>(GC[dim-1], i, j, k-1) * psi_bar_xz<opts BOOST_PP_COMMA() dim>(psi, i, j, k-1)
        ,
          G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k) * (
            abs(psi(pi<dim>(i+1, j  , k  )))
          + abs(psi(pi<dim>(i  , j  , k  )))
          + abs(psi(pi<dim>(i  , j+1, k  )))
          + abs(psi(pi<dim>(i+1, j+1, k  )))
          + abs(psi(pi<dim>(i  , j-1, k  )))
          + abs(psi(pi<dim>(i+1, j-1, k  )))
          + abs(psi(pi<dim>(i+1, j  , k+1)))
          + abs(psi(pi<dim>(i+1, j  , k-1)))
          + abs(psi(pi<dim>(i  , j  , k-1)))
          + abs(psi(pi<dim>(i  , j  , k+1)))
          )
        )
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        (
          GC0_bar_x<dim>(GC[dim  ], i+1, j, k) * psi(pi<dim>(i+1, j, k))
        - GC0_bar_x<dim>(GC[dim  ], i  , j, k) * psi(pi<dim>(i  , j, k))

        + GC1_bar_x<dim>(GC[dim+1], i, j  , k) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j  , k)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1, k) * psi_bar_xy<opts BOOST_PP_COMMA() dim>(psi, i, j-1, k)
        
        + GC2_bar_x<dim>(GC[dim-1], i, j, k  ) * psi_bar_xz<opts BOOST_PP_COMMA() dim>(psi, i, j, k  )
        - GC2_bar_x<dim>(GC[dim-1], i, j, k-1) * psi_bar_xz<opts BOOST_PP_COMMA() dim>(psi, i, j, k-1)
        ) / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k)
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv_gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        12 *
        frac<opts>(
          GC0_bar_x<dim>(GC[dim  ], i+1, j, k) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j, k)
        - GC0_bar_x<dim>(GC[dim  ], i  , j, k) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i  , j, k)

        + GC1_bar_x<dim>(GC[dim+1], i, j  , k) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j  , k)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1, k) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j-1, k)
        
        + GC2_bar_x<dim>(GC[dim-1], i, j, k  ) * gdiv3<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k  )
        - GC2_bar_x<dim>(GC[dim-1], i, j, k-1) * gdiv3<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k-1)
        ,
          G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k) * (
            psi(pi<dim>(i+2,   j, k)) +
            psi(pi<dim>(i+1,   j, k)) +
            psi(pi<dim>(i  ,   j, k)) +
            psi(pi<dim>(i-1,   j, k)) +
            psi(pi<dim>(i  , j+1, k)) +
            psi(pi<dim>(i  , j-1, k)) +
            psi(pi<dim>(i+1, j+1, k)) +
            psi(pi<dim>(i+1, j-1, k)) +
            psi(pi<dim>(i  , j, k+1)) +
            psi(pi<dim>(i  , j, k-1)) +
            psi(pi<dim>(i+1, j, k+1)) +
            psi(pi<dim>(i+1, j, k-1))
          )
        )
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv_gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        12 *
        frac<opts>(
          GC0_bar_x<dim>(GC[dim  ], i+1, j, k) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j, k)
        - GC0_bar_x<dim>(GC[dim  ], i  , j, k) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i  , j, k)

        + GC1_bar_x<dim>(GC[dim+1], i, j  , k) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j  , k)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1, k) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j-1, k)
        
        + GC2_bar_x<dim>(GC[dim-1], i, j, k  ) * gdiv3<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k  )
        - GC2_bar_x<dim>(GC[dim-1], i, j, k-1) * gdiv3<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k-1)
        ,
          G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k) * (
            abs(psi(pi<dim>(i+2,   j, k))) +
            abs(psi(pi<dim>(i+1,   j, k))) +
            abs(psi(pi<dim>(i  ,   j, k))) +
            abs(psi(pi<dim>(i-1,   j, k))) +
            abs(psi(pi<dim>(i  , j+1, k))) +
            abs(psi(pi<dim>(i  , j-1, k))) +
            abs(psi(pi<dim>(i+1, j+1, k))) +
            abs(psi(pi<dim>(i+1, j-1, k))) +
            abs(psi(pi<dim>(i  , j, k+1))) +
            abs(psi(pi<dim>(i  , j, k-1))) +
            abs(psi(pi<dim>(i+1, j, k+1))) +
            abs(psi(pi<dim>(i+1, j, k-1)))
          )
        )
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto gdiv_gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
        (
          GC0_bar_x<dim>(GC[dim  ], i+1, j, k) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j, k)
        - GC0_bar_x<dim>(GC[dim  ], i  , j, k) * gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i  , j, k)

        + GC1_bar_x<dim>(GC[dim+1], i, j  , k) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j  , k)
        - GC1_bar_x<dim>(GC[dim+1], i, j-1, k) * gdiv2<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j-1, k)
        
        + GC2_bar_x<dim>(GC[dim-1], i, j, k  ) * gdiv3<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k  )
        - GC2_bar_x<dim>(GC[dim-1], i, j, k-1) * gdiv3<opts BOOST_PP_COMMA() dim>(psi, GC, G, i, j, k-1)
        ) / G_bar_x<opts BOOST_PP_COMMA() dim>(G, i, j, k)
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto grad_gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && !opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        12 * 
        frac<opts>(
          gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j, k)
        - gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i  , j, k)
        ,
          psi(pi<dim>(i+2,   j, k)) +
          psi(pi<dim>(i+1,   j, k)) +
          psi(pi<dim>(i  ,   j, k)) +
          psi(pi<dim>(i-1,   j, k)) +
          psi(pi<dim>(i  , j+1, k)) +
          psi(pi<dim>(i  , j-1, k)) +
          psi(pi<dim>(i  , j, k-1)) +
          psi(pi<dim>(i  , j, k+1)) +
          psi(pi<dim>(i+1, j+1, k)) +
          psi(pi<dim>(i+1, j-1, k)) +
          psi(pi<dim>(i+1, j, k-1)) +
          psi(pi<dim>(i+1, j, k+1))
        )
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto grad_gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga) && opts::isset(opts, opts::abs)>::type* = 0
      ) return_macro(,
        12 * 
        frac<opts>(
          gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j, k)
        - gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i  , j, k)
        ,
          abs(psi(pi<dim>(i+2,   j, k))) +
          abs(psi(pi<dim>(i+1,   j, k))) +
          abs(psi(pi<dim>(i  ,   j, k))) +
          abs(psi(pi<dim>(i-1,   j, k))) +
          abs(psi(pi<dim>(i  , j+1, k))) +
          abs(psi(pi<dim>(i  , j-1, k))) +
          abs(psi(pi<dim>(i  , j, k-1))) +
          abs(psi(pi<dim>(i  , j, k+1))) +
          abs(psi(pi<dim>(i+1, j+1, k))) +
          abs(psi(pi<dim>(i+1, j-1, k))) +
          abs(psi(pi<dim>(i+1, j, k-1))) +
          abs(psi(pi<dim>(i+1, j, k+1)))
        )
      )
      
      template <opts_t opts, int dim, class arr_3d_t, class arrvec_t>
      inline auto grad_gdiv(
        const arr_3d_t &psi,
        const arrvec_t &GC, 
        const arr_3d_t &G,
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
          gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i+1, j, k)
        - gdiv1<opts BOOST_PP_COMMA() dim>(psi, GC, G, i  , j, k)
      )

      template <opts_t opts, int dim, class arr_3d_t>
      inline auto aux(
        const arr_3d_t &psi,
        const arr_3d_t &GC_corr, 
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<!opts::isset(opts, opts::iga)>::type* = 0
      ) return_macro(,
          abs(GC_corr(pi<dim>(i+h, j, k))) / 2 * dpsi_dx<opts BOOST_PP_COMMA() dim>(psi, i, j, k)
      )
      
      template <opts_t opts, int dim, class arr_3d_t>
      inline typename arr_3d_t::T_numtype aux(
        const arr_3d_t &psi,
        const arr_3d_t &GC_corr, 
        const rng_t &i,
        const rng_t &j,
        const rng_t &k,
        typename std::enable_if<opts::isset(opts, opts::iga)>::type* = 0
      ) {
        return 0;
      }
    } // namespace mpdata
  } // namespace formulae
} // namespace libmpdataxx
