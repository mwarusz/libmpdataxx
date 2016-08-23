/**
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 *
 */

#pragma once

#include <libmpdata++/solvers/mpdata_rhs_vip_prs.hpp>
#include <libmpdata++/solvers/detail/mpdata_rhs_vip_prs_sgs_common.hpp>

namespace libmpdataxx
{
  namespace solvers
  {
    template<typename ct_params_t, class enableif = void> 
    class mpdata_rhs_vip_prs_sgs;

    template<typename ct_params_t>
    class mpdata_rhs_vip_prs_sgs<
      ct_params_t,
      typename std::enable_if<(int)ct_params_t::sgs_scheme == (int)iles>::type
    > : public mpdata_rhs_vip_prs<ct_params_t>
    {
      using parent_t = mpdata_rhs_vip_prs<ct_params_t>; 
      using parent_t::parent_t; // inheriting constructors
    };
    
    template<typename ct_params_t>
    class mpdata_rhs_vip_prs_sgs<
      ct_params_t,
      typename std::enable_if<(int)ct_params_t::sgs_scheme == (int)dns>::type
    > : public detail::mpdata_rhs_vip_prs_sgs_common<ct_params_t>
    {
      using parent_t = detail::mpdata_rhs_vip_prs_sgs_common<ct_params_t>; 
      using parent_t::parent_t; // inheriting constructors
    };
  } // namespace solvers
} // namespace libmpdataxx
