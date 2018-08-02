/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */

#include "pbl_test_def.hpp"

int main()
{
  const double dt = 8.0;
  const int prs_order = 2;
  //test<iles_tag, opts::iga | opts::fct, 2, prs_order>("out_iga_fct_prs2", dt);
  //test<iles_tag, opts::iga | opts::tot | opts::fct, 2, prs_order>("out_iga_tot_fct_prs2", dt);
  test<iles_tag, opts::iga | opts::div_2nd | opts::div_3rd | opts::fct, 2, prs_order>("out_iga_div3_fct_prs2", dt);

  //test<iles_tag, opts::iga, 2, prs_order>("out_iga_prs2", dt);
  //test<iles_tag, opts::iga | opts::tot, 2, prs_order>("out_iga_tot_prs2", dt);
  //test<iles_tag, opts::iga | opts::div_2nd | opts::div_3rd, 2, prs_order>("out_iga_div3_prs2", dt);
  //
  //test<iles_tag, opts::abs, 2, 4>("out_abs_prs2", dt);
  //test<iles_tag, opts::abs | opts::tot, 3, 4>("out_abs_tot_prs2", dt);
  //test<iles_tag, opts::abs | opts::div_2nd | opts::div_3rd, 2, 4>("out_abs_div3_prs2", dt);
}
