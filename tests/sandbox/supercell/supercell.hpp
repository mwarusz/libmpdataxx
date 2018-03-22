/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */
#pragma once
#include <libmpdata++/solvers/mpdata_rhs_vip_prs.hpp>

template <class ct_params_t>
class supercell : public libmpdataxx::solvers::mpdata_rhs_vip_prs<ct_params_t>
{
  using parent_t = libmpdataxx::solvers::mpdata_rhs_vip_prs<ct_params_t>;

  public:
  using real_t = typename ct_params_t::real_t;

  protected:
  // member fields
  using ix = typename ct_params_t::ix;
  typename parent_t::arr_t &tht_e, &tht_0, &rho_0, &thm_e, &qv_e;

  real_t gg, rg, rv, cp, hlatv, hlats, tup, tdn, tt0, ee0;

  void update_rhs(
    libmpdataxx::arrvec_t<
      typename parent_t::arr_t
    > &rhs, 
    const real_t &dt, 
    const int &at 
  ) {
    parent_t::update_rhs(rhs, dt, at); 

    real_t epsb = rv/rg-1;

    const auto &ijk = this->ijk;

    auto &tht = this->state(ix::tht); 
    auto &w = this->state(ix::w); 
    auto &qv = this->state(ix::qv); 
    auto &qc = this->state(ix::qc); 
    
    const auto &tht_e = this->tht_e; 
    const auto &tht_0 = this->tht_0; 
    const auto &thm_e = this->thm_e; 
    const auto &rho_0 = this->rho_0; 
    const auto &qv_e = this->qv_e; 
    
    using T = typename ct_params_t::real_t;
    auto alim01 = [](T x){return std::max(0.0, std::min(1.0, x));};
    auto comb = [&alim01](T tm, T td, T tu, T ad, T au){return alim01((tm-td)/(tu-td))*au + alim01((tu-tm)/(tu-td))*ad;};

    auto a=rg/rv;
    auto c=hlatv/cp;
    auto b=hlats/rv;
    auto d=hlatv/rv;
    auto e=-cp/rg;

    switch (at)
    {
      case (0):
      {
        //rhs.at(ix::w)(ijk) += gg * ((tht(ijk) - this->tht_e(ijk)) / tht_0(ijk)
        //                      + epsb * (qv(ijk) - qv_e(ijk)) - qc(ijk));
        break;
      }
      case (1):
      {
        for (int i = this->i.first(); i <= this->i.last(); ++i)
        {
          for (int j = this->j.first(); j <= this->j.last(); ++j)
          {
            for (int k = this->k.first(); k <= this->k.last(); ++k)
            {
              auto thetme=tht_e(0, j, k)/thm_e(0, j, k);
              auto coe_l=comb(thm_e(0, j, k),tdn,tup,0.,1.);
              auto pre=1.e5*pow(thetme, e);
              auto tt=tht(i,j, k)/thetme;
              auto delt=(tt-tt0)/(tt*tt0);
              auto esw=ee0*exp(d * delt);
              auto esi=ee0*exp(b * delt);
              auto qvsw=a * esw /(pre-esw);
              auto qvsi=a * esi /(pre-esi);
              auto qvs=coe_l*qvsw + (1.-coe_l)*qvsi;
        // linearized condensation rate is next:
              auto cf1=thetme/tht(i,j,k);
              cf1=cf1*cf1;
              cf1=c*cf1*pre/(pre-esw)*d;
              auto delta=(qv(i,j,k)-qvs)/(1.+qvs*cf1);
        //  one Newton-Raphson iteration is next:
              auto thn=tht(i,j, k)+c*thetme*delta;
              tt=thn/thetme;
              delt=(tt-tt0)/(tt*tt0);
              esw=ee0*exp(d * delt);
              esi=ee0*exp(b * delt);
              qvsw=a * esw /(pre-esw);
              qvsi=a * esi /(pre-esi);
              qvs=coe_l*qvsw + (1.-coe_l)*qvsi;
              auto fff=qv(i,j,k)-delta-qvs;
              cf1=thetme/thn;
              cf1=cf1*cf1;
              cf1=c*cf1*pre/(pre-esw)*d;
              auto fffp=-1.-qvs*cf1;
              delta=delta-fff/fffp;
        //      end of the iteration; if required, it can be repeated
              delta=std::min( qv(i,j,k), std::max(-qc(i,j,k),delta) );

              auto new_qv=qv(i,j,k)-delta;
              auto new_qc=qc(i,j,k)+delta;
              auto new_tht=tht(i,j,k)+c*thetme*delta;
              //delta=std::min( new_qv, std::max(-new_qc,delta) );
              
              rhs.at(ix::qv)(i,j,k) += -delta*2./this->dt;
              rhs.at(ix::tht)(i,j,k)+= -c*thetme*rhs.at(ix::qv)(i,j,k);
              rhs.at(ix::qc)(i,j,k) += -rhs.at(ix::qv)(i,j,k);
          
              rhs.at(ix::w)(i,j,k) += gg * ((new_tht - this->tht_e(i, j,k)) / tht_0(i,j,k)
                                  + epsb * (new_qv - qv_e(i, j,k)) - new_qc);
            }
          }
        }
      }
    }
  }

  public:
  struct rt_params_t : parent_t::rt_params_t 
  { 
    real_t gg, rg, rv, cp, hlatv, hlats, tup, tdn, tt0, ee0;
  };

  // ctor
  supercell( 
    typename parent_t::ctor_args_t args, 
    const rt_params_t &p
  ) :
    parent_t(args, p),
    gg(p.gg), 
    rg(p.rg), 
    rv(p.rv), 
    cp(p.cp), 
    hlatv(p.hlatv),
    hlats(p.hlats),
    tup(p.tup),
    tdn(p.tdn),
    tt0(p.tt0),
    ee0(p.ee0),
    tht_e(args.mem->tmp[__FILE__][0][0]),
    tht_0(args.mem->tmp[__FILE__][1][0]),
    rho_0(args.mem->tmp[__FILE__][2][0]),
    thm_e(args.mem->tmp[__FILE__][3][0]),
    qv_e(args.mem->tmp[__FILE__][4][0])
  {}

  static void alloc(typename parent_t::mem_t *mem, const int &n_iters)
  {
    parent_t::alloc(mem, n_iters);
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "tht_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "tht_0");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "rho_0");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "thm_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "qv_e");
  }
};
