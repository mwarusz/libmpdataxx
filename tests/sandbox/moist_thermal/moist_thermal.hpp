/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */
#pragma once
#include <libmpdata++/solvers/boussinesq.hpp>
#include <libmpdata++/solvers/mpdata_rhs_vip_prs.hpp>
#include <algorithm>
#include <fstream>

using namespace libmpdataxx::arakawa_c;

template <class ct_params_t>
class moist_thermal : public libmpdataxx::solvers::mpdata_rhs_vip_prs<ct_params_t>
{
  using parent_t = libmpdataxx::solvers::mpdata_rhs_vip_prs<ct_params_t>;

  public:
  using real_t = typename ct_params_t::real_t;
  

  protected:
  // member fields
  using ix = typename ct_params_t::ix;
  std::ofstream humanstat_file, compstat_file;
  real_t g, cp, Rd, Rv, L, e0, epsa, T0, buoy_eps, initial_totalws;

  std::string name;
  typename parent_t::arr_t &tht_b, &tht_e, &pk_e, &qv_e, &tmp1, &tmp2, &dtht_e, &qr_est, &col_sed;
  libmpdataxx::arrvec_t<typename parent_t::arr_t> &qrhs, &grad_aux;
    const libmpdataxx::rng_t ir;
    const libmpdataxx::rng_t kr;

  void check_neg_water(const std::string& str)
  {
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    this->mem->barrier();
    if (this->rank == 0)
    {
      auto qv_min     = min(qv(ir, kr));
      auto qc_min     = min(qc(ir, kr));
      auto qr_min     = min(qr(ir, kr));

      if(qv_min < 0 || qc_min < 0 || qr_min < 0) std::cout << str << std::endl;
      if (qv_min < 0) std::cout << "fneg qv: " << qv_min << std::endl;
      if (qc_min < 0) std::cout << "fneg qc: " << qc_min << std::endl;
      if (qr_min < 0) std::cout << "fneg qr: " << qr_min << std::endl;
    }
    this->mem->barrier();
  }
  
  void save_stats()
  {
    using namespace libmpdataxx::arakawa_c;

    auto &rho = *this->mem->G;
    auto &u = this->state(ix::u);
    auto &w = this->state(ix::w);
    auto &tht = this->state(ix::tht);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);

    this->mem->barrier();
    if (this->rank == 0)
    {
      int nz = this->kr.last() + 1;
      int np = (ir.last() + 1) * nz;
      
      auto tht_max     = max(tht(ir, kr));
      auto tht_min     = min(tht(ir, kr));
      auto tht_avg     = sum(tht(ir, kr)) / np;

      auto qv_max     = max(qv(ir, kr));
      auto qv_min     = min(qv(ir, kr));
      auto qv_avg     = sum(qv(ir, kr)) / np;
      
      auto qc_max     = max(qc(ir, kr));
      auto qc_min     = min(qc(ir, kr));
      auto qc_avg     = sum(qc(ir, kr)) / np;
      
      auto qr_max     = max(qr(ir, kr));
      auto qr_min     = min(qr(ir, kr));
      auto qr_avg     = sum(qr(ir, kr)) / np;
      
      auto u_min     = min(u(ir, kr));
      auto u_max     = max(u(ir, kr));
      auto u_avg     = sum(u(ir, kr)) / np;

      auto w_min     = min(w(ir, kr));
      auto w_max     = max(w(ir, kr));
      auto w_avg     = sum(w(ir, kr)) / np;
    
      const libmpdataxx::rng_t iri(ir.first(), ir.last() - 1);
      const libmpdataxx::rng_t kri(1, nz - 2);
      
      auto totalws     = 0.5 * sum(rho(iri, 0)   * (qv(iri, 0)   + qc(iri, 0)   + qr(iri, 0))  ) +
                               sum(rho(iri, kri) * (qv(iri, kri) + qc(iri, kri) + qr(iri, kri))) +
                         0.5 * sum(rho(iri, nz - 1)  * (qv(iri, nz - 1)  + qc(iri, nz - 1)  + qr(iri, nz - 1)) ) ;

      if (this->timestep == 0)
      {
        initial_totalws = totalws;
      }

      auto water_change = 100. * (totalws / initial_totalws - 1.);
      
      auto prec_rate  = -this->di * this->dj / this->dt * sum(col_sed(iri,  0)) / 1e5;
    
      humanstat_file.precision(18);
      compstat_file.precision(18);
      //stat_file << this->timestep << ' '
      //          << w_min << ' ' << w_max << ' ' << w_avg << ' '
      //          << qc_min << ' ' << qc_max << ' ' << qc_avg << ' '
      //          << qr_min << ' ' << qr_max << ' ' << qr_avg << ' ' << totalws << std::endl;
      humanstat_file << "timestep/time/dt " << this->timestep << ' ' << this->time << ' ' << this->dt << std::endl
                << "u  " << u_min << ' ' << u_max << ' ' << u_avg << std::endl
                << "w  " << w_min << ' ' << w_max << ' ' << w_avg << std::endl
                << "th " << tht_min << ' ' << tht_max << ' ' << tht_avg << std::endl
                << "qv " << qv_min << ' ' << qv_max << ' ' << qv_avg << std::endl
                << "qc " << qc_min << ' ' << qc_max << ' ' << qc_avg << std::endl
                << "qr " << qr_min << ' ' << qr_max << ' ' << qr_avg << std::endl
                << "totalws/change "<< totalws << ' ' << water_change << std::endl
                << "prec_rate " << prec_rate << std::endl;
      
      compstat_file << this->timestep << ' ' << this->time << ' ' << this->dt << ' '
                    << u_min << ' ' << u_max << ' ' << u_avg << ' '
                    << w_min << ' ' << w_max << ' ' << w_avg << ' '
                    << tht_min << ' ' << tht_max << ' ' << tht_avg << ' '
                    << qv_min << ' ' << qv_max << ' ' << qv_avg << ' '
                    << qc_min << ' ' << qc_max << ' ' << qc_avg << ' '
                    << qr_min << ' ' << qr_max << ' ' << qr_avg << ' '
                    << totalws << ' ' << water_change << ' ' << prec_rate << std::endl;
    }
    this->mem->barrier();
  }

  real_t pk2p(real_t pk)
  {
    const real_t p0 = 1e5;
    return std::pow(pk, cp / Rd) * p0;
  }

  template <int nd = ct_params_t::n_dims>
  void calc_dtht_e(typename std::enable_if<nd == 2>::type* = 0)
  {
    this->xchng_sclr(this->tht_e, this->ijk);
    this->dtht_e(this->ijk) = libmpdataxx::formulae::nabla::grad<1>(this->tht_e, this->j, this->i, this->dj);
  }

  //template <int nd = ct_params_t::n_dims> 
  //void calc_dtht_e(typename std::enable_if<nd == 3>::type* = 0)
  //{
  //  this->xchng_pres(this->tht_e, this->ijk);
  //  this->dtht_e(this->ijk) = libmpdataxx::formulae::nabla::grad<2>(this->tht_e, this->k, this->i, this->j, this->dk);
  //}

  virtual void normalize_vip(const libmpdataxx::arrvec_t<typename parent_t::arr_t> &v)
  {
    const auto &tht_abs = *this->mem->vab_coeff;
    if (static_cast<libmpdataxx::solvers::vip_vab_t>(ct_params_t::vip_vab) == libmpdataxx::solvers::impl)
    {
      for (int d = 0; d < ct_params_t::n_dims - 1; ++d)
      {
        v[d](this->ijk) /= (1 + 0.5 * this->dt * (*this->mem->vab_coeff)(this->ijk));
      }
      v[ct_params_t::n_dims - 1](this->ijk) /=
      (1 + 0.5 * this->dt * (*this->mem->vab_coeff)(this->ijk)
         + 0.25 * this->dt * this->dt * this->g / this->tht_b(this->ijk) * this->dtht_e(this->ijk)
           / (1 + 0.5 * this->dt * tht_abs(this->ijk)));
    }
    else
    {
      v[ct_params_t::n_dims - 1](this->ijk) /=
      (1 + 0.25 * this->dt * this->dt * this->g / this->tht_b(this->ijk) * this->dtht_e(this->ijk)
           / (1 + 0.5 * this->dt * tht_abs(this->ijk)));
    }
  }

  template<typename tht_t>
  void saturation_adjustment(tht_t& tht)
  {
    //auto &tht = this->state(ix::tht);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    
    for (int i = this->i.first(); i <= this->i.last(); ++i)
    for (int k = this->j.first(); k <= this->j.last(); ++k)
    {

      real_t pk = pk_e(i, k);
      real_t p = pk2p(pk);

      real_t th = tht(i, k);
      real_t T = th * pk;
      
      real_t es = e0 * std::exp(L / Rv * ((T - T0) / (T0 * T)));
      real_t qvs = epsa * es / (p - es);
     
      // linerized solution
      real_t cf1 = L * L / (cp * Rv) / (pk * th * pk * th) / (1 - es / p);
      real_t delta = (qv(i, k) - qvs) / (1. + qvs * cf1);
      
      // Newton-Raphson
      th += L / (cp * pk) * delta;
      T = th * pk;
      es = e0 * std::exp(L / Rv * ((T - T0) / (T0 * T)));
      qvs = epsa * es / (p - es);
      
      cf1 = L * L / (cp * Rv) / (pk * th * pk * th) / (1 - es / p);
      delta += (qv(i, k) - delta - qvs) / (1. + qvs * cf1);
      
      // limiting
      delta = std::min(qv(i, k), std::max(-qc(i, k), delta));
      
      // modifying fields
      qv(i, k) -= delta;
      qc(i, k) += delta;
      tht(i, k) += L / (cp * pk) * delta;
      
      // limiting
      delta = std::min(qv(i, k), std::max(-qc(i, k), delta));
      
      // modifying forces
      this->rhs.at(ix::qv)(i, k)  += - 2 * delta / this->dt;
      this->rhs.at(ix::qc)(i, k)  +=   2 * delta / this->dt;
      this->rhs.at(ix::thf)(i, k) +=   2 * L / (cp * pk) * delta / this->dt;
    }
  }
  
  void update_precip_forces(typename parent_t::arr_t &tht, const libmpdataxx::arrvec_t<typename parent_t::arr_t> &qrhs)
  {
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    auto &rho = *this->mem->G;

    for (int i = this->i.first(); i <= this->i.last(); ++i)
    for (int k = this->j.first(); k <= this->j.last(); ++k)
    {
      // remove zeros
      qv(i, k) = std::max(0., qv(i, k));
      qc(i, k) = std::max(0., qc(i, k));
      qr(i, k) = std::max(0., qr(i, k));

      real_t k1 = 1e-3;
      real_t k2 = 2.2;
      real_t qct = 5e-4;
      
      real_t qrp = qr(i, k);

      real_t AP = std::max(0., k1 * (qc(i, k) - qct));
      real_t CP = k2 * qc(i, k) * std::pow(qrp, 0.875);
      
      real_t C = 1.6 + 124.9 * std::pow(1e-3 * rho(i, k) * qrp, 0.2046);

      real_t pk = pk_e(i, k);
      real_t p = pk2p(pk);
      real_t th = tht(i, k);
      real_t T = th * pk;
      
      real_t es = e0 * std::exp(L / Rv * ((T - T0) / (T0 * T)));
      real_t qvs = epsa * es / (p - es);
      
      real_t ss = std::min(qv(i, k) / qvs - 1, 0.);
      qvs = qv(i, k) / (1. + ss + 1e-16);

      real_t EP = 1./ rho(i, k) * ss * C *
                  std::pow(1e-3 * rho(i, k) * qrp, 0.525)
                  / 
                  (5.4e2 + 2.55e5 / (p * qvs));

      //AP = 0.;
      //CP = 0.;
      //EP = 0.;

      real_t dcol = AP + CP;
      real_t devp = EP;
      
      // limiting
      dcol = std::min(dcol,  qc(i, k) / this->dt + qrhs[2](i, k));
      devp = std::max(devp, -qr(i, k) / this->dt - qrhs[3](i, k) - dcol);

      // modifying forces
      this->rhs.at(ix::qv)(i, k)  = -devp;
      this->rhs.at(ix::qc)(i, k)  = -dcol;
      this->rhs.at(ix::qr)(i, k)  = devp + dcol;
      this->rhs.at(ix::thf)(i, k) = L / (cp * pk) * devp;
    }
  }
 
  void sedimentation(const typename parent_t::arr_t &qr)
  {
    //auto &qr = this->state(ix::qr);
    auto &rho = *this->mem->G;

    for (int i = this->i.first(); i <= this->i.last(); ++i)
    {
        int lk = this->j.last();
        real_t rho_g = rho(i, 0);

        real_t rho_h = 0.5 * (rho(i, lk) + rho(i, lk - 1));
        real_t qr_h = 0.5 * (qr(i, lk) + qr(i, lk - 1));
        real_t vr_kmh = -36.34 * rho_h * this->dt / this->dj * 
              std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);
        
        tmp1(i, lk)  = qr(i, lk) / (1 - vr_kmh / rho(i, lk));
       
        // no flux
        //tmp1(i, lk)  = std::max(0., qr(i, lk)) / (1 + 2 *  vr_kmh / rho(i, lk));
        
        this->rhs.at(ix::qr)(i, lk)  += (tmp1(i, lk) - qr(i, lk)) / this->dt;
        col_sed(i, 0) = 0.5 * rho(i, lk) * (tmp1(i, lk) - qr(i, lk));

        for (int k = this->j.last() - 1; k > 0; --k)
        {
          real_t vr_kph = vr_kmh;


          rho_h = 0.5 * (rho(i, k) + rho(i, k - 1));
          qr_h = 0.5 * (qr(i, k) + qr(i, k - 1));
          
          real_t faux1 = -36.34 * 
              std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);
          
          vr_kmh = -36.34 * rho_h * this->dt / this->dj * 
              std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);

          tmp1(i, k) = (qr(i, k) - 1. / rho(i, k) * vr_kph * tmp1(i, k + 1)) / (1 - vr_kmh / rho(i, k));
          this->rhs.at(ix::qr)(i, k)  += (tmp1(i, k) - qr(i, k)) / this->dt;
          col_sed(i, 0) += rho(i, k) * (tmp1(i, k) - qr(i, k));
        }
        real_t vr_kph = vr_kmh;
          
        tmp1(i, 0) = (qr(i, 0) - 1. / rho(i, 0) * vr_kph * tmp1(i, 1)) / (1 - vr_kmh / rho(i, 0));
       
        // no flux
        //tmp1(i, 0) = (qr(i, 0) - 2. / rho(i, 0) * vr_kph * tmp1(i, 1));
        
        this->rhs.at(ix::qr)(i, 0)  += (tmp1(i, 0) - qr(i, 0)) / this->dt;
        col_sed(i, 0) += 0.5 * rho(i, 0) * (tmp1(i, 0) - qr(i, 0));
    }

    //for (int i = this->i.first(); i <= this->i.last(); ++i)
    //for (int j = this->j.first(); j <= this->j.last(); ++j)
    //for (int k = this->k.first(); k <= this->k.last(); ++k)
    //{
    //  qr(i, k) = tmp1(i, k);
    //}
  }

  void hook_ante_loop(const typename parent_t::advance_arg_t nt) 
  {
    if (this->rank == 0)
    {
      humanstat_file.open(name.c_str());
      std::string compname = "c" + name;
      compstat_file.open(compname.c_str());
    }

    calc_dtht_e();

    save_stats();

    parent_t::hook_ante_loop(nt);
  }
  
  // explicit forcings 
  void update_rhs(
    libmpdataxx::arrvec_t<
      typename parent_t::arr_t
    > &rhs, 
    const real_t &dt, 
    const int &at 
  ) 
  {
    
    auto &tht = this->state(ix::tht);
    auto &thf = this->state(ix::thf);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    
    const auto &ijk = this->ijk;
    auto ix_w = this->vip_ixs[ct_params_t::n_dims - 1];
    const auto &tht_abs = *this->mem->vab_coeff;

    switch(at)
    {
      case (0):
      {
        // zero rhs for all equations
        for (int e = 0; e < 7; ++e)
        {
          rhs.at(e)(ijk) = 0;
        }

        break;
      }
      case (1):
      {
        // zero rhs for dynamic equations
        for (int e = 0; e < 3; ++e)
        {
          rhs.at(e)(ijk) = 0;
        }
        
        qrhs[0](ijk) = this->rhs.at(ix::thf)(ijk);
        qrhs[1](ijk) = this->rhs.at(ix::qv)(ijk);
        qrhs[2](ijk) = this->rhs.at(ix::qc)(ijk);
        qrhs[3](ijk) = this->rhs.at(ix::qr)(ijk);

        //tmp2(ijk) = tht(ijk) + tht_e(ijk);

        update_precip_forces(thf, qrhs);

        // construct estimate of qr without fallout
        qr_est(ijk) = max(0., qr(ijk) + this->dt * (qrhs[3](ijk) + this->rhs.at(ix::qr)(ijk)));
       
        //this->state(ix::qr)(this->ijk) = max(0., this->state(ix::qr)(this->ijk));
        
        sedimentation(qr_est);

        //this->state(ix::qr)(this->ijk) += this->dt * this->rhs.at(ix::qr)(this->ijk);
        //this->rhs.at(ix::qr)(this->ijk) = 0;        
        
        check_neg_water("neg_water before trapezoidal forces");
        
        // apply trapezoidal part of force
        this->state(ix::thf)(ijk) += 0.5 * this->dt * qrhs[0](ijk);
        qv(ijk) += 0.5 * this->dt * qrhs[1](ijk);
        qc(ijk) += 0.5 * this->dt * qrhs[2](ijk);
        qr(ijk) += 0.5 * this->dt * qrhs[3](ijk);
        
        check_neg_water("neg_water after trapezoidal forces");
        
        // limit precipitation forces
        //this->rhs.at(ix::qc)(ijk) = max(-qc(ijk) / this->dt, this->rhs.at(ix::qc)(ijk));
        //this->rhs.at(ix::qr)(ijk) = max(-qr(ijk) / this->dt, this->rhs.at(ix::qr)(ijk));

        // limit precipitation forces and restore conservation
        for (int i = this->i.first(); i <= this->i.last(); ++i)
        for (int k = this->j.first(); k <= this->j.last(); ++k)
        {
          auto fqc_lm = std::max(-qc(i, k) / this->dt, this->rhs.at(ix::qc)(i, k));
          auto fqr_lm = std::max(-qr(i, k) / this->dt, this->rhs.at(ix::qr)(i, k));
          auto delql = fqc_lm - this->rhs.at(ix::qc)(i, k) + fqr_lm - this->rhs.at(ix::qr)(i, k);

          auto fqv_adj = this->rhs.at(ix::qv)(i, k) - delql;
          
          const real_t pk = pk_e(i, k);
          auto fthf_adj = this->rhs.at(ix::thf)(i, k) + 2 * L / (cp * pk) * delql;

          this->rhs.at(ix::qv)(i, k) = fqv_adj;
          this->rhs.at(ix::qc)(i, k) = fqc_lm;
          this->rhs.at(ix::qr)(i, k) = fqr_lm;
          this->rhs.at(ix::thf)(i, k) = fthf_adj;
        }

        // apply precipitation temp force to both full tht and tht perturbation
        this->state(ix::thf)(ijk) += this->dt * this->rhs.at(ix::thf)(ijk);
        this->state(ix::tht)(ijk) += this->dt * this->rhs.at(ix::thf)(ijk);

        //qv(ijk)  += this->dt * this->rhs.at(ix::qv)(ijk);
        //qc(ijk)  += this->dt * this->rhs.at(ix::qc)(ijk);
        //qr(ijk)  += this->dt * this->rhs.at(ix::qr)(ijk);
        
        qv(ijk)  = max(0., qv(ijk) + this->dt * this->rhs.at(ix::qv)(ijk));
        qc(ijk)  = max(0., qc(ijk) + this->dt * this->rhs.at(ix::qc)(ijk));
        qr(ijk)  = max(0., qr(ijk) + this->dt * this->rhs.at(ix::qr)(ijk));
        
        check_neg_water("neg_water after precip forces");
        
        // zero moist forces
        this->rhs.at(ix::thf)(ijk) = 0;
        this->rhs.at(ix::qv)(ijk) = 0;
        this->rhs.at(ix::qc)(ijk) = 0;
        this->rhs.at(ix::qr)(ijk) = 0;        
        
        saturation_adjustment(thf);
        
        check_neg_water("neg_water after condensation");
        
        // add condensation force to perturbation tht forcings
        this->rhs.at(ix::tht)(this->ijk) += this->rhs.at(ix::thf)(this->ijk);

        // apply condensation before calculating buoyancy
        this->state(ix::tht)(this->ijk) += 0.5 * this->dt * this->rhs.at(ix::tht)(this->ijk);

        rhs.at(ix_w)(ijk) += this->g * (
                ( tht(ijk) / this->tht_b(ijk) / (1 + 0.5 * dt * tht_abs(ijk))
                + buoy_eps * (this->state(ix::qv)(ijk) - this->qv_e(ijk))
                - this->state(ix::qc)(ijk) - this->state(ix::qr)(ijk) 
                ));
        break;
      }
    }
  }

  void hook_mixed_rhs_ante_step()
  {
    //this->apply_rhs(this->dt / 2);

    // only apply rhs for dynamic equations
    for (int e = 0; e < 3; ++e)
    {
      this->state(e)(this->ijk) += 0.5 * this->dt * this->rhs.at(e)(this->ijk);
    }

    if (this->rank == 0) std::cout << "timestep: " << this->timestep << std::endl;
    check_neg_water("neg_water in mixed_ante_step");
    
    // advec moist forcings with upwind
    this->self_advec_donorcell(this->rhs.at(ix::qv));
    this->self_advec_donorcell(this->rhs.at(ix::qc));
    this->self_advec_donorcell(this->rhs.at(ix::qr));
    this->self_advec_donorcell(this->rhs.at(ix::thf));
  }

  void hook_mixed_rhs_post_step()
  {
    this->update_rhs(this->rhs, this->dt / 2, 1);
    this->state(ix::w)(this->ijk) += 0.5 * this->dt * this->rhs.at(ix::w)(this->ijk);
  }

  void vip_rhs_impl_fnlz()
  {
    parent_t::vip_rhs_impl_fnlz();
    
    const auto &ijk = this->ijk;
    
    auto &tht = this->state(ix::tht);
    auto &w = this->state(ix::w);
    
    const auto &tht_abs = *this->mem->vab_coeff;
   
    tht(ijk) = (tht(ijk) - 0.5 * this->dt * w(ijk) * this->dtht_e(ijk))
               /
               (1 + 0.5 * this->dt * tht_abs(ijk));
    
    this->rhs.at(ix::tht)(ijk) += -w(ijk) * this->dtht_e(ijk) - tht_abs(ijk) * tht(ijk);
  }
  
  void hook_ante_step()
  {
    //save_stats();

    parent_t::hook_ante_step();

    //this->state(ix::qv)(this->ijk) = max(0., this->state(ix::qv)(this->ijk));
    //this->state(ix::qc)(this->ijk) = max(0., this->state(ix::qc)(this->ijk));
    //this->state(ix::qr)(this->ijk) = max(0., this->state(ix::qr)(this->ijk));
    //this->mem->barrier();
  }


  void hook_post_step()
  {
    parent_t::hook_post_step();
    //this->mem->barrier();
    
    this->state(ix::thf)(this->ijk) = this->state(ix::tht)(this->ijk) + tht_e(this->ijk);
    
    save_stats();
  }

  public:

  struct rt_params_t : parent_t::rt_params_t 
  { 
    real_t g, cp, Rd, Rv, L, e0, epsa, T0, buoy_eps;
    std::string name;
  };

  // ctor
  moist_thermal( 
    typename parent_t::ctor_args_t args, 
    const rt_params_t &p
  ) :
    parent_t(args, p),
    g(p.g),
    cp(p.cp),
    Rd(p.Rd),
    Rv(p.Rv),
    L(p.L),
    e0(p.e0),
    epsa(p.epsa),
    T0(p.T0),
    buoy_eps(p.buoy_eps),
    name(p.name),
    tht_b(args.mem->tmp[__FILE__][0][0]),
    tht_e(args.mem->tmp[__FILE__][1][0]),
    pk_e(args.mem->tmp[__FILE__][2][0]),
    qv_e(args.mem->tmp[__FILE__][3][0]),
    tmp1(args.mem->tmp[__FILE__][4][0]),
    tmp2(args.mem->tmp[__FILE__][4][1]),
    dtht_e(args.mem->tmp[__FILE__][5][0]),
    qr_est(args.mem->tmp[__FILE__][6][0]),
    col_sed(args.mem->tmp[__FILE__][7][0]),
    qrhs(args.mem->tmp[__FILE__][8]),
    grad_aux(args.mem->tmp[__FILE__][9]),
    ir(0, p.grid_size[0] - 1),
    kr(0, p.grid_size[1] - 1)
  {}

  static void alloc(typename parent_t::mem_t *mem, const int &n_iters)
  {
    parent_t::alloc(mem, n_iters);
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "tht_b");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "tht_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "pk_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "qv_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 2); // tmp1, tmp2
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "dtht_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1); // qr_est
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "", true); // col_sed
    parent_t::alloc_tmp_sclr(mem, __FILE__, 4, "qrhs");
    parent_t::alloc_tmp_vctr(mem, __FILE__); // grad_aux
  }
};
