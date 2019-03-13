/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */
#pragma once
#include <libmpdata++/solvers/boussinesq.hpp>
#include <algorithm>
#include <fstream>

using namespace libmpdataxx::arakawa_c;

template <class ct_params_t>
class supercell : public libmpdataxx::solvers::mpdata_rhs_vip_prs_sgs<ct_params_t>
{
  using parent_t = libmpdataxx::solvers::mpdata_rhs_vip_prs_sgs<ct_params_t>;

  public:
  using real_t = typename ct_params_t::real_t;
  

  protected:
  // member fields
  using ix = typename ct_params_t::ix;
  std::ofstream humanstat_file, compstat_file;
  real_t g, cp, Rd, Rv, L, e0, epsa, T0, buoy_eps, initial_totalws, v_mult;

  std::string name;
  typename parent_t::arr_t &tht_b, &tht_e, &pk_e, &qv_e, &tmp1, &tmp2, &u_e, &dtht_e, &qr_est, &col_sed;
  libmpdataxx::arrvec_t<typename parent_t::arr_t> &qrhs, &grad_aux;
    const libmpdataxx::rng_t ir;
    const libmpdataxx::rng_t jr;
    const libmpdataxx::rng_t kr;

  void check_neg_water(const std::string& str)
  {
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    this->mem->barrier();
    if (this->rank == 0)
    {
      auto qv_min     = min(qv(ir, jr, kr));
      auto qc_min     = min(qc(ir, jr, kr));
      auto qr_min     = min(qr(ir, jr, kr));

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
    auto &v = this->state(ix::v);
    auto &w = this->state(ix::w);
    auto &tht = this->state(ix::tht);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);

    this->mem->barrier();
    if (this->rank == 0)
    {
      int nz = this->kr.last() + 1;
      int np = (ir.last() + 1) * (jr.last() + 1) * nz;
      
      auto tht_max     = max(tht(ir, jr, kr));
      auto tht_min     = min(tht(ir, jr, kr));
      auto tht_avg     = sum(tht(ir, jr, kr)) / np;

      auto qv_max     = max(qv(ir, jr, kr));
      auto qv_min     = min(qv(ir, jr, kr));
      auto qv_avg     = sum(qv(ir, jr, kr)) / np;
      
      auto qc_max     = max(qc(ir, jr, kr));
      auto qc_min     = min(qc(ir, jr, kr));
      auto qc_avg     = sum(qc(ir, jr, kr)) / np;
      
      auto qr_max     = max(qr(ir, jr, kr));
      auto qr_min     = min(qr(ir, jr, kr));
      auto qr_avg     = sum(qr(ir, jr, kr)) / np;
      
      auto u_min     = min(u(ir, jr, kr));
      auto u_max     = max(u(ir, jr, kr));
      auto u_avg     = sum(u(ir, jr, kr)) / np;
      
      auto v_min     = min(v(ir, jr, kr));
      auto v_max     = max(v(ir, jr, kr));
      auto v_avg     = sum(v(ir, jr, kr)) / np;

      auto w_min     = min(w(ir, jr, kr));
      auto w_max     = max(w(ir, jr, kr));
      auto w_avg     = sum(w(ir, jr, kr)) / np;
    
      const libmpdataxx::rng_t iri(ir.first(), ir.last() - 1);
      const libmpdataxx::rng_t jri(jr.first(), jr.last() - 1);
      const libmpdataxx::rng_t kri(1, nz - 2);
      
      auto totalws     = 0.5 * sum(rho(iri, jri, 0)   * (qv(iri, jri, 0)   + qc(iri, jri, 0)   + qr(iri, jri, 0))  ) +
                               sum(rho(iri, jri, kri) * (qv(iri, jri, kri) + qc(iri, jri, kri) + qr(iri, jri, kri))) +
                         0.5 * sum(rho(iri, jri, nz - 1)  * (qv(iri, jri, nz - 1)  + qc(iri, jri, nz - 1)  + qr(iri, jri, nz - 1)) ) ;

      if (this->timestep == 0)
      {
        initial_totalws = totalws;
      }

      auto water_change = 100. * (totalws / initial_totalws - 1.);
      
      auto prec_rate  = -this->di * this->dj * this->dk / this->dt * sum(col_sed(iri, jri, 0)) / 1e5;
    
      humanstat_file.precision(18);
      compstat_file.precision(18);
      //stat_file << this->timestep << ' '
      //          << w_min << ' ' << w_max << ' ' << w_avg << ' '
      //          << qc_min << ' ' << qc_max << ' ' << qc_avg << ' '
      //          << qr_min << ' ' << qr_max << ' ' << qr_avg << ' ' << totalws << std::endl;
      humanstat_file << "timestep/time/dt " << this->timestep << ' ' << this->time << ' ' << this->dt << std::endl
                << "u  " << u_min << ' ' << u_max << ' ' << u_avg << std::endl
                << "v  " << v_min << ' ' << v_max << ' ' << v_avg << std::endl
                << "w  " << w_min << ' ' << w_max << ' ' << w_avg << std::endl
                << "th " << tht_min << ' ' << tht_max << ' ' << tht_avg << std::endl
                << "qv " << qv_min << ' ' << qv_max << ' ' << qv_avg << std::endl
                << "qc " << qc_min << ' ' << qc_max << ' ' << qc_avg << std::endl
                << "qr " << qr_min << ' ' << qr_max << ' ' << qr_avg << std::endl
                << "totalws/change "<< totalws << ' ' << water_change << std::endl
                << "prec_rate " << prec_rate << std::endl;
      
      compstat_file << this->timestep << ' ' << this->time << ' ' << this->dt << ' '
                    << u_min << ' ' << u_max << ' ' << u_avg << ' '
                    << v_min << ' ' << v_max << ' ' << v_avg << ' '
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
  void calc_dtht_e(typename std::enable_if<nd == 3>::type* = 0)
  {
    this->xchng_pres(this->tht_e, this->ijk);
    this->dtht_e(this->ijk) = libmpdataxx::formulae::nabla::grad<2>(this->tht_e, this->k, this->i, this->j, this->dk);
  }

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
  
  template<typename ijk_t, typename ijkm_t>
  auto vlap_cmpct(
    typename parent_t::arr_t &arr, 
    real_t visc,
    const ijk_t &ijk, 
    const ijkm_t &ijkm, 
    const std::array<real_t, parent_t::n_dims>& dijk
  ) return_macro(
    this->xchng_pres(arr, ijk);
    libmpdataxx::formulae::nabla::calc_grad_cmpct<parent_t::n_dims>(this->grad_aux, arr, ijk, ijkm, dijk);
    this->mem->barrier();
    if (this->mem->G)
    {
      this->grad_aux[0](this->i + h, this->j, this->k) *= visc * 0.5 * (
        (*this->mem->G)(this->i + 1, this->j, this->k) + (*this->mem->G)(this->i, this->j, this->k)
      );
      this->grad_aux[1](this->i, this->j + h, this->k) *= visc * 0.5 * (
        (*this->mem->G)(this->i, this->j + 1, this->k) + (*this->mem->G)(this->i, this->j, this->k)
      );
      this->grad_aux[2](this->i, this->j, this->k + h) *= visc * 0.5 * (
        (*this->mem->G)(this->i, this->j, this->k + 1) + (*this->mem->G)(this->i, this->j, this->k)
      );
    }
    this->xchng_vctr_alng(this->grad_aux);
    ,
    (
      (this->grad_aux[0](this->i + h, this->j, this->k) - this->grad_aux[0](this->i - h, this->j, this->k)) / this->di
    + (this->grad_aux[1](this->i, this->j + h, this->k) - this->grad_aux[1](this->i, this->j - h, this->k)) / this->dj
    + (this->grad_aux[2](this->i, this->j, this->k + h) - this->grad_aux[2](this->i, this->j, this->k - h)) / this->dk
    ) / libmpdataxx::formulae::G<ct_params_t::opts>(*this->mem->G, this->ijk)
  )
  
  void diffusion_cmpct()
  {

    std::array<libmpdataxx::rng_t, ct_params_t::n_dims> ijkm;
    for (int d = 0; d < ct_params_t::n_dims; ++d)
    {
      ijkm[d] = libmpdataxx::rng_t(this->ijk[d].first() - 1, this->ijk[d].last());
    }

    auto &u = this->state(ix::u);
    auto &v = this->state(ix::v);
    auto &w = this->state(ix::w);

    auto &tht = this->state(ix::tht);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);

    auto v_mom = v_mult * 500.;
    auto v_sclr = v_mult * 1500.;
   
    tmp1(this->ijk) = u(this->ijk) - u_e(this->ijk);
    this->rhs.at(ix::u)(this->ijk) += 2.0 * vlap_cmpct(tmp1, v_mom, this->ijk, ijkm, this->dijk);
    this->rhs.at(ix::v)(this->ijk) += 2.0 * vlap_cmpct(v, v_mom, this->ijk   , ijkm, this->dijk);
    this->rhs.at(ix::w)(this->ijk) += 2.0 * vlap_cmpct(w, v_mom, this->ijk   , ijkm, this->dijk);
    
    tmp2(this->ijk) = qv(this->ijk) - qv_e(this->ijk);
    
    tmp1(this->ijk) = 2.0 * vlap_cmpct(tht, v_sclr, this->ijk, ijkm, this->dijk);
    this->rhs.at(ix::tht)(this->ijk) += tmp1(this->ijk);

    // add diffusion force to full tht forcings
    this->rhs.at(ix::thf)(this->ijk) += tmp1(this->ijk);

    this->rhs.at(ix::qv)(this->ijk)  += 2.0 * vlap_cmpct(tmp2, v_sclr, this->ijk, ijkm, this->dijk);
    this->rhs.at(ix::qc)(this->ijk)  += 2.0 * vlap_cmpct(qc , v_sclr, this->ijk , ijkm, this->dijk);
    this->rhs.at(ix::qr)(this->ijk)  += 2.0 * vlap_cmpct(qr , v_sclr, this->ijk , ijkm, this->dijk);
  }

  template<typename tht_t>
  void saturation_adjustment(tht_t& tht)
  {
    //auto &tht = this->state(ix::tht);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    
    for (int i = this->i.first(); i <= this->i.last(); ++i)
    for (int j = this->j.first(); j <= this->j.last(); ++j)
    for (int k = this->k.first(); k <= this->k.last(); ++k)
    {

      real_t pk = pk_e(i, j, k);
      real_t p = pk2p(pk);

      real_t th = tht(i, j, k);
      real_t T = th * pk;
      
      real_t es = e0 * std::exp(L / Rv * ((T - T0) / (T0 * T)));
      real_t qvs = epsa * es / (p - es);
     
      // linerized solution
      real_t cf1 = L * L / (cp * Rv) / (pk * th * pk * th) / (1 - es / p);
      real_t delta = (qv(i, j, k) - qvs) / (1. + qvs * cf1);
      
      // Newton-Raphson
      th += L / (cp * pk) * delta;
      T = th * pk;
      es = e0 * std::exp(L / Rv * ((T - T0) / (T0 * T)));
      qvs = epsa * es / (p - es);
      
      cf1 = L * L / (cp * Rv) / (pk * th * pk * th) / (1 - es / p);
      delta += (qv(i, j, k) - delta - qvs) / (1. + qvs * cf1);
      
      // limiting
      delta = std::min(qv(i, j, k), std::max(-qc(i, j, k), delta));
      
      // modifying fields
      qv(i, j, k) -= delta;
      qc(i, j, k) += delta;
      tht(i, j, k) += L / (cp * pk) * delta;
      
      // limiting
      delta = std::min(qv(i, j, k), std::max(-qc(i, j, k), delta));
      
      // modifying forces
      this->rhs.at(ix::qv)(i, j, k)  += - 2 * delta / this->dt;
      this->rhs.at(ix::qc)(i, j, k)  +=   2 * delta / this->dt;
      this->rhs.at(ix::thf)(i, j, k) +=   2 * L / (cp * pk) * delta / this->dt;
    }
  }
  
  void update_precip_forces(typename parent_t::arr_t &tht, const libmpdataxx::arrvec_t<typename parent_t::arr_t> &qrhs)
  {
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    auto &rho = *this->mem->G;

    for (int i = this->i.first(); i <= this->i.last(); ++i)
    for (int j = this->j.first(); j <= this->j.last(); ++j)
    for (int k = this->k.first(); k <= this->k.last(); ++k)
    {
      // remove zeros
      if (v_mult == 0.)
      {
        qv(i, j, k) = std::max(0., qv(i, j, k));
        qc(i, j, k) = std::max(0., qc(i, j, k));
        qr(i, j, k) = std::max(0., qr(i, j, k));
      }

      real_t k1 = 1e-3;
      real_t k2 = 2.2;
      real_t qct = 1e-3;
      
      real_t qrp = qr(i, j, k);

      real_t AP = std::max(0., k1 * (qc(i, j, k) - qct));
      real_t CP = k2 * qc(i, j, k) * std::pow(qrp, 0.875);
      
      real_t C = 1.6 + 124.9 * std::pow(1e-3 * rho(i, j, k) * qrp, 0.2046);

      real_t pk = pk_e(i, j, k);
      real_t p = pk2p(pk);
      real_t th = tht(i, j, k);
      real_t T = th * pk;
      
      real_t es = e0 * std::exp(L / Rv * ((T - T0) / (T0 * T)));
      real_t qvs = epsa * es / (p - es);
      
      real_t ss = std::min(qv(i, j, k) / qvs - 1, 0.);
      qvs = qv(i, j, k) / (1. + ss + 1e-16);

      real_t EP = 1./ rho(i, j, k) * ss * C *
                  std::pow(1e-3 * rho(i, j, k) * qrp, 0.525)
                  / 
                  (5.4e2 + 2.55e5 / (p * qvs));

      //EP = 0.;
      real_t dcol = AP + CP;
      real_t devp = EP;
      
      // limiting
      dcol = std::min(dcol,  qc(i, j, k) / this->dt + qrhs[2](i, j, k));
      devp = std::max(devp, -qr(i, j, k) / this->dt - qrhs[3](i, j, k) - dcol);

      // modifying forces
      this->rhs.at(ix::qv)(i, j, k)  = -devp;
      this->rhs.at(ix::qc)(i, j, k)  = -dcol;
      this->rhs.at(ix::qr)(i, j, k)  = devp + dcol;
      this->rhs.at(ix::thf)(i, j, k) = L / (cp * pk) * devp;
    }
  }
 
  void sedimentation(const typename parent_t::arr_t &qr)
  {
    //auto &qr = this->state(ix::qr);
    auto &rho = *this->mem->G;

    for (int i = this->i.first(); i <= this->i.last(); ++i)
    {
      for (int j = this->j.first(); j <= this->j.last(); ++j)
      {
        int lk = this->k.last();
        real_t rho_g = rho(i, j, 0);

        real_t rho_h = 0.5 * (rho(i, j, lk) + rho(i, j, lk - 1));
        real_t qr_h = 0.5 * (qr(i, j, lk) + qr(i, j, lk - 1));
        real_t vr_kmh = -36.34 * rho_h * this->dt / this->dk * 
              std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);
        
        //tmp1(i, j, lk)  = qr(i, j, lk) / (1 - vr_kmh / rho(i, j, lk));
       
        // no flux
        tmp1(i, j, lk)  = std::max(0., qr(i, j, lk)) / (1 + 2 *  vr_kmh / rho(i, j, lk));
        
        this->rhs.at(ix::qr)(i, j, lk)  += (tmp1(i, j, lk) - qr(i, j, lk)) / this->dt;
        col_sed(i, j, 0) = 0.5 * rho(i, j, lk) * (tmp1(i, j, lk) - qr(i, j, lk));

        for (int k = this->k.last() - 1; k > 0; --k)
        {
          real_t vr_kph = vr_kmh;

          rho_h = 0.5 * (rho(i, j, k) + rho(i, j, k - 1));
          qr_h = 0.5 * (qr(i, j, k) + qr(i, j, k - 1));
          
          real_t faux1 = -36.34 * 
              std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);

          
          vr_kmh = -36.34 * rho_h * this->dt / this->dk * 
              std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);

          tmp1(i, j, k) = (qr(i, j, k) - 1. / rho(i, j, k) * vr_kph * tmp1(i, j, k + 1)) / (1 - vr_kmh / rho(i, j, k));
          this->rhs.at(ix::qr)(i, j, k)  += (tmp1(i, j, k) - qr(i, j, k)) / this->dt;
          col_sed(i, j, 0) += rho(i, j, k) * (tmp1(i, j, k) - qr(i, j, k));
          
        }
        real_t vr_kph = vr_kmh;
          
        //tmp1(i, j, 0) = (qr(i, j, 0) - 1. / rho(i, j, 0) * vr_kph * tmp1(i, j, 1)) / (1 - vr_kmh / rho(i, j, 0));
       
        // no flux
        tmp1(i, j, 0) = (qr(i, j, 0) - 2. / rho(i, j, 0) * vr_kph * tmp1(i, j, 1));
        
        this->rhs.at(ix::qr)(i, j, 0)  += (tmp1(i, j, 0) - qr(i, j, 0)) / this->dt;
        col_sed(i, j, 0) += 0.5 * rho(i, j, 0) * (tmp1(i, j, 0) - qr(i, j, 0));
      }
    }

    //for (int i = this->i.first(); i <= this->i.last(); ++i)
    //for (int j = this->j.first(); j <= this->j.last(); ++j)
    //for (int k = this->k.first(); k <= this->k.last(); ++k)
    //{
    //  qr(i, j, k) = tmp1(i, j, k);
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
        for (int e = 0; e < 8; ++e)
        {
          rhs.at(e)(ijk) = 0;
        }

        break;
      }
      case (1):
      {
        // zero rhs for dynamic equations
        for (int e = 0; e < 4; ++e)
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
        for (int j = this->j.first(); j <= this->j.last(); ++j)
        for (int k = this->k.first(); k <= this->k.last(); ++k)
        {
          auto fqc_lm = std::max(-qc(i, j, k) / this->dt, this->rhs.at(ix::qc)(i, j, k));
          auto fqr_lm = std::max(-qr(i, j, k) / this->dt, this->rhs.at(ix::qr)(i, j, k));
          auto delql = fqc_lm - this->rhs.at(ix::qc)(i, j, k) + fqr_lm - this->rhs.at(ix::qr)(i, j, k);

          auto fqv_adj = this->rhs.at(ix::qv)(i, j, k) - delql;
          
          const real_t pk = pk_e(i, j, k);
          auto fthf_adj = this->rhs.at(ix::thf)(i, j, k) + 2 * L / (cp * pk) * delql;

          this->rhs.at(ix::qv)(i, j, k) = fqv_adj;
          this->rhs.at(ix::qc)(i, j, k) = fqc_lm;
          this->rhs.at(ix::qr)(i, j, k) = fqr_lm;
          this->rhs.at(ix::thf)(i, j, k) = fthf_adj;
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
    for (int e = 0; e < 4; ++e)
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
   
    if (v_mult > 0)
    {
      diffusion_cmpct();
    }
    
    this->state(ix::thf)(this->ijk) = this->state(ix::tht)(this->ijk) + tht_e(this->ijk);
    
    save_stats();
  }

  public:

  struct rt_params_t : parent_t::rt_params_t 
  { 
    real_t g, cp, Rd, Rv, L, e0, epsa, T0, buoy_eps, v_mult;
    std::string name;
  };

  // ctor
  supercell( 
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
    v_mult(p.v_mult),
    name(p.name),
    tht_b(args.mem->tmp[__FILE__][0][0]),
    tht_e(args.mem->tmp[__FILE__][1][0]),
    pk_e(args.mem->tmp[__FILE__][2][0]),
    qv_e(args.mem->tmp[__FILE__][3][0]),
    tmp1(args.mem->tmp[__FILE__][4][0]),
    tmp2(args.mem->tmp[__FILE__][4][1]),
    u_e(args.mem->tmp[__FILE__][5][0]),
    dtht_e(args.mem->tmp[__FILE__][6][0]),
    qr_est(args.mem->tmp[__FILE__][7][0]),
    col_sed(args.mem->tmp[__FILE__][8][0]),
    qrhs(args.mem->tmp[__FILE__][9]),
    grad_aux(args.mem->tmp[__FILE__][10]),
    ir(0, p.grid_size[0] - 1),
    jr(0, p.grid_size[1] - 1),
    kr(0, p.grid_size[2] - 1)
  {}

  static void alloc(typename parent_t::mem_t *mem, const int &n_iters)
  {
    parent_t::alloc(mem, n_iters);
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "tht_b");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "tht_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "pk_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "qv_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 2); // tmp1, tmp2
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "u_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "dtht_e");
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1); // qr_est
    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "", true); // col_sed
    parent_t::alloc_tmp_sclr(mem, __FILE__, 4, "qrhs");
    parent_t::alloc_tmp_vctr(mem, __FILE__); // grad_aux
  }
};
