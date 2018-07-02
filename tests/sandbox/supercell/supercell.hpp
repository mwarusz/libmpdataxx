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
  std::ofstream stat_file;
  real_t g, cp, Rd, Rv, L, e0, epsa, T0, buoy_eps;

  std::string name;
  typename parent_t::arr_t &tht_b, &tht_e, &pk_e, &qv_e, &tmp1, &tmp2, &u_e, &dtht_e;
  libmpdataxx::arrvec_t<typename parent_t::arr_t> &grad_aux;
    const libmpdataxx::rng_t ir;
    const libmpdataxx::rng_t jr;
    const libmpdataxx::rng_t kr;
  
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
      int np = (ir.last() + 1) * (jr.last() + 1) * 41;
      
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
      const libmpdataxx::rng_t kri(1, 39);
      
      auto totalws     = 0.5 * sum(rho(iri, jri, 0)   * (qv(iri, jri, 0)   + qc(iri, jri, 0)   + qr(iri, jri, 0))  ) +
                               sum(rho(iri, jri, kri) * (qv(iri, jri, kri) + qc(iri, jri, kri) + qr(iri, jri, kri))) +
                         0.5 * sum(rho(iri, jri, 40)  * (qv(iri, jri, 40)  + qc(iri, jri, 40)  + qr(iri, jri, 40)) ) ;
    
      stat_file.precision(18);
      //stat_file << this->timestep << ' '
      //          << w_min << ' ' << w_max << ' ' << w_avg << ' '
      //          << qc_min << ' ' << qc_max << ' ' << qc_avg << ' '
      //          << qr_min << ' ' << qr_max << ' ' << qr_avg << ' ' << totalws << std::endl;
      stat_file << "timestep " << this->timestep << std::endl
                << "u  " << u_min << ' ' << u_max << ' ' << u_avg << std::endl
                << "v  " << v_min << ' ' << v_max << ' ' << v_avg << std::endl
                << "w  " << w_min << ' ' << w_max << ' ' << w_avg << std::endl
                << "th " << tht_min << ' ' << tht_max << ' ' << tht_avg << std::endl
                << "qv " << qv_min << ' ' << qv_max << ' ' << qv_avg << std::endl
                << "qc " << qc_min << ' ' << qc_max << ' ' << qc_avg << std::endl
                << "qr " << qr_min << ' ' << qr_max << ' ' << qr_avg << ' ' << totalws << std::endl;
    }
    this->mem->barrier();
  }
  void print_stats(const std::string& str)
  {
    using namespace libmpdataxx::arakawa_c;
    this->mem->barrier();
    if (this->rank == 0)
    {
      auto tht_max     = max(this->state(ix::tht)(ir, jr, kr));
      auto tht_min     = min(this->state(ix::tht)(ir, jr, kr));
      
      auto qr_max     = max(this->state(ix::qr)(ir, jr, kr));
      auto qr_min     = min(this->state(ix::qr)(ir, jr, kr));
      auto qr_sum     = sum(this->state(ix::qr)(ir, jr, kr));
      
      auto qv_max     = max(this->state(ix::qv)(ir, jr, kr));
      auto qv_min     = min(this->state(ix::qv)(ir, jr, kr));
      auto qv_sum     = sum(this->state(ix::qv)(ir, jr, kr));
      
      auto qc_max     = max(this->state(ix::qc)(ir, jr, kr));
      auto qc_min     = min(this->state(ix::qc)(ir, jr, kr));
      auto qc_sum     = sum(this->state(ix::qc)(ir, jr, kr));

      auto u_min     = min(this->state(ix::u)(ir, jr, kr));
      auto u_max     = max(this->state(ix::u)(ir, jr, kr));
      auto u_sum     = sum(this->state(ix::u)(ir, jr, kr));

      auto w_min     = min(this->state(ix::w)(ir, jr, kr));
      auto w_max     = max(this->state(ix::w)(ir, jr, kr));
      auto w_sum     = sum(this->state(ix::w)(ir, jr, kr));
      auto w_loc     = maxIndex(this->state(ix::w)(ir, jr, kr));
      std::cout << str << std::endl;
      std::cout << "ux: " << u_min << ' ' << u_max << ' ' << u_sum << std::endl;
      std::cout << "uz: " << w_min << ' ' << w_max << ' ' << w_sum << std::endl;
      std::cout << "qv: " << qv_min << ' ' << qv_max << ' ' << qv_sum << std::endl;
      std::cout << "qc: " << qc_min << ' ' << qc_max << ' ' << qc_sum << std::endl;
      std::cout << "qr: " << qr_min << ' ' << qr_max << ' ' << qr_sum << std::endl;
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
   
    tmp1(this->ijk) = u(this->ijk) - u_e(this->ijk);
    this->rhs.at(ix::u)(this->ijk) += 2.0 * vlap_cmpct(tmp1, 500., this->ijk, ijkm, this->dijk);
    this->rhs.at(ix::v)(this->ijk) += 2.0 * vlap_cmpct(v, 500., this->ijk   , ijkm, this->dijk);
    this->rhs.at(ix::w)(this->ijk) += 2.0 * vlap_cmpct(w, 500., this->ijk   , ijkm, this->dijk);
    
    tmp2(this->ijk) = qv(this->ijk) - qv_e(this->ijk);
    
    tmp1(this->ijk) = 2.0 * vlap_cmpct(tht, 1500., this->ijk, ijkm, this->dijk);
    this->rhs.at(ix::tht)(this->ijk) += tmp1(this->ijk);

    // add diffusion force to full tht forcings
    this->rhs.at(ix::thf)(this->ijk) += tmp1(this->ijk);


    this->rhs.at(ix::qv)(this->ijk)  += 2.0 * vlap_cmpct(tmp2, 1500., this->ijk, ijkm, this->dijk);
    this->rhs.at(ix::qc)(this->ijk)  += 2.0 * vlap_cmpct(qc , 1500., this->ijk , ijkm, this->dijk);
    this->rhs.at(ix::qr)(this->ijk)  += 2.0 * vlap_cmpct(qr , 1500., this->ijk , ijkm, this->dijk);
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
      this->rhs.at(ix::tht)(i, j, k) +=   2 * L / (cp * pk) * delta / this->dt;
     
    }
  }
  
  template<typename tht_t>
  void update_moist_forces(tht_t &tht)
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
      qc(i, j, k) = std::max(0., qc(i, j, k));
      qr(i, j, k) = std::max(0., qr(i, j, k));

      real_t k1 = 1e-3;
      real_t k2 = 2.2;
      real_t qct = 1e-3;

      real_t AP = std::max(0., k1 * (qc(i, j, k) - qct));
      real_t CP = k2 * qc(i, j, k) * std::pow(qr(i, j, k), 0.875);
      
      real_t C = 1.6 + 124.9 * std::pow(1e-3 * rho(i, j, k) * qr(i, j, k), 0.2046);

      real_t pk = pk_e(i, j, k);
      real_t p = pk2p(pk);
      real_t th = tht(i, j, k);
      real_t T = th * pk;
      
      real_t es = e0 * std::exp(L / Rv * ((T - T0) / (T0 * T)));
      real_t qvs = epsa * es / (p - es);
      
      real_t ss = std::min(qv(i, j, k) / qvs - 1, 0.);
      qvs = qv(i, j, k) / (1. + ss);

      //if (i == 42 && j == 42)
      //{
      //  std::cout << k << ' ' << p << ' ' << ss << ' ' << qvs << std::endl;
      //}

      real_t EP = 1./ rho(i, j, k) * ss * C *
                  std::pow(1e-3 * rho(i, j, k) * qr(i, j, k), 0.525)
                  / 
                  (5.4e2 + 2.55e5 / (p * qvs));

      //EP = 0.;
      real_t dcol = AP + CP;
      real_t devp = EP;
      
      // limiting
      dcol = std::min(dcol, qc(i, j, k) / this->dt);
      devp = std::max(devp, -qr(i, j, k) / this->dt - dcol);

      // modifying forces
      this->rhs.at(ix::qv)(i, j, k)  = -devp;
      this->rhs.at(ix::qc)(i, j, k)  = -dcol;
      this->rhs.at(ix::qr)(i, j, k)  = devp + dcol;
      this->rhs.at(ix::tht)(i, j, k) = L / (cp * pk) * devp;
      
      //precip
      //real_t rho_g = rho(i, j, 0);
      //real_t rho_k = rho(i, j, k);
      //real_t rho_kp1 = rho(i, j, k + 1);
      //real_t qr_k = qr(i, j, k);
      //real_t qr_kp1 = qr(i, j, k + 1);

      //real_t vr_k = 36.34 * std::pow(1e-3 * rho_k * qr_k, 0.1364) * std::pow(rho_k / rho_g, -0.5);
      //real_t vr_kp1 = 36.34 * std::pow(1e-3 * rho_kp1 * qr_kp1, 0.1364) * std::pow(rho_kp1 / rho_g, -0.5);

      //real_t sed = (rho_kp1 * vr_kp1 * qr_kp1 - rho_k * vr_k * qr_k) / (rho_k * this->dk);
      //sed = (k == this->k.last()) ? - qr_k * vr_k / (0.5 * this->dk) : sed;
      //this->rhs.at(ix::qr)(i, j, k)  += 2 * sed;
    }
  }
 
  void sedimentation()
  {
    auto &qr = this->state(ix::qr);
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
        
        tmp1(i, j, lk)  = qr(i, j, lk) / (1 - vr_kmh / rho(i, j, lk));
        
        this->rhs.at(ix::qr)(i, j, lk)  += (tmp1(i, j, lk) - qr(i, j, lk)) / this->dt;

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
          
          //if (i == 42 && j == 42)
          //{
          //  std::cout << "test " << k << ' ' << rho_h << ' ' << qr_h << ' ' << tmp1(i, j, k) << ' ' << faux1 << std::endl;
          //}
        }
        real_t vr_kph = vr_kmh;
          
        tmp1(i, j, 0) = (qr(i, j, 0) - 1. / rho(i, j, 0) * vr_kph * tmp1(i, j, 1)) / (1 - vr_kmh / rho(i, j, 0));
        
        this->rhs.at(ix::qr)(i, j, 0)  += (tmp1(i, j, 0) - qr(i, j, 0)) / this->dt;
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
      stat_file.open(name.c_str());
    }

    calc_dtht_e();
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
    parent_t::update_rhs(rhs, dt, at); 
    
    auto &tht = this->state(ix::tht);
    auto &thf = this->state(ix::thf);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    
    //tmp2(this->ijk) = this->state(ix::tht)(this->ijk) + tht_e(this->ijk);

    //if (at > 0)
    //{
    //  //update_moist_forces();
    //}
    
    //this->state(ix::tht)(this->ijk) = tmp2(this->ijk) - tht_e(this->ijk);

    const auto &ijk = this->ijk;
    auto ix_w = this->vip_ixs[ct_params_t::n_dims - 1];
    const auto &tht_abs = *this->mem->vab_coeff;

    switch(at)
    {
      case (0):
      {
        //rhs.at(ix::tht)(ijk) += -w(ijk) * this->dtht_e(ijk);

        //rhs.at(ix_w)(ijk) += this->g * (
        //        ( tht(ijk) / this->tht_b(ijk)
        //        + buoy_eps * (this->state(ix::qv)(ijk) - this->qv_e(ijk))
        //        - this->state(ix::qc)(ijk) - this->state(ix::qr)(ijk) 
        //        ));

        break;
      }
      case (1):
      {
        tmp2(ijk) = tht(ijk) + tht_e(ijk);

        update_moist_forces(thf);
       
        // apply precipitation temp force to both full tht and tht perturbation
        this->state(ix::thf)(this->ijk) += this->dt * this->rhs.at(ix::tht)(this->ijk);
        this->state(ix::tht)(this->ijk) += this->dt * this->rhs.at(ix::tht)(this->ijk);

        this->state(ix::qv)(this->ijk)  += this->dt * this->rhs.at(ix::qv)(this->ijk);
        this->state(ix::qc)(this->ijk)  += this->dt * this->rhs.at(ix::qc)(this->ijk);
        this->state(ix::qr)(this->ijk)  += this->dt * this->rhs.at(ix::qr)(this->ijk);
        
        this->rhs.at(ix::tht)(this->ijk) = 0;
        this->rhs.at(ix::qv)(this->ijk) = 0;
        this->rhs.at(ix::qc)(this->ijk) = 0;
        this->rhs.at(ix::qr)(this->ijk) = 0;        

        this->state(ix::qr)(this->ijk) = max(0., this->state(ix::qr)(this->ijk));
        
        sedimentation();
        this->state(ix::qr)(this->ijk) += this->dt * this->rhs.at(ix::qr)(this->ijk);
        this->rhs.at(ix::qr)(this->ijk) = 0;        
        
        saturation_adjustment(thf);
        
        // add condensation force to full tht forcings
        this->rhs.at(ix::thf)(this->ijk) += this->rhs.at(ix::tht)(this->ijk);

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

  void hook_mixed_rhs()
  {
    this->state(ix::w)(this->ijk) += 0.5 * this->dt * this->rhs.at(ix::w)(this->ijk);
    //this->mem->barrier();
    //if (this->rank == 0)
    //{
    //  auto w_max  = max(this->state(ix::w)(ir, jr, kr));
    //  std::cout << "after exp b: " << this->timestep << ' ' << w_max << std::endl;
    //}
    //this->mem->barrier();
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
    
    //this->mem->barrier();
    //if (this->rank == 0)
    //{
    //  auto w_max  = max(this->state(ix::w)(ir, jr, kr));
    //  std::cout << "after prs solver: " << this->timestep << ' ' << w_max << std::endl;
    //}
    //this->mem->barrier();
  }
  
  void hook_ante_step()
  {
    save_stats();

    //if (this->rank == 0) std::cout << "itime, time: " << this->timestep+1 << ' ' << this->time / 60.  << std::endl;
    //print_stats("before half");
    parent_t::hook_ante_step();

    this->state(ix::qc)(this->ijk) = max(0., this->state(ix::qc)(this->ijk));
    this->state(ix::qr)(this->ijk) = max(0., this->state(ix::qr)(this->ijk));
    

    this->mem->barrier();
  }


  void hook_post_step()
  {
    parent_t::hook_post_step();
    
    diffusion_cmpct();
    
    this->state(ix::thf)(this->ijk) = this->state(ix::tht)(this->ijk) + tht_e(this->ijk);
  }

  public:

  struct rt_params_t : parent_t::rt_params_t 
  { 
    real_t g, cp, Rd, Rv, L, e0, epsa, T0, buoy_eps;
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
    name(p.name),
    tht_b(args.mem->tmp[__FILE__][0][0]),
    tht_e(args.mem->tmp[__FILE__][1][0]),
    pk_e(args.mem->tmp[__FILE__][2][0]),
    qv_e(args.mem->tmp[__FILE__][3][0]),
    tmp1(args.mem->tmp[__FILE__][4][0]),
    tmp2(args.mem->tmp[__FILE__][4][1]),
    u_e(args.mem->tmp[__FILE__][5][0]),
    dtht_e(args.mem->tmp[__FILE__][6][0]),
    grad_aux(args.mem->tmp[__FILE__][7]),
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
    parent_t::alloc_tmp_vctr(mem, __FILE__); // grad_aux
  }
};
