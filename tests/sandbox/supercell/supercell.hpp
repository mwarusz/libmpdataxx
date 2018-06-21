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

  bool buoy_filter;
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

  void adjust_GC(int e) override
  {
    //using namespace libmpdataxx::arakawa_c;
    //
    //auto &tht = this->state(ix::tht);
    //auto &qr = this->state(ix::qr);
    //auto &rho = *this->mem->G;

    //if (e == 6)
    //{
    //  for (int i = this->i.first(); i <= this->i.last(); ++i)
    //  {
    //    for (int j = this->j.first(); j <= this->j.last(); ++j)
    //    {
    //      real_t rho_h = 0.5 * (3 * rho(i, j, 0) - rho(i, j, 1));
    //      real_t qr_h = std::max(0.5 * (3 * qr(i, j, 0) - qr(i, j, 1)), 0.);
    //      real_t rho_g = rho(i, j, 0);
    //      this->mem->GC[2](i, j, 0 - h) -= 36.34 * rho_h * this->dt / this->dk * 
    //        std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);
    //      for (int k = this->k.first(); k <= this->k.last() - 1; ++k)
    //      {
    //        rho_h = 0.5 * (rho(i, j, k + 1) + rho(i, j, k));
    //        qr_h = 0.5 * (qr(i, j, k + 1) + qr(i, j, k));
    //        this->mem->GC[2](i, j, k + h) -= 36.34 * rho_h * this->dt / this->dk * 
    //          std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);
    //      }
    //      this->mem->GC[2](i, j, this->k.last() + h) += 36.34 * rho_h * this->dt / this->dk *
    //        std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);
    //    }
    //  }
    //}
  }

  real_t pk2p(real_t pk)
  {
    const real_t p0 = 1e5;
    return std::pow(pk, cp / Rd) * p0;
  }

  template <int nd = ct_params_t::n_dims> 
  void calc_dtht_e(typename std::enable_if<nd == 3>::type* = 0)
  {
    this->xchng_sclr(this->tht_e, this->ijk);
    this->dtht_e(this->ijk) = libmpdataxx::formulae::nabla::grad<2>(this->tht_e, this->k, this->i, this->j, this->dk);
  }

  template <int nd = ct_params_t::n_dims> 
  void filter(typename std::enable_if<nd == 3>::type* = 0)
  {
    const auto &i(this->i), &j(this->j), &k(this->k);
    this->xchng_sclr(tmp1, this->ijk);
    tmp2(i, j, k) = 0.25 * (tmp1(i, j, k + 1) + 2 * tmp1(i, j, k) + tmp1(i, j, k - 1));
  }
  
  // helpers for buoyancy forces
  template<class ijk_t>
  inline auto buoy_at_0(const ijk_t &ijk)
  {
    return libmpdataxx::return_helper<libmpdataxx::rng_t>(
      this->g * (
                  (this->state(ix::tht)(ijk) - this->tht_e(ijk)) / this->tht_b(ijk)
                + buoy_eps * (this->state(ix::qv)(ijk) - this->qv_e(ijk))
                - this->state(ix::qc)(ijk) - this->state(ix::qr)(ijk) 
                )
    );
  }
  
  template<class ijk_t>
  inline auto buoy_at_1(const ijk_t &ijk)
  {
    const auto &tht_abs = *this->mem->vab_coeff;
    return libmpdataxx::return_helper<libmpdataxx::rng_t>(
      this->g * (
                  ( (this->state(ix::tht)(ijk) + 0.5 * this->dt * tht_abs(ijk) * this->tht_e(ijk))
                    / (1 + 0.5 * this->dt * tht_abs(ijk))
                   - this->tht_e(ijk)
                  ) / this->tht_b(ijk)
                + buoy_eps * (this->state(ix::qv)(ijk) - this->qv_e(ijk))
                - this->state(ix::qc)(ijk) - this->state(ix::qr)(ijk) 
                )
    );
  }

  virtual void normalize_vip(const libmpdataxx::arrvec_t<typename parent_t::arr_t> &v)
  {
    if (ct_params_t::impl_tht)
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
    else
    {
      parent_t::normalize_vip(v);
    }
  }
  void vip_rhs_expl_calc()
  {
    this->state(ix::u)(this->ijk) -= u_e(this->ijk);
    parent_t::vip_rhs_expl_calc();
    this->state(ix::u)(this->ijk) += u_e(this->ijk);
  }

  template<typename ijk_t>
  auto vlap(
    typename parent_t::arr_t &arr, 
    real_t visc,
    const ijk_t &ijk, 
    const std::array<real_t, parent_t::n_dims>& dijk
  ) return_macro(
    this->xchng_pres(arr, ijk);
    libmpdataxx::formulae::nabla::calc_grad<parent_t::n_dims>(this->lap_tmp, arr, ijk, dijk);
    if (this->mem->G)
    {
      for (int d = 0; d < parent_t::n_dims; ++d)
      {
        this->lap_tmp[d](this->ijk) *= visc * (*this->mem->G)(this->ijk);
      }
    }
    for (int d = 0; d < parent_t::n_dims; ++d)
    {
      this->xchng_pres(this->lap_tmp[d], ijk);
    }
    ,
    libmpdataxx::formulae::nabla::div<parent_t::n_dims>(this->lap_tmp, ijk, dijk)
    / libmpdataxx::formulae::G<ct_params_t::opts>(*this->mem->G, this->ijk)
  )
  
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
  
  void diffusion_simple()
  {
    auto &u = this->state(ix::u);
    auto &v = this->state(ix::v);
    auto &w = this->state(ix::w);


    auto &tht = this->state(ix::tht);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    
    tmp1(this->ijk) = u(this->ijk) - u_e(this->ijk);
    this->rhs.at(ix::u)(this->ijk) += 2.0 * vlap(tmp1, 500., this->ijk, this->dijk);
    this->rhs.at(ix::v)(this->ijk) += 2.0 * vlap(v, 500., this->ijk, this->dijk);
    this->rhs.at(ix::w)(this->ijk) += 2.0 * vlap(w, 500., this->ijk, this->dijk);
    
    tmp1(this->ijk) = tht(this->ijk) - tht_e(this->ijk);
    tmp2(this->ijk) = qv(this->ijk) - qv_e(this->ijk);
    
    this->rhs.at(ix::tht)(this->ijk) += 2.0 * vlap(tmp1, 1500., this->ijk, this->dijk);
    this->rhs.at(ix::qv)(this->ijk)  += 2.0 * vlap(tmp2, 1500., this->ijk, this->dijk);
    this->rhs.at(ix::qc)(this->ijk)  += 2.0 * vlap(qc , 1500., this->ijk, this->dijk);
    this->rhs.at(ix::qr)(this->ijk)  += 2.0 * vlap(qr , 1500., this->ijk, this->dijk);
                                                
  }
  
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
    
    tmp1(this->ijk) = tht(this->ijk) - tht_e(this->ijk);
    tmp2(this->ijk) = qv(this->ijk) - qv_e(this->ijk);
    
    this->rhs.at(ix::tht)(this->ijk) += 2.0 * vlap_cmpct(tmp1, 1500., this->ijk, ijkm, this->dijk);
    this->rhs.at(ix::qv)(this->ijk)  += 2.0 * vlap_cmpct(tmp2, 1500., this->ijk, ijkm, this->dijk);
    this->rhs.at(ix::qc)(this->ijk)  += 2.0 * vlap_cmpct(qc , 1500., this->ijk , ijkm, this->dijk);
    this->rhs.at(ix::qr)(this->ijk)  += 2.0 * vlap_cmpct(qr , 1500., this->ijk , ijkm, this->dijk);
                                                
    //tmp2(this->ijk) = vlap_cmpct(tmp1, 1500., this->ijk, ijkm, this->dijk);
    //this->mem->barrier();
    //if (this->rank == 0)
    //{
    //  auto dtht_max     = max(tmp2(ir, jr, kr));
    //  auto dtht_min     = min(tmp2(ir, jr, kr));

    //  std::cout << this->timestep << ' ' << dtht_min << ' ' << dtht_max << std::endl;
    //}
    //this->mem->barrier();

    //this->rhs.at(ix::tht)(this->ijk) += 2.0 * tmp2(this->ijk);
  }
  void diffusion_old()
  {
    //const libmpdataxx::rng_t ir(0, 128);
    //const libmpdataxx::rng_t jr(0, 128);
    //const libmpdataxx::rng_t kr(1, 39);
    //this->mem->barrier();
    //if (this->rank == 0)
    //{
    //  std::cout << "timestep: " << this->timestep << std::endl;
    //  std::cout << "dt: " << this->dt << std::endl;
    //  std::cout << "ftht max: " << max(abs(this->rhs.at(ix::tht)(ir, jr, kr))) << std::endl;
    //  std::cout << "fqv  max: " << max(abs(this->rhs.at(ix::qv)(ir, jr, kr))) << std::endl;
    //  std::cout << "fqc  max: " << max(abs(this->rhs.at(ix::qc)(ir, jr, kr))) << std::endl;
    //  std::cout << "fqr  max: " << max(abs(this->rhs.at(ix::qr)(ir, jr, kr))) << std::endl;
    //}
    //this->mem->barrier();

    const auto &ijk = this->ijk;

    auto &tht = this->state(ix::tht);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);

    using namespace libmpdataxx::formulae;

    // tht
    if (!ct_params_t::impl_tht)
    {
      tmp1(ijk) = tht(ijk) - tht_e(ijk);
      this->xchng_pres(tmp1, ijk);
      nabla::calc_grad_cmpct<parent_t::n_dims>(grad_aux, tmp1, ijk, this->ijkm, this->dijk);
    }
    else
    {
      this->xchng_pres(tht, ijk);
      nabla::calc_grad_cmpct<parent_t::n_dims>(grad_aux, tht, ijk, this->ijkm, this->dijk);
    }

    //using namespace libmpdataxx::arakawa_c;
    //this->mem->barrier();
    //if (this->rank == 0)
    //{
    //  std::cout << "tht_dz  max: " << max(abs(grad_aux[2](ir, jr, kr + h))) << std::endl;
    //  std::cout << "tht_dzz max: " << max(abs(grad_aux[2](ir, jr, kr + h) - grad_aux[2](ir, jr, kr - h))) / this->dk << std::endl;
    //}
    //this->mem->barrier();

    this->mem->barrier();
    stress::multiply_vctr_cmpct<ct_params_t::n_dims, ct_params_t::opts>(grad_aux,
                                                                        3.0 * this->eta,
                                                                        *this->mem->G,
                                                                        ijk);

    this->xchng_vctr_alng(grad_aux);
    this->rhs.at(ix::tht)(ijk) += 2.0 * stress::flux_div_cmpct<parent_t::n_dims, ct_params_t::opts>(grad_aux,
                                                                                                    *this->mem->G,
                                                                                                    ijk,
                                                                                                    this->dijk);
    // qv
    tmp1(ijk) = qv(ijk) - qv_e(ijk);
    this->xchng_pres(tmp1, ijk);
    nabla::calc_grad_cmpct<parent_t::n_dims>(grad_aux, tmp1, ijk, this->ijkm, this->dijk);
    this->mem->barrier();
    stress::multiply_vctr_cmpct<ct_params_t::n_dims, ct_params_t::opts>(grad_aux,
                                                                        3.0 * this->eta,
                                                                        *this->mem->G,
                                                                        ijk);

    this->xchng_vctr_alng(grad_aux);
    this->rhs.at(ix::qv)(ijk) += 2.0 * stress::flux_div_cmpct<parent_t::n_dims, ct_params_t::opts>(grad_aux,
                                                                                                   *this->mem->G,
                                                                                                   ijk,
                                                                                                   this->dijk);
    // qc
    this->xchng_pres(qc, ijk);
    nabla::calc_grad_cmpct<parent_t::n_dims>(grad_aux, qc, ijk, this->ijkm, this->dijk);
    this->mem->barrier();
    stress::multiply_vctr_cmpct<ct_params_t::n_dims, ct_params_t::opts>(grad_aux,
                                                                        3.0 * this->eta,
                                                                        *this->mem->G,
                                                                        ijk);

    this->xchng_vctr_alng(grad_aux);
    this->rhs.at(ix::qc)(ijk) += 2.0 * stress::flux_div_cmpct<parent_t::n_dims, ct_params_t::opts>(grad_aux,
                                                                                                   *this->mem->G,
                                                                                                   ijk,
                                                                                                   this->dijk);
    // qr
    this->xchng_pres(qr, ijk);
    nabla::calc_grad_cmpct<parent_t::n_dims>(grad_aux, qr, ijk, this->ijkm, this->dijk);
    this->mem->barrier();
    stress::multiply_vctr_cmpct<ct_params_t::n_dims, ct_params_t::opts>(grad_aux,
                                                                        3.0 * this->eta,
                                                                        *this->mem->G,
                                                                        ijk);

    this->xchng_vctr_alng(grad_aux);
    this->rhs.at(ix::qr)(ijk) += 2.0 * stress::flux_div_cmpct<parent_t::n_dims, ct_params_t::opts>(grad_aux,
                                                                                                   *this->mem->G,
                                                                                                   ijk,
                                                                                                   this->dijk);
   
    //this->mem->barrier();
    //if (this->rank == 0)
    //{
    //  std::cout << "ftht max: " << max(abs(this->rhs.at(ix::tht)(ir, jr, kr))) << std::endl;
    //  std::cout << "fqv  max: " << max(abs(this->rhs.at(ix::qv)(ir, jr, kr))) << std::endl;
    //  std::cout << "fqc  max: " << max(abs(this->rhs.at(ix::qc)(ir, jr, kr))) << std::endl;
    //  std::cout << "fqr  max: " << max(abs(this->rhs.at(ix::qr)(ir, jr, kr))) << std::endl;
    //}
    //this->mem->barrier();
  }

  void saturation_adjustment()
  {
    auto &tht = this->state(ix::tht);
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
      
      //if (i == 0 && j == 0)
      //{
      //  std::cout << "test " << k << " " << qvs << std::endl;
      //}
     
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
      this->rhs.at(ix::qv)(i, j, k)  = - 2 * delta / this->dt;
      this->rhs.at(ix::qc)(i, j, k)  =   2 * delta / this->dt;
      this->rhs.at(ix::tht)(i, j, k) =   2 * L / (cp * pk) * delta / this->dt;
     
      // remove zeros
      qv(i, j, k) = std::max(0., qv(i, j, k));
      qc(i, j, k) = std::max(0., qc(i, j, k));
      qr(i, j, k) = std::max(0., qr(i, j, k));
    }
  }
  
  void update_moist_forces()
  {

    auto &tht = this->state(ix::tht);
    auto &qv = this->state(ix::qv);
    auto &qc = this->state(ix::qc);
    auto &qr = this->state(ix::qr);
    auto &rho = *this->mem->G;
    
    //real_t max_devp = 0.;
    //real_t min_devp = 0.;
    //real_t max_rhs = 0.;

    for (int i = this->i.first(); i <= this->i.last(); ++i)
    for (int j = this->j.first(); j <= this->j.last(); ++j)
    for (int k = this->k.first(); k <= this->k.last(); ++k)
    {
      real_t k1 = 1e-3;
      real_t k2 = 2.2;
      real_t qct = 1e-3;
      

      real_t AP = std::max(0., k1 * (qc(i, j, k) - qct));
      //real_t AP = 0.;
      real_t CP = k2 * qc(i, j, k) * std::pow(qr(i, j, k), 0.875);
      //real_t CP = 0.;
      
      real_t C = 1.6 + 124.9 * std::pow(1e-3 * rho(i, j, k) * qr(i, j, k), 0.2046);

      real_t pk = pk_e(i, j, k);
      real_t p = pk2p(pk);
      real_t th = tht(i, j, k);
      real_t T = th * pk;
      
      real_t es = e0 * std::exp(L / Rv * ((T - T0) / (T0 * T)));
      real_t qvs = epsa * es / (p - es);

      real_t EP = 1./ rho(i, j, k) * (qv(i, j, k) / qvs - 1) * C *
                  std::pow(1e-3 * rho(i, j, k) * qr(i, j, k), 0.525)
                  / 
                  (5.4e2 + 2.55e5 / (p * qvs));

      //real_t EP = 0.;
      // limiting
      real_t dcol = 2 * (AP + CP);
      //dcol = std::min(dcol, 2. / this->dt * qc(i, j, k) + this->rhs.at(ix::qc)(i, j, k));
      real_t devp = 2 * EP;
      //real_t devp = 0.;
      //devp = std::max(devp, -2. / this->dt * qr(i, j, k) - dcol);
      
      //precip
      real_t sed = 0.;
      //real_t rho_g = rho(i, j, 0);
      //real_t rho_k = rho(i, j, k);
      //real_t rho_kp1 = rho(i, j, k + 1);
      //real_t qr_k = qr(i, j, k);
      //real_t qr_kp1 = qr(i, j, k + 1);

      ////if (rho_kp1 * qr_kp1 < 0)
      ////{
      ////  std::cout << "BUG1" << std::endl;
      ////  std::cout << k  << std::endl;
      ////  std::cout << rho_kp1 << std::endl;
      ////  std::cout << qr_kp1 << std::endl;
      ////}

      //real_t vr_k = 36.34 * std::pow(1e-3 * rho_k * qr_k, 0.1346) * std::pow(rho_k / rho_g, -0.5);
      //real_t vr_kp1 = 36.34 * std::pow(1e-3 * rho_kp1 * qr_kp1, 0.1346) * std::pow(rho_kp1 / rho_g, -0.5);

      //sed = (rho_kp1 * vr_kp1 * qr_kp1 - rho_k * vr_k * qr_k) / (rho_k * this->dk);
      //sed = ((k == this->k.last()) ? - qr_k * vr_k / (0.5 * this->dk) : sed);


      // modifying forces
      this->rhs.at(ix::qv)(i, j, k)  += -devp;
      this->rhs.at(ix::qc)(i, j, k)  += -dcol;
      this->rhs.at(ix::qr)(i, j, k)  += (devp + dcol + 2 * sed);
      this->rhs.at(ix::tht)(i, j, k) += L / (cp * pk) * devp;
      
      //max_devp = std::max(devp + dcol, max_devp);
      //min_devp = std::min(sed, min_devp);
      //max_rhs = std::max(this->rhs.at(ix::qr)(i, j, k) , max_rhs);
    }
    
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
        
        this->rhs.at(ix::qr)(i, j, lk)  += 2 * (tmp1(i, j, lk) - qr(i, j, lk)) / this->dt;

        for (int k = this->k.last() - 1; k > 0; --k)
        {
          real_t vr_kph = vr_kmh;

          rho_h = 0.5 * (rho(i, j, k) + rho(i, j, k - 1));
          qr_h = 0.5 * (qr(i, j, k) + qr(i, j, k - 1));
          
          vr_kmh = -36.34 * rho_h * this->dt / this->dk * 
              std::pow(1e-3 * rho_h * qr_h, 0.1346) * std::pow(rho_h / rho_g, -0.5);

          tmp1(i, j, k) = (qr(i, j, k) - 1. / rho(i, j, k) * vr_kph * tmp1(i, j, k + 1)) / (1 - vr_kmh / rho(i, j, k));
          this->rhs.at(ix::qr)(i, j, k)  += 2 * (tmp1(i, j, k) - qr(i, j, k)) / this->dt;
        }
        real_t vr_kph = vr_kmh;
          
        tmp1(i, j, 0) = (qr(i, j, 0) - 1. / rho(i, j, 0) * vr_kph * tmp1(i, j, 1)) / (1 - vr_kmh / rho(i, j, 0));
        
        this->rhs.at(ix::qr)(i, j, 0)  += 2 * (tmp1(i, j, 0) - qr(i, j, 0)) / this->dt;
      }
    }

    //if (this->rank == 0) std::cout << "minmax_devp: " << min_devp << ' ' << max_devp << ' ' << max_rhs << std::endl;
  }

  template <typename arr_1d_t>
  void kessler(arr_1d_t qv, arr_1d_t qc, arr_1d_t qr, arr_1d_t theta, arr_1d_t rho, arr_1d_t pk, int nz)
  {
    const real_t dt = this->dt;
    const real_t dz = this->dk;

    blitz::Array<real_t, 1> r(nz), rhalf(nz), velqr(nz), sed(nz), pc(nz);

    const real_t f2x = 17.27;
    const real_t f5 = 237.3 * f2x * 2500000 / 1003.;
    const real_t xk = 0.2875;
    const real_t psl = 1000.0;
    const real_t rhoqr = 1000.0;

    for (int k = 0; k < nz; ++k)
    {
      r(k) = 0.001 * rho(k);
      rhalf(k) = std::sqrt(rho(0) / rho(k));
      pc(k) = 3.8 / (std::pow(pk(k), 1. / xk) * psl);

      velqr(k) = 36.34 * std::pow(qr(k) * r(k), 0.1364) * rhalf(k);
    }

    auto dt_max = dt;
    
    for (int k = 0; k < nz - 1; ++k)
    {
      if (velqr(k) != 0.)
      {
        dt_max = std::min(dt_max, 0.8 * dz / velqr(k));
      }
    }

    int rainsplit = std::ceil(dt / dt_max);
    const auto dt0 = dt / rainsplit;

    real_t precl = 0.;

    for (int m = 0; m < rainsplit; ++m)
    {
      precl += rho(0) * qr(0) * velqr(0) / rhoqr;

      for (int k = 0; k < nz - 1; ++k)
      {
        sed(k) = dt0 * ((r(k + 1) * qr(k + 1) * velqr(k + 1) - r(k) * qr(k) * velqr(k)) / (r(k) * dz));
      }
      sed(nz - 1) = -dt0 * qr(nz - 1) * velqr(nz - 1) / (0.5 * dz);
      
      for (int k = 0; k < nz; ++k)
      {

        const real_t qrprod = qc(k) - (qc(k) - dt0 * std::max(.001 * (qc(k) - .001), 0.)) / (1. + dt0 * 2.2 * std::pow(qr(k), .875));
        qc(k) = std::max(qc(k) - qrprod, 0.);
        qr(k) = std::max(qr(k) + qrprod + sed(k), 0.);

        const real_t qvs = pc(k) * std::exp(f2x * (pk(k) * theta(k) - 273.)
                                            / (pk(k) * theta(k)- 36.));
        const real_t prod = (qv(k) - qvs) / (1. + qvs * f5 / std::pow(pk(k) * theta(k) - 36., 2));

        const real_t ern = std::min({dt0 * (((1.6 + 124.9 * std::pow(r(k) * qr(k), .2046))
              * std::pow(r(k) * qr(k), .525)) / (2550000. * pc(k) / (3.8 * qvs) + 540000))
              * (std::max(0., qvs - qv(k)) / (r(k) * qvs))
              ,
              std::max(-prod - qc(k), 0.)
              ,
              qr(k)
              });

        theta(k) += 2500000 / (1003. * pk(k)) * (std::max(prod, -qc(k)) - ern);
        qv(k) = std::max(qv(k) - std::max(prod, -qc(k)) + ern, 0.);
        qc(k) += std::max(prod, -qc(k));
        qr(k) -= ern;
      }

      if (m != rainsplit - 1)
      {
        for (int k = 0; k < nz; ++k)
        {
          velqr(k) = 36.34 * std::pow(qr(k) * r(k), 0.1364) * rhalf(k);
        }
      }
    }

    precl /= rainsplit;
  }

  void hook_ante_loop(const typename parent_t::advance_arg_t nt) 
  {
    if (this->rank == 0)
    {
      stat_file.open(name.c_str());
    }

    if (ct_params_t::impl_tht)
    {
      calc_dtht_e();
    }
    parent_t::hook_ante_loop(nt);
  }
  
  // explicit forcings 
  void update_rhs(
    libmpdataxx::arrvec_t<
      typename parent_t::arr_t
    > &rhs, 
    const real_t &dt, 
    const int &at 
  ) {
    parent_t::update_rhs(rhs, dt, at); 
    
    if (ct_params_t::impl_tht)
    {
      tmp2(this->ijk) = this->state(ix::tht)(this->ijk) + tht_e(this->ijk);
    } 
    
    //for (int i = this->i.first(); i <= this->i.last(); ++i)
    //{
    //  for (int j = this->j.first(); j <= this->j.last(); ++j)
    //  {
    //    auto 
    //    rho_c    = (*this->mem->G)(i, j, this->k).reindex(0),
    //    qv_c     = this->state(ix::qv)(i, j, this->k).reindex(0),
    //    qc_c     = this->state(ix::qc)(i, j, this->k).reindex(0),
    //    qr_c     = this->state(ix::qr)(i, j, this->k).reindex(0),
    //    pk_c     = pk_e(i, j, this->k).reindex(0);
    //   
    //    if (ct_params_t::impl_tht)
    //    {
    //      auto
    //      tht_c    = tmp2(i, j, this->k).reindex(0);
    //      kessler(qv_c, qc_c, qr_c, tht_c, rho_c, pk_c, this->mem->grid_size[2].last() + 1);
    //    }
    //    else
    //    {
    //      auto
    //      tht_c    = this->state(ix::tht)(i, j, this->k).reindex(0);
    //      kessler(qv_c, qc_c, qr_c, tht_c, rho_c, pk_c, this->mem->grid_size[2].last() + 1);
    //    }
    //  }
    //}

    if (at > 0)
    {
      saturation_adjustment();
      //update_moist_forces();
    }
    
    if (ct_params_t::impl_tht)
    {
      this->state(ix::tht)(this->ijk) = tmp2(this->ijk) - tht_e(this->ijk);
    } 

    const auto &ijk = this->ijk;
    auto ix_w = this->vip_ixs[ct_params_t::n_dims - 1];
    const auto &tht_abs = *this->mem->vab_coeff;

    if (!ct_params_t::impl_tht)
    {
      switch (at)
      {
        case (0):
        {
          //rhs.at(ix::tht)(ijk) += -tht_abs(ijk) * (this->state(ix::tht)(ijk) - tht_e(ijk));

          //if (!buoy_filter)
          //{
          //  rhs.at(ix_w)(ijk) += buoy_at_0(ijk);
          //}
          //else
          //{
          //  tmp1(ijk) = buoy_at_0(ijk);
          //  filter();
          //  rhs.at(ix_w)(ijk) += (tmp2)(ijk);
          //}
          break;
        }
        case (1):
        {
          this->state(ix::tht)(ijk) = (this->state(ix::tht)(ijk) +  0.5 * this->dt * tht_abs(ijk) * this->tht_e(ijk)) /
                                      (1 + 0.5 * this->dt * tht_abs(ijk));

          rhs.at(ix::tht)(ijk) += -tht_abs(ijk) * (this->state(ix::tht)(ijk) - this->tht_e(ijk));

          if (!buoy_filter)
          {
            rhs.at(ix_w)(ijk) += buoy_at_0(ijk);
          }
          else
          {
            //tmp1(ijk) = buoy_at_0(ijk);
            //filter();
            //rhs.at(ix_w)(ijk) += (tmp2)(ijk);
            
            tmp1(ijk) = this->state(ix::tht)(ijk) - this->tht_e(ijk);
            filter();
            rhs.at(ix_w)(ijk) += this->g * (
                  (tmp2)(ijk) / this->tht_b(ijk)
                + buoy_eps * (this->state(ix::qv)(ijk) - this->qv_e(ijk))
                - this->state(ix::qc)(ijk) - this->state(ix::qr)(ijk) 
                );
          }
        }
      }
    }
    else
    {
      const auto &tht = this->state(ix::tht); 
      const auto &w = this->state(ix_w);

      switch (at)
      {
        case (0):
        {
          rhs.at(ix::tht)(ijk) += -w(ijk) * this->dtht_e(ijk) - tht_abs(ijk) * tht(ijk);

          rhs.at(ix_w)(ijk) += this->g * (
                  ( tht(ijk) / this->tht_b(ijk)
                  + buoy_eps * (this->state(ix::qv)(ijk) - this->qv_e(ijk))
                  - this->state(ix::qc)(ijk) - this->state(ix::qr)(ijk) 
                  ));

          break;
        }
        case (1):
        {
          rhs.at(ix::tht)(ijk) += (-tht_abs(ijk) * tht(ijk))
                                  / (1 + 0.5 * this->dt * tht_abs(ijk));

          rhs.at(ix_w)(ijk) += this->g * (
                  ( (tht(ijk) + 0.5 * this->dt * rhs.at(ix::tht)(ijk))  / this->tht_b(ijk)
                  + buoy_eps * (this->state(ix::qv)(ijk) - this->qv_e(ijk))
                  - this->state(ix::qc)(ijk) - this->state(ix::qr)(ijk) 
                  ));
          break;
        }
      }
    }
  }

  void hook_mixed_rhs()
  {
    this->state(ix::w)(this->ijk) += 0.5 * this->dt * this->rhs.at(ix::w)(this->ijk);
  }

  void vip_rhs_impl_fnlz()
  {
    parent_t::vip_rhs_impl_fnlz();
   
    if (ct_params_t::impl_tht)
    {
      const auto &w = this->vips()[ct_params_t::n_dims - 1];
      this->state(ix::tht)(this->ijk) += - 0.5 * this->dt * w(this->ijk) * this->dtht_e(this->ijk);
      this->rhs.at(ix::tht)(this->ijk) += -w(this->ijk) * this->dtht_e(this->ijk);
    }
  }
  
  void hook_ante_step()
  {
    save_stats();

    //if (this->rank == 0) std::cout << "itime, time: " << this->timestep+1 << ' ' << this->time / 60.  << std::endl;
    //print_stats("before half");
    parent_t::hook_ante_step();

    this->state(ix::qv)(this->ijk) = max(0., this->state(ix::qv)(this->ijk));
    this->state(ix::qc)(this->ijk) = max(0., this->state(ix::qc)(this->ijk));
    this->state(ix::qr)(this->ijk) = max(0., this->state(ix::qr)(this->ijk));
    
    this->state(ix::tht)(this->ijk) -= 300.;
    this->mem->barrier();
  
    //print_stats("after half");
  }


  void hook_post_step()
  {
    this->state(ix::tht)(this->ijk) += 300.;
    this->mem->barrier();
    
    //print_stats("after adv");
    parent_t::hook_post_step();

    //if ((this->timestep % 12) == 0)
    //{
    //  auto cfl = this->courant_number(this->mem->GC);
    //  this->mem->barrier();
    //  const libmpdataxx::rng_t ir(0, 84);
    //  const libmpdataxx::rng_t jr(0, 84);
    //  const libmpdataxx::rng_t kr(0, 40);
    //  if (this->rank == 0)
    //  {
    //    auto tht_max     = max(this->state(ix::tht)(ir, jr, kr));
    //    auto tht_min     = min(this->state(ix::tht)(ir, jr, kr));
    //    auto qv_max     = max(this->state(ix::qv)(ir, jr, kr));
    //    auto qv_min     = min(this->state(ix::qv)(ir, jr, kr));
    //    auto qc_max     = max(this->state(ix::qc)(ir, jr, kr));
    //    auto qc_min     = min(this->state(ix::qc)(ir, jr, kr));
    //    auto qr_max     = max(this->state(ix::qr)(ir, jr, kr));
    //    auto qr_min     = min(this->state(ix::qr)(ir, jr, kr));
    //    auto w_max     = max(this->state(ix::w)(ir, jr, kr));
    //    auto w_min     = min(this->state(ix::w)(ir, jr, kr));
    //    std::cout << "time: " << this->time / 60.  << std::endl;
    //    std::cout << "cfl: " << cfl  << std::endl;
    //    std::cout << "tht: " << tht_min << ' ' << tht_max << std::endl;
    //    std::cout << "qv: " << qv_min << ' ' << qv_max << std::endl;
    //    std::cout << "qc: " << qc_min << ' ' << qc_max << std::endl;
    //    std::cout << "qr: " << qr_min << ' ' << qr_max << std::endl;
    //    std::cout << "w: " << w_min << ' ' << w_max << std::endl;
    //  }
    //  this->mem->barrier();
    //}

    if (ct_params_t::stress_diff == libmpdataxx::solvers::compact)
    {
      diffusion_cmpct();
    }
    else
    {
      diffusion_simple();
    }
  }

  public:

  struct rt_params_t : parent_t::rt_params_t 
  { 
    real_t g, cp, Rd, Rv, L, e0, epsa, T0, buoy_eps;
    bool buoy_filter = false;
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
    buoy_filter(p.buoy_filter),
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
