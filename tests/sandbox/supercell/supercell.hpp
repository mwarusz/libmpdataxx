/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */
#pragma once
#include <libmpdata++/solvers/boussinesq.hpp>
#include <algorithm>

template <class ct_params_t>
class supercell : public libmpdataxx::solvers::mpdata_rhs_vip_prs_sgs<ct_params_t>
{
  using parent_t = libmpdataxx::solvers::mpdata_rhs_vip_prs_sgs<ct_params_t>;

  public:
  using real_t = typename ct_params_t::real_t;

  protected:
  // member fields
  using ix = typename ct_params_t::ix;
  const real_t buoy_eps = 0.608;
  real_t g;
  bool buoy_filter;
  typename parent_t::arr_t &tht_b, &tht_e, &pk_e, &qv_e, &tmp1, &tmp2, &u_e;
  libmpdataxx::arrvec_t<typename parent_t::arr_t> &grad_aux;

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

  void vip_rhs_expl_calc()
  {
    this->state(ix::u)(this->ijk) -= u_e(this->ijk);
    parent_t::vip_rhs_expl_calc();
    this->state(ix::u)(this->ijk) += u_e(this->ijk);
  }

  void diffusion()
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
    tmp1(ijk) = tht(ijk) - tht_e(ijk);
    this->xchng_pres(tmp1, ijk);
    nabla::calc_grad_cmpct<parent_t::n_dims>(grad_aux, tmp1, ijk, this->ijkm, this->dijk);

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

  // explicit forcings 
  void update_rhs(
    libmpdataxx::arrvec_t<
      typename parent_t::arr_t
    > &rhs, 
    const real_t &dt, 
    const int &at 
  ) {
    parent_t::update_rhs(rhs, dt, at); 

    const auto &ijk = this->ijk;

    auto ix_w = this->vip_ixs[ct_params_t::n_dims - 1];

    switch (at)
    {
      case (0):
      {
        if (!buoy_filter)
        {
          rhs.at(ix_w)(ijk) += buoy_at_0(ijk);
        }
        else
        {
          tmp1(ijk) = buoy_at_0(ijk);
          filter();
          rhs.at(ix_w)(ijk) += (tmp2)(ijk);
        }
        break;
      }
      case (1):
      {
        if (!buoy_filter)
        {
          rhs.at(ix_w)(ijk) += buoy_at_1(ijk);
        }
        else
        {
          tmp1(ijk) = buoy_at_1(ijk);
          filter();
          rhs.at(ix_w)(ijk) += (tmp2)(ijk);
        }
      }
    }
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

  void hook_post_step()
  {
    parent_t::hook_post_step();
    for (int i = this->i.first(); i <= this->i.last(); ++i)
    {
      for (int j = this->j.first(); j <= this->j.last(); ++j)
      {
        auto 
        rho_c    = (*this->mem->G)(i, j, this->k).reindex(0),
  	qv_c     = this->state(ix::qv)(i, j, this->k).reindex(0),
  	qc_c     = this->state(ix::qc)(i, j, this->k).reindex(0),
  	qr_c     = this->state(ix::qr)(i, j, this->k).reindex(0),
  	tht_c    = this->state(ix::tht)(i, j, this->k).reindex(0),
  	pk_c     = pk_e(i, j, this->k).reindex(0);
        kessler(qv_c, qc_c, qr_c, tht_c, rho_c, pk_c, this->mem->grid_size[2].last() + 1);
      }
    }

    diffusion();
  }

  public:

  struct rt_params_t : parent_t::rt_params_t 
  { 
    real_t g = 9.81;
    bool buoy_filter = false;
  };

  // ctor
  supercell( 
    typename parent_t::ctor_args_t args, 
    const rt_params_t &p
  ) :
    parent_t(args, p),
    g(p.g),
    buoy_filter(p.buoy_filter),
    tht_b(args.mem->tmp[__FILE__][0][0]),
    tht_e(args.mem->tmp[__FILE__][1][0]),
    pk_e(args.mem->tmp[__FILE__][2][0]),
    qv_e(args.mem->tmp[__FILE__][3][0]),
    tmp1(args.mem->tmp[__FILE__][4][0]),
    tmp2(args.mem->tmp[__FILE__][4][1]),
    u_e(args.mem->tmp[__FILE__][5][0]),
    grad_aux(args.mem->tmp[__FILE__][6])
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
    parent_t::alloc_tmp_vctr(mem, __FILE__); // grad_aux
  }
};
