#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>

#include "core/parallel.hpp"
#include "numerics/incompressible.hpp"
#include "numerics/projection.hpp"

namespace saturn {
namespace edac {

// Entropically Damped Artificial Compressibility (Clausen 2013) on a
// staggered MAC grid:
//
//   du/dt = -(u.grad)u + nu laplacian(u) - grad(p)
//   dp/dt = -c^2 div(u) + nu_p laplacian(p),   nu_p = c^2 tau,  c = 1/M_c
//
// No pressure Poisson solve is performed. `tau` is set to tau_factor*h/c, so
// nu_p = c*tau_factor*h. All fields are periodic in this prototype.

struct Parameters {
  double nu = 0.01;
  double c = 10.0;
  double nu_p = 0.01;
};

template <class ExecSpace>
void laplacian(const projection::ScalarView<ExecSpace>& p,
               projection::ScalarView<ExecSpace>& lap,
               const std::array<double, 3>& h) {
  const int n0 = static_cast<int>(p.extent(2));
  const int n1 = static_cast<int>(p.extent(1));
  const int n2 = static_cast<int>(p.extent(0));
  const double hx2 = h[0] * h[0];
  const double hy2 = h[1] * h[1];
  const double hz2 = h[2] * h[2];
  const auto policy =
      make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "edac_laplacian", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? n0 - 1 : i - 1;
        const int ip = (i + 1 == n0) ? 0 : i + 1;
        const int jm = (j == 0) ? n1 - 1 : j - 1;
        const int jp = (j + 1 == n1) ? 0 : j + 1;
        const int km = (k == 0) ? n2 - 1 : k - 1;
        const int kp = (k + 1 == n2) ? 0 : k + 1;
        lap(k, j, i) = (p(k, j, ip) - 2.0 * p(k, j, i) + p(k, j, im)) / hx2 +
                       (p(k, jp, i) - 2.0 * p(k, j, i) + p(k, jm, i)) / hy2 +
                       (p(kp, j, i) - 2.0 * p(k, j, i) + p(km, j, i)) / hz2;
      });
}

template <class ExecSpace>
void subtract(projection::ScalarView<ExecSpace>& a,
              const projection::ScalarView<ExecSpace>& b) {
  const int n0 = static_cast<int>(a.extent(2));
  const int n1 = static_cast<int>(a.extent(1));
  const int n2 = static_cast<int>(a.extent(0));
  const auto policy =
      make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "edac_subtract", policy,
      KOKKOS_LAMBDA(int k, int j, int i) { a(k, j, i) -= b(k, j, i); });
}

// Velocity RHS: advection + diffusion from the shared momentum operator, then
// subtract grad(p) evaluated at the faces.
template <class ExecSpace>
void velocity_rhs(const projection::ScalarView<ExecSpace>& u,
                  const projection::ScalarView<ExecSpace>& v,
                  const projection::ScalarView<ExecSpace>& w,
                  const projection::ScalarView<ExecSpace>& p,
                  projection::ScalarView<ExecSpace>& du,
                  projection::ScalarView<ExecSpace>& dv,
                  projection::ScalarView<ExecSpace>& dw,
                  projection::ScalarView<ExecSpace>& gx,
                  projection::ScalarView<ExecSpace>& gy,
                  projection::ScalarView<ExecSpace>& gz,
                  const std::array<double, 3>& h, double nu) {
  incompressible::momentum_rhs<ExecSpace>(u, v, w, du, dv, dw, h, nu);
  projection::gradient_x<ExecSpace>(p, gx, h[0]);
  projection::gradient_y<ExecSpace>(p, gy, h[1]);
  projection::gradient_z<ExecSpace>(p, gz, h[2]);
  subtract<ExecSpace>(du, gx);
  subtract<ExecSpace>(dv, gy);
  subtract<ExecSpace>(dw, gz);
}

template <class ExecSpace>
void pressure_rhs(const projection::ScalarView<ExecSpace>& p,
                  const projection::ScalarView<ExecSpace>& u,
                  const projection::ScalarView<ExecSpace>& v,
                  const projection::ScalarView<ExecSpace>& w,
                  projection::ScalarView<ExecSpace>& dp,
                  projection::ScalarView<ExecSpace>& div,
                  projection::ScalarView<ExecSpace>& lap,
                  const std::array<double, 3>& h, const Parameters& params) {
  projection::divergence<ExecSpace>(u, v, w, div, h);
  laplacian<ExecSpace>(p, lap, h);
  const int n0 = static_cast<int>(p.extent(2));
  const int n1 = static_cast<int>(p.extent(1));
  const int n2 = static_cast<int>(p.extent(0));
  const double c2 = params.c * params.c;
  const auto policy =
      make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "edac_pressure_rhs", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        dp(k, j, i) = -c2 * div(k, j, i) + params.nu_p * lap(k, j, i);
      });
}

// SSP-RK3 for (u,v,w,p).
template <class ExecSpace>
void advance_ssprk3(projection::ScalarView<ExecSpace>& u,
                    projection::ScalarView<ExecSpace>& v,
                    projection::ScalarView<ExecSpace>& w,
                    projection::ScalarView<ExecSpace>& p,
                    projection::ScalarView<ExecSpace>& u1,
                    projection::ScalarView<ExecSpace>& v1,
                    projection::ScalarView<ExecSpace>& w1,
                    projection::ScalarView<ExecSpace>& p1,
                    projection::ScalarView<ExecSpace>& u2,
                    projection::ScalarView<ExecSpace>& v2,
                    projection::ScalarView<ExecSpace>& w2,
                    projection::ScalarView<ExecSpace>& p2,
                    projection::ScalarView<ExecSpace>& ku,
                    projection::ScalarView<ExecSpace>& kv,
                    projection::ScalarView<ExecSpace>& kw,
                    projection::ScalarView<ExecSpace>& kp,
                    projection::ScalarView<ExecSpace>& gx,
                    projection::ScalarView<ExecSpace>& gy,
                    projection::ScalarView<ExecSpace>& gz,
                    projection::ScalarView<ExecSpace>& div,
                    projection::ScalarView<ExecSpace>& lap,
                    const std::array<double, 3>& h, const Parameters& params,
                    double dt) {
  auto rhs = [&](auto& su, auto& sv, auto& sw, auto& sp, auto& du, auto& dv,
                 auto& dw, auto& dp) {
    velocity_rhs<ExecSpace>(su, sv, sw, sp, du, dv, dw, gx, gy, gz, h,
                            params.nu);
    pressure_rhs<ExecSpace>(sp, su, sv, sw, dp, div, lap, h, params);
  };

  rhs(u, v, w, p, ku, kv, kw, kp);
  incompressible::combine<ExecSpace>(u, 1.0, ku, dt, ku, 0.0, u1);
  incompressible::combine<ExecSpace>(v, 1.0, kv, dt, ku, 0.0, v1);
  incompressible::combine<ExecSpace>(w, 1.0, kw, dt, ku, 0.0, w1);
  incompressible::combine<ExecSpace>(p, 1.0, kp, dt, ku, 0.0, p1);

  rhs(u1, v1, w1, p1, ku, kv, kw, kp);
  incompressible::combine<ExecSpace>(u, 0.75, u1, 0.25, ku, 0.25 * dt, u2);
  incompressible::combine<ExecSpace>(v, 0.75, v1, 0.25, kv, 0.25 * dt, v2);
  incompressible::combine<ExecSpace>(w, 0.75, w1, 0.25, kw, 0.25 * dt, w2);
  incompressible::combine<ExecSpace>(p, 0.75, p1, 0.25, kp, 0.25 * dt, p2);

  rhs(u2, v2, w2, p2, ku, kv, kw, kp);
  incompressible::combine<ExecSpace>(u, 1.0 / 3.0, u2, 2.0 / 3.0, ku,
                                     (2.0 / 3.0) * dt, u);
  incompressible::combine<ExecSpace>(v, 1.0 / 3.0, v2, 2.0 / 3.0, kv,
                                     (2.0 / 3.0) * dt, v);
  incompressible::combine<ExecSpace>(w, 1.0 / 3.0, w2, 2.0 / 3.0, kw,
                                     (2.0 / 3.0) * dt, w);
  incompressible::combine<ExecSpace>(p, 1.0 / 3.0, p2, 2.0 / 3.0, kp,
                                     (2.0 / 3.0) * dt, p);
}

}  // namespace edac
}  // namespace saturn
