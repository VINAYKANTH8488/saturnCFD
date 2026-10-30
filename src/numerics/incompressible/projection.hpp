/**
 * @file projection.hpp
 * @brief MAC velocity layout and discrete divergence/gradient operators.
 */
#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>

#include "core/parallel.hpp"
#include "numerics/incompressible/poisson.hpp"

namespace saturn {
namespace projection {

// MAC (staggered) velocity layout on a periodic box.
//
// Face index i is the x-face on the low side of cell i, so u,v,w all have the
// same extents as the cell-centered pressure. The discrete divergence and
// gradient are exact adjoints:
//
//   div(u)_ijk = (u_{i+1} - u_i)/hx + (v_{j+1} - v_j)/hy + (w_{k+1} - w_k)/hz
//   grad(p)_i  = (p_i - p_{i-1})/hx
//
// and div(grad(p)) is the 7-point Laplacian used by the Poisson solver.

template <class ExecSpace>
using ScalarView = Kokkos::View<double***, Kokkos::LayoutRight,
                                typename ExecSpace::memory_space>;

template <class ExecSpace>
void divergence(const ScalarView<ExecSpace>& u, const ScalarView<ExecSpace>& v,
                const ScalarView<ExecSpace>& w, ScalarView<ExecSpace>& div,
                const std::array<double, 3>& h) {
  const int n0 = static_cast<int>(u.extent(2));
  const int n1 = static_cast<int>(u.extent(1));
  const int n2 = static_cast<int>(u.extent(0));
  const double hx = h[0];
  const double hy = h[1];
  const double hz = h[2];
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "projection_divergence", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int ip = (i + 1 == n0) ? 0 : i + 1;
        const int jp = (j + 1 == n1) ? 0 : j + 1;
        const int kp = (k + 1 == n2) ? 0 : k + 1;
        div(k, j, i) = (u(k, j, ip) - u(k, j, i)) / hx +
                       (v(k, jp, i) - v(k, j, i)) / hy +
                       (w(kp, j, i) - w(k, j, i)) / hz;
      });
}

template <class ExecSpace>
void gradientX(const ScalarView<ExecSpace>& p, ScalarView<ExecSpace>& gx,
               double hx) {
  const int n0 = static_cast<int>(p.extent(2));
  const int n1 = static_cast<int>(p.extent(1));
  const int n2 = static_cast<int>(p.extent(0));
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "projection_grad_x", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? n0 - 1 : i - 1;
        gx(k, j, i) = (p(k, j, i) - p(k, j, im)) / hx;
      });
}

template <class ExecSpace>
void gradientY(const ScalarView<ExecSpace>& p, ScalarView<ExecSpace>& gy,
               double hy) {
  const int n0 = static_cast<int>(p.extent(2));
  const int n1 = static_cast<int>(p.extent(1));
  const int n2 = static_cast<int>(p.extent(0));
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "projection_grad_y", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int jm = (j == 0) ? n1 - 1 : j - 1;
        gy(k, j, i) = (p(k, j, i) - p(k, jm, i)) / hy;
      });
}

template <class ExecSpace>
void gradientZ(const ScalarView<ExecSpace>& p, ScalarView<ExecSpace>& gz,
               double hz) {
  const int n0 = static_cast<int>(p.extent(2));
  const int n1 = static_cast<int>(p.extent(1));
  const int n2 = static_cast<int>(p.extent(0));
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "projection_grad_z", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int km = (k == 0) ? n2 - 1 : k - 1;
        gz(k, j, i) = (p(k, j, i) - p(km, j, i)) / hz;
      });
}

// Remove the discrete divergence from a MAC velocity:
//
//   u* = u + dt grad(phi),   laplacian(phi) = -div(u)/dt.
//
// `scratch` holds the right-hand side and is overwritten. `phi` holds the
// pressure-correction potential. The returned Poisson result describes the
// quality of the projection.
template <class ExecSpace, class PoissonSolver>
poisson::Result project(ScalarView<ExecSpace>& u, ScalarView<ExecSpace>& v,
                        ScalarView<ExecSpace>& w, ScalarView<ExecSpace>& phi,
                        ScalarView<ExecSpace>& scratch,
                        const std::array<double, 3>& h, double dt,
                        PoissonSolver& solver,
                        const poisson::Options& options) {
  divergence<ExecSpace>(u, v, w, scratch, h);
  const int n0 = static_cast<int>(u.extent(2));
  const int n1 = static_cast<int>(u.extent(1));
  const int n2 = static_cast<int>(u.extent(0));
  const double hx = h[0];
  const double hy = h[1];
  const double hz = h[2];
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "projection_rhs", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        scratch(k, j, i) = -scratch(k, j, i) / dt;
      });
  const poisson::Result result = solver.solve(phi, scratch, options);

  Kokkos::parallel_for(
      "projection_update", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? n0 - 1 : i - 1;
        const int jm = (j == 0) ? n1 - 1 : j - 1;
        const int km = (k == 0) ? n2 - 1 : k - 1;
        u(k, j, i) += dt * (phi(k, j, i) - phi(k, j, im)) / hx;
        v(k, j, i) += dt * (phi(k, j, i) - phi(k, jm, i)) / hy;
        w(k, j, i) += dt * (phi(k, j, i) - phi(km, j, i)) / hz;
      });
  return result;
}

}  // namespace projection
}  // namespace saturn
