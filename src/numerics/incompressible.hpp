#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>

#include "core/parallel.hpp"
#include "numerics/projection.hpp"

namespace saturn {
namespace incompressible {

// Explicit momentum right-hand side for the MAC (staggered) formulation:
//
//   du/dt = -(u.grad)u + nu laplacian(u)
//
// with second-order central differences. Pressure is applied separately by
// `projection::project`. The advecting transverse velocities are interpolated
// to the face from the four surrounding faces.

template <class ExecSpace>
void momentumRhs(const projection::ScalarView<ExecSpace>& u,
                 const projection::ScalarView<ExecSpace>& v,
                 const projection::ScalarView<ExecSpace>& w,
                 projection::ScalarView<ExecSpace>& du,
                 projection::ScalarView<ExecSpace>& dv,
                 projection::ScalarView<ExecSpace>& dw,
                 const std::array<double, 3>& h, double nu) {
  const int n0 = static_cast<int>(u.extent(2));
  const int n1 = static_cast<int>(u.extent(1));
  const int n2 = static_cast<int>(u.extent(0));
  const double hx = h[0];
  const double hy = h[1];
  const double hz = h[2];
  const double hx2 = hx * hx;
  const double hy2 = hy * hy;
  const double hz2 = hz * hz;
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "incompressible_rhs", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? n0 - 1 : i - 1;
        const int ip = (i + 1 == n0) ? 0 : i + 1;
        const int jm = (j == 0) ? n1 - 1 : j - 1;
        const int jp = (j + 1 == n1) ? 0 : j + 1;
        const int km = (k == 0) ? n2 - 1 : k - 1;
        const int kp = (k + 1 == n2) ? 0 : k + 1;

        // u-momentum at the x-face (i,j,k).
        const double dudx = (u(k, j, ip) - u(k, j, im)) / (2.0 * hx);
        const double dudy = (u(k, jp, i) - u(k, jm, i)) / (2.0 * hy);
        const double dudz = (u(kp, j, i) - u(km, j, i)) / (2.0 * hz);
        const double v_at_u =
            0.25 * (v(k, j, im) + v(k, j, i) + v(k, jp, im) + v(k, jp, i));
        const double w_at_u =
            0.25 * (w(k, j, im) + w(k, j, i) + w(kp, j, im) + w(kp, j, i));
        const double lap_u =
            (u(k, j, ip) - 2.0 * u(k, j, i) + u(k, j, im)) / hx2 +
            (u(k, jp, i) - 2.0 * u(k, j, i) + u(k, jm, i)) / hy2 +
            (u(kp, j, i) - 2.0 * u(k, j, i) + u(km, j, i)) / hz2;
        du(k, j, i) =
            -(u(k, j, i) * dudx + v_at_u * dudy + w_at_u * dudz) + nu * lap_u;

        // v-momentum at the y-face (i,j,k).
        const double dvdx = (v(k, j, ip) - v(k, j, im)) / (2.0 * hx);
        const double dvdy = (v(k, jp, i) - v(k, jm, i)) / (2.0 * hy);
        const double dvdz = (v(kp, j, i) - v(km, j, i)) / (2.0 * hz);
        const double u_at_v =
            0.25 * (u(k, jm, i) + u(k, j, i) + u(k, jm, ip) + u(k, j, ip));
        const double w_at_v =
            0.25 * (w(k, jm, i) + w(k, j, i) + w(kp, jm, i) + w(kp, j, i));
        const double lap_v =
            (v(k, j, ip) - 2.0 * v(k, j, i) + v(k, j, im)) / hx2 +
            (v(k, jp, i) - 2.0 * v(k, j, i) + v(k, jm, i)) / hy2 +
            (v(kp, j, i) - 2.0 * v(k, j, i) + v(km, j, i)) / hz2;
        dv(k, j, i) =
            -(u_at_v * dvdx + v(k, j, i) * dvdy + w_at_v * dvdz) + nu * lap_v;

        // w-momentum at the z-face (i,j,k).
        const double dwdx = (w(k, j, ip) - w(k, j, im)) / (2.0 * hx);
        const double dwdy = (w(k, jp, i) - w(k, jm, i)) / (2.0 * hy);
        const double dwdz = (w(kp, j, i) - w(km, j, i)) / (2.0 * hz);
        const double u_at_w =
            0.25 * (u(km, j, i) + u(k, j, i) + u(km, j, ip) + u(k, j, ip));
        const double v_at_w =
            0.25 * (v(km, j, i) + v(k, j, i) + v(km, jp, i) + v(k, jp, i));
        const double lap_w =
            (w(k, j, ip) - 2.0 * w(k, j, i) + w(k, j, im)) / hx2 +
            (w(k, jp, i) - 2.0 * w(k, j, i) + w(k, jm, i)) / hy2 +
            (w(kp, j, i) - 2.0 * w(k, j, i) + w(km, j, i)) / hz2;
        dw(k, j, i) =
            -(u_at_w * dwdx + v_at_w * dwdy + w(k, j, i) * dwdz) + nu * lap_w;
      });
}

template <class ExecSpace>
void combine(const projection::ScalarView<ExecSpace>& a, double alpha,
             const projection::ScalarView<ExecSpace>& b, double beta,
             const projection::ScalarView<ExecSpace>& c, double gamma,
             projection::ScalarView<ExecSpace>& out) {
  const int n0 = static_cast<int>(out.extent(2));
  const int n1 = static_cast<int>(out.extent(1));
  const int n2 = static_cast<int>(out.extent(0));
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "incompressible_combine", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        out(k, j, i) =
            alpha * a(k, j, i) + beta * b(k, j, i) + gamma * c(k, j, i);
      });
}

// Fourth-order periodic filter along one dimension:
//
//   out = in - alpha/16 (in_{d-2} - 4 in_{d-1} + 6 in - 4 in_{d+1} + in_{d+2})
//
// The symbol is 1 - alpha sin^4(k h / 2), so grid-scale modes are damped while
// well-resolved modes are barely touched. This is a numerical stabilization
// for under-resolved transitional flows, not part of the physical model.
template <class ExecSpace>
void filterDimension(const projection::ScalarView<ExecSpace>& in,
                     projection::ScalarView<ExecSpace>& out, double alpha,
                     int dimension) {
  const int n0 = static_cast<int>(in.extent(2));
  const int n1 = static_cast<int>(in.extent(1));
  const int n2 = static_cast<int>(in.extent(0));
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "incompressible_filter", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        double value = 0.0;
        if (dimension == 0) {
          const int im2 = (i - 2 + n0) % n0;
          const int im1 = (i - 1 + n0) % n0;
          const int ip1 = (i + 1) % n0;
          const int ip2 = (i + 2) % n0;
          value = in(k, j, im2) - 4.0 * in(k, j, im1) + 6.0 * in(k, j, i) -
                  4.0 * in(k, j, ip1) + in(k, j, ip2);
        } else if (dimension == 1) {
          const int jm2 = (j - 2 + n1) % n1;
          const int jm1 = (j - 1 + n1) % n1;
          const int jp1 = (j + 1) % n1;
          const int jp2 = (j + 2) % n1;
          value = in(k, jm2, i) - 4.0 * in(k, jm1, i) + 6.0 * in(k, j, i) -
                  4.0 * in(k, jp1, i) + in(k, jp2, i);
        } else {
          const int km2 = (k - 2 + n2) % n2;
          const int km1 = (k - 1 + n2) % n2;
          const int kp1 = (k + 1) % n2;
          const int kp2 = (k + 2) % n2;
          value = in(km2, j, i) - 4.0 * in(km1, j, i) + 6.0 * in(k, j, i) -
                  4.0 * in(kp1, j, i) + in(kp2, j, i);
        }
        out(k, j, i) = in(k, j, i) - (alpha / 16.0) * value;
      });
}

template <class ExecSpace>
void filter4(projection::ScalarView<ExecSpace>& u,
             projection::ScalarView<ExecSpace>& v,
             projection::ScalarView<ExecSpace>& w,
             projection::ScalarView<ExecSpace>& tu,
             projection::ScalarView<ExecSpace>& tv,
             projection::ScalarView<ExecSpace>& tw, double alpha) {
  if (alpha <= 0.0) {
    return;
  }
  filterDimension<ExecSpace>(u, tu, alpha, 0);
  filterDimension<ExecSpace>(tu, u, alpha, 1);
  filterDimension<ExecSpace>(u, tu, alpha, 2);
  Kokkos::deep_copy(u, tu);

  filterDimension<ExecSpace>(v, tv, alpha, 0);
  filterDimension<ExecSpace>(tv, v, alpha, 1);
  filterDimension<ExecSpace>(v, tv, alpha, 2);
  Kokkos::deep_copy(v, tv);

  filterDimension<ExecSpace>(w, tw, alpha, 0);
  filterDimension<ExecSpace>(tw, w, alpha, 1);
  filterDimension<ExecSpace>(w, tw, alpha, 2);
  Kokkos::deep_copy(w, tw);
}

// SSP-RK3 for the three face-velocity components, with a projection applied
// after every stage. The projection callback takes the three velocity views
// and must make them discretely divergence-free.
template <class ExecSpace, class Project>
void advanceSsprk3(projection::ScalarView<ExecSpace>& u,
                   projection::ScalarView<ExecSpace>& v,
                   projection::ScalarView<ExecSpace>& w,
                   projection::ScalarView<ExecSpace>& u1,
                   projection::ScalarView<ExecSpace>& v1,
                   projection::ScalarView<ExecSpace>& w1,
                   projection::ScalarView<ExecSpace>& u2,
                   projection::ScalarView<ExecSpace>& v2,
                   projection::ScalarView<ExecSpace>& w2,
                   projection::ScalarView<ExecSpace>& ku,
                   projection::ScalarView<ExecSpace>& kv,
                   projection::ScalarView<ExecSpace>& kw,
                   const std::array<double, 3>& h, double nu, double dt,
                   Project& project) {
  momentumRhs<ExecSpace>(u, v, w, ku, kv, kw, h, nu);
  combine<ExecSpace>(u, 1.0, ku, dt, ku, 0.0, u1);
  combine<ExecSpace>(v, 1.0, kv, dt, ku, 0.0, v1);
  combine<ExecSpace>(w, 1.0, kw, dt, ku, 0.0, w1);
  project(u1, v1, w1);

  momentumRhs<ExecSpace>(u1, v1, w1, ku, kv, kw, h, nu);
  combine<ExecSpace>(u, 0.75, u1, 0.25, ku, 0.25 * dt, u2);
  combine<ExecSpace>(v, 0.75, v1, 0.25, kv, 0.25 * dt, v2);
  combine<ExecSpace>(w, 0.75, w1, 0.25, kw, 0.25 * dt, w2);
  project(u2, v2, w2);

  momentumRhs<ExecSpace>(u2, v2, w2, ku, kv, kw, h, nu);
  combine<ExecSpace>(u, 1.0 / 3.0, u2, 2.0 / 3.0, ku, (2.0 / 3.0) * dt, u);
  combine<ExecSpace>(v, 1.0 / 3.0, v2, 2.0 / 3.0, kv, (2.0 / 3.0) * dt, v);
  combine<ExecSpace>(w, 1.0 / 3.0, w2, 2.0 / 3.0, kw, (2.0 / 3.0) * dt, w);
  project(u, v, w);
}

}  // namespace incompressible
}  // namespace saturn
