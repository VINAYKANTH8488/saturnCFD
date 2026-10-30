#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>

#include "core/parallel.hpp"
#include "numerics/incompressible/incompressible.hpp"
#include "numerics/incompressible/poisson.hpp"
#include "numerics/incompressible/projection.hpp"

namespace saturn {
namespace channel {

// Wall-bounded channel on a staggered MAC grid.
//
//   x, z: periodic
//   y:    no-slip walls at y=0 and y=Ly
//
// Velocity layout (same extents as the cell-centered pressure):
//
//   u(i,j,k): x-face, y and z at cell centers
//   v(i,j,k): y-face; v(j=0) is the bottom wall, v(j=n1) (implicit) the top
//   w(i,j,k): z-face, x and y at cell centers
//
// The top wall can move in x with speed `u_top` (Couette flow); the bottom
// wall is stationary. A constant body force `gx` drives plane Poiseuille flow.
// Pressure uses homogeneous Neumann conditions at both walls.

struct Geometry {
  int n0 = 1;
  int n1 = 1;
  int n2 = 1;
  double hx = 1.0;
  double hy = 1.0;
  double hz = 1.0;
  double ly = 1.0;
  double u_top = 0.0;
  double gx = 0.0;
  double nu = 1.0;
};

template <class ExecSpace>
void divergence(const projection::ScalarView<ExecSpace>& u,
                const projection::ScalarView<ExecSpace>& v,
                const projection::ScalarView<ExecSpace>& w,
                projection::ScalarView<ExecSpace>& div, const Geometry& g) {
  const int n0 = g.n0;
  const int n1 = g.n1;
  const int n2 = g.n2;
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "channel_divergence", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int ip = (i + 1 == n0) ? 0 : i + 1;
        const int kp = (k + 1 == n2) ? 0 : k + 1;
        const double vp = (j + 1 == n1) ? 0.0 : v(k, j + 1, i);
        div(k, j, i) = (u(k, j, ip) - u(k, j, i)) / g.hx +
                       (vp - v(k, j, i)) / g.hy +
                       (w(kp, j, i) - w(k, j, i)) / g.hz;
      });
}

template <class ExecSpace>
void momentumRhs(const projection::ScalarView<ExecSpace>& u,
                 const projection::ScalarView<ExecSpace>& v,
                 const projection::ScalarView<ExecSpace>& w,
                 projection::ScalarView<ExecSpace>& du,
                 projection::ScalarView<ExecSpace>& dv,
                 projection::ScalarView<ExecSpace>& dw, const Geometry& g,
                 double nu) {
  const int n0 = g.n0;
  const int n1 = g.n1;
  const int n2 = g.n2;
  const double hx = g.hx;
  const double hy = g.hy;
  const double hz = g.hz;
  const double hx2 = hx * hx;
  const double hy2 = hy * hy;
  const double hz2 = hz * hz;
  const double u_top = g.u_top;
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "channel_rhs", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? n0 - 1 : i - 1;
        const int ip = (i + 1 == n0) ? 0 : i + 1;
        const int jm = (j == 0) ? -1 : j - 1;
        const int jp = (j + 1 == n1) ? -1 : j + 1;
        const int km = (k == 0) ? n2 - 1 : k - 1;
        const int kp = (k + 1 == n2) ? 0 : k + 1;

        const double u_here = u(k, j, i);
        const double u_jm = (j == 0) ? -u_here : u(k, jm, i);
        const double u_jp = (j + 1 == n1) ? 2.0 * u_top - u_here : u(k, jp, i);
        const double w_here = w(k, j, i);
        const double w_jm = (j == 0) ? -w_here : w(k, jm, i);
        const double w_jp = (j + 1 == n1) ? -w_here : w(k, jp, i);
        const double v_jm = (j == 0) ? 0.0 : v(k, j - 1, i);
        const double v_jp = (j + 1 == n1) ? 0.0 : v(k, j + 1, i);
        const double v_jp_im = (j + 1 == n1) ? 0.0 : v(k, j + 1, im);
        const double v_jp_km = (j + 1 == n1) ? 0.0 : v(km, j + 1, i);

        // u-momentum.
        const double dudx = (u(k, j, ip) - u(k, j, im)) / (2.0 * hx);
        const double dudy = (u_jp - u_jm) / (2.0 * hy);
        const double dudz = (u(kp, j, i) - u(km, j, i)) / (2.0 * hz);
        const double v_at_u =
            0.25 * (v(k, j, im) + v(k, j, i) + v_jp_im + v_jp);
        const double w_at_u =
            0.25 * (w(k, j, im) + w(k, j, i) + w(kp, j, im) + w(kp, j, i));
        const double lap_u = (u(k, j, ip) - 2.0 * u_here + u(k, j, im)) / hx2 +
                             (u_jp - 2.0 * u_here + u_jm) / hy2 +
                             (u(kp, j, i) - 2.0 * u_here + u(km, j, i)) / hz2;
        du(k, j, i) = -(u_here * dudx + v_at_u * dudy + w_at_u * dudz) +
                      nu * lap_u + g.gx;

        // v-momentum at interior y-faces; the wall faces stay zero.
        if (j == 0) {
          dv(k, j, i) = 0.0;
        } else {
          const double dvdx = (v(k, j, ip) - v(k, j, im)) / (2.0 * hx);
          const double dvdy = (v_jp - v_jm) / (2.0 * hy);
          const double dvdz = (v(kp, j, i) - v(km, j, i)) / (2.0 * hz);
          const double u_at_v =
              0.25 * (u(k, jm, i) + u_here + u(k, jm, ip) + u(k, j, ip));
          const double w_at_v =
              0.25 * (w(k, jm, i) + w_here + w(kp, jm, i) + w(kp, j, i));
          const double lap_v =
              (v(k, j, ip) - 2.0 * v(k, j, i) + v(k, j, im)) / hx2 +
              (v_jp - 2.0 * v(k, j, i) + v_jm) / hy2 +
              (v(kp, j, i) - 2.0 * v(k, j, i) + v(km, j, i)) / hz2;
          dv(k, j, i) =
              -(u_at_v * dvdx + v(k, j, i) * dvdy + w_at_v * dvdz) + nu * lap_v;
        }

        // w-momentum.
        const double dwdx = (w(k, j, ip) - w(k, j, im)) / (2.0 * hx);
        const double dwdy = (w_jp - w_jm) / (2.0 * hy);
        const double dwdz = (w(kp, j, i) - w(km, j, i)) / (2.0 * hz);
        const double u_at_w =
            0.25 * (u(km, j, i) + u_here + u(km, j, ip) + u(k, j, ip));
        const double v_at_w =
            0.25 * (v(km, j, i) + v(k, j, i) + v_jp_km + v_jp);
        const double lap_w = (w(k, j, ip) - 2.0 * w_here + w(k, j, im)) / hx2 +
                             (w_jp - 2.0 * w_here + w_jm) / hy2 +
                             (w(kp, j, i) - 2.0 * w_here + w(km, j, i)) / hz2;
        dw(k, j, i) =
            -(u_at_w * dwdx + v_at_w * dwdy + w_here * dwdz) + nu * lap_w;
      });
}

// Pressure projection with Neumann walls in y. `phi` and `scratch` are
// workspace; `solver` must use periodic x/z and Neumann y.
template <class ExecSpace, class PoissonSolver>
poisson::Result project(projection::ScalarView<ExecSpace>& u,
                        projection::ScalarView<ExecSpace>& v,
                        projection::ScalarView<ExecSpace>& w,
                        projection::ScalarView<ExecSpace>& phi,
                        projection::ScalarView<ExecSpace>& scratch,
                        const Geometry& g, PoissonSolver& solver,
                        const poisson::Options& options) {
  divergence<ExecSpace>(u, v, w, scratch, g);
  const int n0 = g.n0;
  const int n1 = g.n1;
  const int n2 = g.n2;
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  const double dt = 1.0;
  Kokkos::parallel_for(
      "channel_projection_rhs", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        scratch(k, j, i) = -scratch(k, j, i) / dt;
      });
  const poisson::Result result = solver.solve(phi, scratch, options);

  Kokkos::parallel_for(
      "channel_projection_update", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? n0 - 1 : i - 1;
        const int km = (k == 0) ? n2 - 1 : k - 1;
        u(k, j, i) += dt * (phi(k, j, i) - phi(k, j, im)) / g.hx;
        if (j > 0) {
          v(k, j, i) += dt * (phi(k, j, i) - phi(k, j - 1, i)) / g.hy;
        } else {
          v(k, j, i) = 0.0;
        }
        w(k, j, i) += dt * (phi(k, j, i) - phi(km, j, i)) / g.hz;
      });
  return result;
}

// SSP-RK3 with a projection after every stage. The callback takes the three
// velocity views and must make them divergence-free.
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
                   projection::ScalarView<ExecSpace>& kw, const Geometry& g,
                   double nu, double dt, Project& project) {
  momentumRhs<ExecSpace>(u, v, w, ku, kv, kw, g, nu);
  incompressible::combine<ExecSpace>(u, 1.0, ku, dt, ku, 0.0, u1);
  incompressible::combine<ExecSpace>(v, 1.0, kv, dt, ku, 0.0, v1);
  incompressible::combine<ExecSpace>(w, 1.0, kw, dt, ku, 0.0, w1);
  project(u1, v1, w1);

  momentumRhs<ExecSpace>(u1, v1, w1, ku, kv, kw, g, nu);
  incompressible::combine<ExecSpace>(u, 0.75, u1, 0.25, ku, 0.25 * dt, u2);
  incompressible::combine<ExecSpace>(v, 0.75, v1, 0.25, kv, 0.25 * dt, v2);
  incompressible::combine<ExecSpace>(w, 0.75, w1, 0.25, kw, 0.25 * dt, w2);
  project(u2, v2, w2);

  momentumRhs<ExecSpace>(u2, v2, w2, ku, kv, kw, g, nu);
  incompressible::combine<ExecSpace>(u, 1.0 / 3.0, u2, 2.0 / 3.0, ku,
                                     (2.0 / 3.0) * dt, u);
  incompressible::combine<ExecSpace>(v, 1.0 / 3.0, v2, 2.0 / 3.0, kv,
                                     (2.0 / 3.0) * dt, v);
  incompressible::combine<ExecSpace>(w, 1.0 / 3.0, w2, 2.0 / 3.0, kw,
                                     (2.0 / 3.0) * dt, w);
  project(u, v, w);
}

// Wall-aware fourth-order filter for the channel (periodic x/z, walls in y).
// `component`: 0 = u, 1 = v, 2 = w. Temps must be distinct from the inputs.
template <class ExecSpace>
void filterDimension(const projection::ScalarView<ExecSpace>& in,
                     projection::ScalarView<ExecSpace>& out, double alpha,
                     int dimension, int component, const Geometry& g) {
  const int n0 = g.n0;
  const int n1 = g.n1;
  const int n2 = g.n2;
  const double u_top = (component == 0) ? g.u_top : 0.0;
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "channel_filter", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        auto at = [&](int jj) {
          if (jj >= 0 && jj < n1) {
            return in(k, jj, i);
          }
          if (jj < 0) {
            if (component == 1) {
              return -in(k, -jj, i);
            }
            return -in(k, -jj - 1, i);
          }
          if (component == 1) {
            if (jj == n1) {
              return 0.0;
            }
            return -in(k, 2 * n1 - jj, i);
          }
          const int mirror = 2 * n1 - jj - 1;
          if (component == 0) {
            return 2.0 * u_top - in(k, mirror, i);
          }
          return -in(k, mirror, i);
        };
        double value = 0.0;
        if (dimension == 0) {
          const int im2 = (i - 2 + n0) % n0;
          const int im1 = (i - 1 + n0) % n0;
          const int ip1 = (i + 1) % n0;
          const int ip2 = (i + 2) % n0;
          value = in(k, j, im2) - 4.0 * in(k, j, im1) + 6.0 * in(k, j, i) -
                  4.0 * in(k, j, ip1) + in(k, j, ip2);
        } else if (dimension == 1) {
          value = at(j - 2) - 4.0 * at(j - 1) + 6.0 * in(k, j, i) -
                  4.0 * at(j + 1) + at(j + 2);
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
void filter4Channel(projection::ScalarView<ExecSpace>& u,
                    projection::ScalarView<ExecSpace>& v,
                    projection::ScalarView<ExecSpace>& w,
                    projection::ScalarView<ExecSpace>& tu,
                    projection::ScalarView<ExecSpace>& tv,
                    projection::ScalarView<ExecSpace>& tw, const Geometry& g,
                    double alpha) {
  if (alpha <= 0.0) {
    return;
  }
  filterDimension<ExecSpace>(u, tu, alpha, 0, 0, g);
  filterDimension<ExecSpace>(tu, u, alpha, 1, 0, g);
  filterDimension<ExecSpace>(u, tu, alpha, 2, 0, g);
  Kokkos::deep_copy(u, tu);

  filterDimension<ExecSpace>(v, tv, alpha, 0, 1, g);
  filterDimension<ExecSpace>(tv, v, alpha, 1, 1, g);
  filterDimension<ExecSpace>(v, tv, alpha, 2, 1, g);
  Kokkos::deep_copy(v, tv);

  filterDimension<ExecSpace>(w, tw, alpha, 0, 2, g);
  filterDimension<ExecSpace>(tw, w, alpha, 1, 2, g);
  filterDimension<ExecSpace>(w, tw, alpha, 2, 2, g);
  Kokkos::deep_copy(w, tw);

  const int n0 = g.n0;
  const int n1 = g.n1;
  const int n2 = g.n2;
  const auto policy =
      makeMdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "channel_filter_walls", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        (void)k;
        if (i == 0) {
          u(k, j, i) = 0.0;
        }
        if (j == 0) {
          v(k, j, i) = 0.0;
        }
      });
}

}  // namespace channel
}  // namespace saturn
