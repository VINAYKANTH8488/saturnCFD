#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>

#include "core/parallel.hpp"
#include "numerics/incompressible.hpp"
#include "numerics/poisson.hpp"
#include "numerics/projection.hpp"
#include "numerics/reconstruction.hpp"

namespace saturn {
namespace cavity {

// Lid-driven cavity on a staggered MAC grid.
//
//   x, y: no-slip walls; the top wall moves in x with speed `u_lid`
//   z:    periodic
//
// Layout (same extents as the cell-centered pressure):
//
//   u(i,j,k): x-face; u(0)=0 at the left wall, u(n0)=0 (implicit) at the right
//   v(i,j,k): y-face; v(0)=0 at the bottom, v(n1)=0 (implicit) at the top
//   w(i,j,k): z-face; tangential no-slip ghosts in x and y
//
// Pressure uses homogeneous Neumann conditions on the x/y walls and periodic
// conditions in z.

struct Geometry {
  int n0 = 1;
  int n1 = 1;
  int n2 = 1;
  double hx = 1.0;
  double hy = 1.0;
  double hz = 1.0;
  double u_lid = 1.0;
  // 0 = central, 1 = first-order upwind, 2 = upwind-biased MUSCL,
  // 3 = blended central/upwind, 4 = WENO5.
  int adv = 0;
  double adv_blend = 0.25;
  bool z_wall = false;
};

KOKKOS_INLINE_FUNCTION
double minmod(double a, double b) {
  if (a * b <= 0.0) {
    return 0.0;
  }
  return Kokkos::abs(a) < Kokkos::abs(b) ? a : b;
}

// WENO5 derivative from six samples q[i-3..i+2]: reconstruct the two
// midpoints with weno5z and difference them.
KOKKOS_INLINE_FUNCTION
double weno5_derivative(double qm3, double qm2, double qm1, double q0,
                        double qp1, double qp2, double qp3, double vel,
                        double h) {
  if (vel >= 0.0) {
    const double left = recon::weno5z(qm3, qm2, qm1, q0, qp1, 0.0);
    const double right = recon::weno5z(qm2, qm1, q0, qp1, qp2, 0.0);
    return (right - left) / h;
  }
  // Mirror the stencil for negative advection velocity.
  const double left = recon::weno5z(qp3, qp2, qp1, q0, qm1, 0.0);
  const double right = recon::weno5z(qp2, qp1, q0, qm1, qm2, 0.0);
  return (left - right) / h;
}

// Second-order upwind derivative with a minmod limiter.
KOKKOS_INLINE_FUNCTION
double limited_upwind(double q0, double qm1, double qm2, double qp1, double qp2,
                      double vel, double h) {
  if (vel > 0.0) {
    const double d1 = (q0 - qm1) / h;
    const double d2 = (3.0 * q0 - 4.0 * qm1 + qm2) / (2.0 * h);
    return minmod(d1, d2);
  }
  const double d1 = (qp1 - q0) / h;
  const double d2 = (-3.0 * q0 + 4.0 * qp1 - qp2) / (2.0 * h);
  return minmod(d1, d2);
}

template <class ExecSpace>
void divergence(const projection::ScalarView<ExecSpace>& u,
                const projection::ScalarView<ExecSpace>& v,
                const projection::ScalarView<ExecSpace>& w,
                projection::ScalarView<ExecSpace>& div, const Geometry& g) {
  const int n0 = g.n0;
  const int n1 = g.n1;
  const int n2 = g.n2;
  const auto policy =
      make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "cavity_divergence", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int ip = (i + 1 == n0) ? -1 : i + 1;
        const int jp = (j + 1 == n1) ? -1 : j + 1;
        const int kp = (k + 1 == n2) ? (g.z_wall ? -1 : 0) : k + 1;
        const double up = (ip < 0) ? 0.0 : u(k, j, ip);
        const double vp = (jp < 0) ? 0.0 : v(k, jp, i);
        const double wp = (kp < 0) ? 0.0 : w(kp, j, i);
        div(k, j, i) = (up - u(k, j, i)) / g.hx + (vp - v(k, j, i)) / g.hy +
                       (wp - w(k, j, i)) / g.hz;
      });
}

template <class ExecSpace>
void momentum_rhs(const projection::ScalarView<ExecSpace>& u,
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
  const double u_lid = g.u_lid;
  const int adv = g.adv;
  const double adv_blend = g.adv_blend;
  const auto policy =
      make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "cavity_rhs", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? -1 : i - 1;
        const int ip = (i + 1 == n0) ? -1 : i + 1;
        const int jm = (j == 0) ? -1 : j - 1;
        const int jp = (j + 1 == n1) ? -1 : j + 1;
        const int km = (k == 0) ? n2 - 1 : k - 1;
        const int kp = (k + 1 == n2) ? 0 : k + 1;
        const double u_here = u(k, j, i);
        const double v_here = v(k, j, i);
        const double w_here = w(k, j, i);

        // ----- u-momentum (interior x-faces only) -----
        if (i == 0) {
          du(k, j, i) = 0.0;
        } else {
          const double u_im = (im < 0) ? 0.0 : u(k, j, im);
          const double u_ip = (ip < 0) ? 0.0 : u(k, j, ip);
          const double u_jm = (jm < 0) ? -u_here : u(k, jm, i);
          const double u_jp = (jp < 0) ? 2.0 * u_lid - u_here : u(k, jp, i);
          const double u_km = (k == 0) ? (g.z_wall ? -u_here : u(n2 - 1, j, i))
                                       : u(k - 1, j, i);
          const double u_kp = (k + 1 == n2) ? (g.z_wall ? -u_here : u(0, j, i))
                                            : u(k + 1, j, i);
          double dudx = (u_ip - u_im) / (2.0 * hx);
          double dudy = (u_jp - u_jm) / (2.0 * hy);
          double dudz = (u_kp - u_km) / (2.0 * hz);
          const double v_jm = (jm < 0) ? 0.0 : v(k, jm, i);
          const double v_jp = (jp < 0) ? 0.0 : v(k, jp, i);
          const double v_jm_im = (jm < 0) ? 0.0 : v(k, jm, im);
          const double v_jp_im = (jp < 0) ? 0.0 : v(k, jp, im);
          const double v_at_u = 0.25 * (v_jm_im + v_jm + v_jp_im + v_jp);
          const double w_at_u =
              0.25 * (w(k, j, im) + w(k, j, i) + w(kp, j, im) + w(kp, j, i));
          if (adv == 1) {
            dudx = u_here > 0.0 ? (u_here - u_im) / hx : (u_ip - u_here) / hx;
            dudy = v_at_u > 0.0 ? (u_here - u_jm) / hy : (u_jp - u_here) / hy;
            dudz = w_at_u > 0.0 ? (u_here - u_km) / hz : (u_kp - u_here) / hz;
          } else if (adv == 2) {
            dudx = minmod(u_here - u_im, u_ip - u_here) / hx;
            dudy = minmod(u_here - u_jm, u_jp - u_here) / hy;
            dudz = minmod(u_here - u_km, u_kp - u_here) / hz;
          } else if (adv == 3) {
            const double cx = (u_ip - u_im) / (2.0 * hx);
            const double cy = (u_jp - u_jm) / (2.0 * hy);
            const double cz = (u_kp - u_km) / (2.0 * hz);
            const double ux =
                u_here > 0.0 ? (u_here - u_im) / hx : (u_ip - u_here) / hx;
            const double uy =
                v_at_u > 0.0 ? (u_here - u_jm) / hy : (u_jp - u_here) / hy;
            const double uz =
                w_at_u > 0.0 ? (u_here - u_km) / hz : (u_kp - u_here) / hz;
            dudx = (1.0 - adv_blend) * cx + adv_blend * ux;
            dudy = (1.0 - adv_blend) * cy + adv_blend * uy;
            dudz = (1.0 - adv_blend) * cz + adv_blend * uz;
          } else if (adv == 4) {
            if (i >= 3 && i + 3 < n0) {
              dudx = weno5_derivative(
                  u(k, j, i - 3), u(k, j, i - 2), u(k, j, i - 1), u_here,
                  u(k, j, i + 1), u(k, j, i + 2), u(k, j, i + 3), u_here, hx);
            } else {
              dudx = minmod(u_here - u_im, u_ip - u_here) / hx;
            }
            if (j >= 3 && j + 3 < n1) {
              dudy = weno5_derivative(
                  u(k, j - 3, i), u(k, j - 2, i), u(k, j - 1, i), u_here,
                  u(k, j + 1, i), u(k, j + 2, i), u(k, j + 3, i), v_at_u, hy);
            } else {
              dudy = minmod(u_here - u_jm, u_jp - u_here) / hy;
            }
            if (!g.z_wall && n2 >= 5) {
              dudz = weno5_derivative(
                  u((k + n2 - 3) % n2, j, i), u((k + n2 - 2) % n2, j, i),
                  u((k + n2 - 1) % n2, j, i), u_here, u((k + 1) % n2, j, i),
                  u((k + 2) % n2, j, i), u((k + 3) % n2, j, i), w_at_u, hz);
            }
          }
          const double lap_u = (u_ip - 2.0 * u_here + u_im) / hx2 +
                               (u_jp - 2.0 * u_here + u_jm) / hy2 +
                               (u_kp - 2.0 * u_here + u_km) / hz2;
          du(k, j, i) =
              -(u_here * dudx + v_at_u * dudy + w_at_u * dudz) + nu * lap_u;
        }

        // ----- v-momentum (interior y-faces only) -----
        if (j == 0) {
          dv(k, j, i) = 0.0;
        } else {
          const double v_im = (im < 0) ? -v_here : v(k, j, im);
          const double v_ip = (ip < 0) ? -v_here : v(k, j, ip);
          const double v_jm = (jm < 0) ? 0.0 : v(k, jm, i);
          const double v_jp = (jp < 0) ? 0.0 : v(k, jp, i);
          const double v_km = (k == 0) ? (g.z_wall ? -v_here : v(n2 - 1, j, i))
                                       : v(k - 1, j, i);
          const double v_kp = (k + 1 == n2) ? (g.z_wall ? -v_here : v(0, j, i))
                                            : v(k + 1, j, i);
          double dvdx = (v_ip - v_im) / (2.0 * hx);
          double dvdy = (v_jp - v_jm) / (2.0 * hy);
          double dvdz = (v_kp - v_km) / (2.0 * hz);
          const double u_here_i = (i == 0) ? 0.0 : u(k, j, i);
          const double u_here_ip = (ip < 0) ? 0.0 : u(k, j, ip);
          const double u_jm_i = (jm < 0) ? 0.0 : u(k, jm, i);
          const double u_jm_ip = (jm < 0 || ip < 0) ? 0.0 : u(k, jm, ip);
          const double u_at_v =
              0.25 * (u_jm_i + u_here_i + u_jm_ip + u_here_ip);
          const double w_at_v =
              0.25 * (w(k, jm, i) + w(k, j, i) + w(kp, jm, i) + w(kp, j, i));
          if (adv == 1) {
            dvdx = u_at_v > 0.0 ? (v_here - v_im) / hx : (v_ip - v_here) / hx;
            dvdy = v_here > 0.0 ? (v_here - v_jm) / hy : (v_jp - v_here) / hy;
            dvdz = w_at_v > 0.0 ? (v_here - v_km) / hz : (v_kp - v_here) / hz;
          } else if (adv == 2) {
            dvdx = minmod(v_here - v_im, v_ip - v_here) / hx;
            dvdy = minmod(v_here - v_jm, v_jp - v_here) / hy;
            dvdz = minmod(v_here - v_km, v_kp - v_here) / hz;
          } else if (adv == 3) {
            const double cx = (v_ip - v_im) / (2.0 * hx);
            const double cy = (v_jp - v_jm) / (2.0 * hy);
            const double cz = (v_kp - v_km) / (2.0 * hz);
            const double ux =
                u_at_v > 0.0 ? (v_here - v_im) / hx : (v_ip - v_here) / hx;
            const double uy =
                v_here > 0.0 ? (v_here - v_jm) / hy : (v_jp - v_here) / hy;
            const double uz =
                w_at_v > 0.0 ? (v_here - v_km) / hz : (v_kp - v_here) / hz;
            dvdx = (1.0 - adv_blend) * cx + adv_blend * ux;
            dvdy = (1.0 - adv_blend) * cy + adv_blend * uy;
            dvdz = (1.0 - adv_blend) * cz + adv_blend * uz;
          } else if (adv == 4) {
            if (i >= 3 && i + 3 < n0) {
              dvdx = weno5_derivative(
                  v(k, j, i - 3), v(k, j, i - 2), v(k, j, i - 1), v_here,
                  v(k, j, i + 1), v(k, j, i + 2), v(k, j, i + 3), u_at_v, hx);
            } else {
              dvdx = minmod(v_here - v_im, v_ip - v_here) / hx;
            }
            if (j >= 3 && j + 3 < n1) {
              dvdy = weno5_derivative(
                  v(k, j - 3, i), v(k, j - 2, i), v(k, j - 1, i), v_here,
                  v(k, j + 1, i), v(k, j + 2, i), v(k, j + 3, i), v_here, hy);
            } else {
              dvdy = minmod(v_here - v_jm, v_jp - v_here) / hy;
            }
            if (!g.z_wall && n2 >= 5) {
              dvdz = weno5_derivative(
                  v((k + n2 - 3) % n2, j, i), v((k + n2 - 2) % n2, j, i),
                  v((k + n2 - 1) % n2, j, i), v_here, v((k + 1) % n2, j, i),
                  v((k + 2) % n2, j, i), v((k + 3) % n2, j, i), w_at_v, hz);
            }
          }
          const double lap_v = (v_ip - 2.0 * v_here + v_im) / hx2 +
                               (v_jp - 2.0 * v_here + v_jm) / hy2 +
                               (v_kp - 2.0 * v_here + v_km) / hz2;
          dv(k, j, i) =
              -(u_at_v * dvdx + v_here * dvdy + w_at_v * dvdz) + nu * lap_v;
        }

        // ----- w-momentum -----
        const double w_im = (im < 0) ? -w_here : w(k, j, im);
        const double w_ip = (ip < 0) ? -w_here : w(k, j, ip);
        const double w_jm = (jm < 0) ? -w_here : w(k, jm, i);
        const double w_jp = (jp < 0) ? -w_here : w(k, jp, i);
        const double w_km =
            (k == 0) ? 0.0 : ((g.z_wall) ? w(k - 1, j, i) : w(km, j, i));
        const double w_kp =
            (k + 1 == n2) ? 0.0 : ((g.z_wall) ? w(k + 1, j, i) : w(kp, j, i));
        double dwdx = (w_ip - w_im) / (2.0 * hx);
        double dwdy = (w_jp - w_jm) / (2.0 * hy);
        double dwdz = (w_kp - w_km) / (2.0 * hz);
        const double u_ip = (ip < 0) ? 0.0 : u(k, j, ip);
        const double u_km_ip = (ip < 0) ? 0.0 : u(km, j, ip);
        const double u_at_w =
            0.25 * (u(km, j, i) + u(k, j, i) + u_km_ip + u_ip);
        const double v_jp_local = (jp < 0) ? 0.0 : v(k, jp, i);
        const double v_km_jp = (jp < 0) ? 0.0 : v(km, jp, i);
        const double v_at_w =
            0.25 * (v(km, j, i) + v(k, j, i) + v_km_jp + v_jp_local);
        if (adv == 1) {
          dwdx = u_at_w > 0.0 ? (w_here - w_im) / hx : (w_ip - w_here) / hx;
          dwdy = v_at_w > 0.0 ? (w_here - w_jm) / hy : (w_jp - w_here) / hy;
          dwdz = w_here > 0.0 ? (w_here - w_km) / hz : (w_kp - w_here) / hz;
        } else if (adv == 2) {
          dwdx = minmod(w_here - w_im, w_ip - w_here) / hx;
          dwdy = minmod(w_here - w_jm, w_jp - w_here) / hy;
          dwdz = minmod(w_here - w_km, w_kp - w_here) / hz;
        } else if (adv == 3) {
          const double cx = (w_ip - w_im) / (2.0 * hx);
          const double cy = (w_jp - w_jm) / (2.0 * hy);
          const double cz = (w_kp - w_km) / (2.0 * hz);
          const double ux =
              u_at_w > 0.0 ? (w_here - w_im) / hx : (w_ip - w_here) / hx;
          const double uy =
              v_at_w > 0.0 ? (w_here - w_jm) / hy : (w_jp - w_here) / hy;
          const double uz =
              w_here > 0.0 ? (w_here - w_km) / hz : (w_kp - w_here) / hz;
          dwdx = (1.0 - adv_blend) * cx + adv_blend * ux;
          dwdy = (1.0 - adv_blend) * cy + adv_blend * uy;
          dwdz = (1.0 - adv_blend) * cz + adv_blend * uz;
        } else if (adv == 4) {
          if (i >= 3 && i + 3 < n0) {
            dwdx = weno5_derivative(w(k, j, i - 3), w(k, j, i - 2),
                                    w(k, j, i - 1), w_here, w(k, j, i + 1),
                                    w(k, j, i + 2), w(k, j, i + 3), u_at_w, hx);
          } else {
            dwdx = minmod(w_here - w_im, w_ip - w_here) / hx;
          }
          if (j >= 3 && j + 3 < n1) {
            dwdy = weno5_derivative(w(k, j - 3, i), w(k, j - 2, i),
                                    w(k, j - 1, i), w_here, w(k, j + 1, i),
                                    w(k, j + 2, i), w(k, j + 3, i), v_at_w, hy);
          } else {
            dwdy = minmod(w_here - w_jm, w_jp - w_here) / hy;
          }
          if (!g.z_wall && n2 >= 5) {
            dwdz = weno5_derivative(
                w((k + n2 - 3) % n2, j, i), w((k + n2 - 2) % n2, j, i),
                w((k + n2 - 1) % n2, j, i), w_here, w((k + 1) % n2, j, i),
                w((k + 2) % n2, j, i), w((k + 3) % n2, j, i), w_here, hz);
          }
        }
        const double lap_w = (w_ip - 2.0 * w_here + w_im) / hx2 +
                             (w_jp - 2.0 * w_here + w_jm) / hy2 +
                             (w_kp - 2.0 * w_here + w_km) / hz2;
        dw(k, j, i) =
            -(u_at_w * dwdx + v_at_w * dwdy + w_here * dwdz) + nu * lap_w;
        if (g.z_wall && k == 0) {
          dw(k, j, i) = 0.0;
        }
      });
}

// Weak low-pass (dealiasing) filter: q <- q + alpha * laplacian(q).
// Applied to the velocity components after a time step; it damps the
// grid-scale content that a high-order reconstruction can alias.
template <class ExecSpace>
void filter_velocity(const projection::ScalarView<ExecSpace>& u,
                     const projection::ScalarView<ExecSpace>& v,
                     const projection::ScalarView<ExecSpace>& w,
                     projection::ScalarView<ExecSpace>& u_out,
                     projection::ScalarView<ExecSpace>& v_out,
                     projection::ScalarView<ExecSpace>& w_out,
                     const Geometry& g, double alpha) {
  const int n0 = g.n0;
  const int n1 = g.n1;
  const int n2 = g.n2;
  const auto policy =
      make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  Kokkos::parallel_for(
      "cavity_filter", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        auto at = [&](const projection::ScalarView<ExecSpace>& q, int kk,
                      int jj, int ii) {
          const int ci = ii < 0 ? 0 : (ii >= n0 ? n0 - 1 : ii);
          const int cj = jj < 0 ? 0 : (jj >= n1 ? n1 - 1 : jj);
          int ck = ((kk % n2) + n2) % n2;
          if (g.z_wall) {
            ck = kk < 0 ? 0 : (kk >= n2 ? n2 - 1 : kk);
          }
          return q(ck, cj, ci);
        };
        const double uu = u(k, j, i);
        const double vv = v(k, j, i);
        const double ww = w(k, j, i);
        const double lu = at(u, k, j, i - 1) + at(u, k, j, i + 1) - 2.0 * uu +
                          at(u, k, j - 1, i) + at(u, k, j + 1, i) - 2.0 * uu +
                          at(u, k - 1, j, i) + at(u, k + 1, j, i) - 2.0 * uu;
        const double lv = at(v, k, j, i - 1) + at(v, k, j, i + 1) - 2.0 * vv +
                          at(v, k, j - 1, i) + at(v, k, j + 1, i) - 2.0 * vv +
                          at(v, k - 1, j, i) + at(v, k + 1, j, i) - 2.0 * vv;
        const double lw = at(w, k, j, i - 1) + at(w, k, j, i + 1) - 2.0 * ww +
                          at(w, k, j - 1, i) + at(w, k, j + 1, i) - 2.0 * ww +
                          at(w, k - 1, j, i) + at(w, k + 1, j, i) - 2.0 * ww;
        u_out(k, j, i) = (i == 0) ? 0.0 : uu + alpha * lu;
        v_out(k, j, i) = (j == 0) ? 0.0 : vv + alpha * lv;
        w_out(k, j, i) = (k == 0) ? 0.0 : ww + alpha * lw;
      });
}

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
      make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
  const double dt = 1.0;
  Kokkos::parallel_for(
      "cavity_projection_rhs", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        scratch(k, j, i) = -scratch(k, j, i) / dt;
      });
  const poisson::Result result = solver.solve(phi, scratch, options);

  Kokkos::parallel_for(
      "cavity_projection_update", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? n0 - 1 : i - 1;
        const int jm = (j == 0) ? n1 - 1 : j - 1;
        const int km = (k == 0) ? (g.z_wall ? -1 : n2 - 1) : k - 1;
        if (i > 0) {
          u(k, j, i) += dt * (phi(k, j, i) - phi(k, j, im)) / g.hx;
        } else {
          u(k, j, i) = 0.0;
        }
        if (j > 0) {
          v(k, j, i) += dt * (phi(k, j, i) - phi(k, jm, i)) / g.hy;
        } else {
          v(k, j, i) = 0.0;
        }
        if (k > 0) {
          w(k, j, i) += dt * (phi(k, j, i) - phi(km, j, i)) / g.hz;
        } else if (g.z_wall) {
          w(k, j, i) = 0.0;
        } else {
          w(k, j, i) += dt * (phi(k, j, i) - phi(km, j, i)) / g.hz;
        }
      });
  return result;
}

template <class ExecSpace, class Project>
void advance_ssprk3(projection::ScalarView<ExecSpace>& u,
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
  momentum_rhs<ExecSpace>(u, v, w, ku, kv, kw, g, nu);
  incompressible::combine<ExecSpace>(u, 1.0, ku, dt, ku, 0.0, u1);
  incompressible::combine<ExecSpace>(v, 1.0, kv, dt, ku, 0.0, v1);
  incompressible::combine<ExecSpace>(w, 1.0, kw, dt, ku, 0.0, w1);
  project(u1, v1, w1);

  momentum_rhs<ExecSpace>(u1, v1, w1, ku, kv, kw, g, nu);
  incompressible::combine<ExecSpace>(u, 0.75, u1, 0.25, ku, 0.25 * dt, u2);
  incompressible::combine<ExecSpace>(v, 0.75, v1, 0.25, kv, 0.25 * dt, v2);
  incompressible::combine<ExecSpace>(w, 0.75, w1, 0.25, kw, 0.25 * dt, w2);
  project(u2, v2, w2);

  momentum_rhs<ExecSpace>(u2, v2, w2, ku, kv, kw, g, nu);
  incompressible::combine<ExecSpace>(u, 1.0 / 3.0, u2, 2.0 / 3.0, ku,
                                     (2.0 / 3.0) * dt, u);
  incompressible::combine<ExecSpace>(v, 1.0 / 3.0, v2, 2.0 / 3.0, kv,
                                     (2.0 / 3.0) * dt, v);
  incompressible::combine<ExecSpace>(w, 1.0 / 3.0, w2, 2.0 / 3.0, kw,
                                     (2.0 / 3.0) * dt, w);
  project(u, v, w);
}

}  // namespace cavity
}  // namespace saturn
