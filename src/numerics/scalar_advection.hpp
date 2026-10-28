#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>

#include "core/parallel.hpp"
#include "fields/patch_field.hpp"
#include "mesh/box.hpp"

namespace saturn {

// Pointwise second-order central advection-diffusion operator:
//
//   L(c) = -u . grad(c) + nu * laplacian(c)
//
// The caller must ensure the required neighbors are valid. This helper is
// shared by the interior and boundary-shell kernels.
template <class PatchType>
KOKKOS_INLINE_FUNCTION double scalar_rhs_value(const PatchType& c, int i, int j,
                                               int k, double ux, double uy,
                                               double uz, double hx, double hy,
                                               double hz, double nu) {
  const double cc = c(0, i, j, k);
  const double adv = 0.5 * ux * (c(0, i + 1, j, k) - c(0, i - 1, j, k)) / hx +
                     0.5 * uy * (c(0, i, j + 1, k) - c(0, i, j - 1, k)) / hy +
                     0.5 * uz * (c(0, i, j, k + 1) - c(0, i, j, k - 1)) / hz;
  const double lap =
      (c(0, i + 1, j, k) - 2.0 * cc + c(0, i - 1, j, k)) / (hx * hx) +
      (c(0, i, j + 1, k) - 2.0 * cc + c(0, i, j - 1, k)) / (hy * hy) +
      (c(0, i, j, k + 1) - 2.0 * cc + c(0, i, j, k - 1)) / (hz * hz);
  return -adv + nu * lap;
}

template <class ExecSpace>
void scalar_advection_diffusion_rhs(const PatchField<ExecSpace>& c,
                                    PatchField<ExecSpace>& rhs,
                                    const std::array<double, 3>& velocity,
                                    double nu, const std::array<double, 3>& h,
                                    const Box& region) {
  if (region.empty()) {
    return;
  }
  const auto policy = make_mdrange3<ExecSpace>(
      Int3{{region.lo[2], region.lo[1], region.lo[0]}},
      Int3{{region.hi[2] + 1, region.hi[1] + 1, region.hi[0] + 1}});

  const double ux = velocity[0];
  const double uy = velocity[1];
  const double uz = velocity[2];
  const double hx = h[0];
  const double hy = h[1];
  const double hz = h[2];

  Kokkos::parallel_for(
      "scalar_advection_diffusion", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        rhs(0, i, j, k) =
            scalar_rhs_value(c, i, j, k, ux, uy, uz, hx, hy, hz, nu);
      });
}

// Full-box operator (all interior cells).
template <class ExecSpace>
void scalar_advection_diffusion_rhs(const PatchField<ExecSpace>& c,
                                    PatchField<ExecSpace>& rhs,
                                    const std::array<double, 3>& velocity,
                                    double nu, const std::array<double, 3>& h) {
  scalar_advection_diffusion_rhs(c, rhs, velocity, nu, h, c.box());
}

// Boundary shell of a box: six faces, partitioned so every shell cell is
// visited exactly once. Requires valid ghosts (call after FaceExchange::end).
template <class ExecSpace>
void scalar_advection_diffusion_rhs_shell(const PatchField<ExecSpace>& c,
                                          PatchField<ExecSpace>& rhs,
                                          const std::array<double, 3>& velocity,
                                          double nu,
                                          const std::array<double, 3>& h,
                                          const Box& box) {
  const int lo0 = box.lo[0];
  const int hi0 = box.hi[0];
  const int lo1 = box.lo[1];
  const int hi1 = box.hi[1];
  const int lo2 = box.lo[2];
  const int hi2 = box.hi[2];
  const int nx = box.length(0);
  const int ny = box.length(1);
  const int nz = box.length(2);
  const int nmax = std::max({nx, ny, nz});

  const auto policy =
      make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{6, nmax, nmax}});

  const double ux = velocity[0];
  const double uy = velocity[1];
  const double uz = velocity[2];
  const double hx = h[0];
  const double hy = h[1];
  const double hz = h[2];

  Kokkos::parallel_for(
      "scalar_rhs_shell", policy, KOKKOS_LAMBDA(int face, int a, int b) {
        int i = 0;
        int j = 0;
        int k = 0;
        bool active = false;
        if (face < 2) {
          // x faces: all tangential cells.
          if (a < ny && b < nz) {
            i = (face == 0) ? lo0 : hi0;
            j = lo1 + a;
            k = lo2 + b;
            active = true;
          }
        } else if (face < 4) {
          // y faces: exclude the x edges.
          if (a < nx && b < nz && nx > 2) {
            i = lo0 + a;
            j = (face == 2) ? lo1 : hi1;
            k = lo2 + b;
            active = (i > lo0 && i < hi0);
          }
        } else {
          // z faces: exclude the x and y edges.
          if (a < nx && b < ny && nx > 2 && ny > 2) {
            i = lo0 + a;
            j = lo1 + b;
            k = (face == 4) ? lo2 : hi2;
            active = (i > lo0 && i < hi0 && j > lo1 && j < hi1);
          }
        }
        if (active) {
          rhs(0, i, j, k) =
              scalar_rhs_value(c, i, j, k, ux, uy, uz, hx, hy, hz, nu);
        }
      });
}

}  // namespace saturn
