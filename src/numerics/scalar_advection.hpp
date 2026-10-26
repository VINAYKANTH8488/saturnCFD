#pragma once

#include <Kokkos_Core.hpp>

#include <array>

#include "fields/patch_field.hpp"

namespace saturn {

// Second-order central advection-diffusion operator for a passive scalar:
//
//   L(c) = -u . grad(c) + nu * laplacian(c)
//
// Periodic ghost cells must be filled before calling this.
template <class ExecSpace>
void scalar_advection_diffusion_rhs(const PatchField<ExecSpace>& c,
                                    PatchField<ExecSpace>& rhs,
                                    const std::array<double, 3>& velocity,
                                    double nu,
                                    const std::array<double, 3>& h) {
  const Box box = c.box();
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;

  const double ux = velocity[0];
  const double uy = velocity[1];
  const double uz = velocity[2];
  const double hx = h[0];
  const double hy = h[1];
  const double hz = h[2];
  const double inv_hx2 = 1.0 / (hx * hx);
  const double inv_hy2 = 1.0 / (hy * hy);
  const double inv_hz2 = 1.0 / (hz * hz);

  Kokkos::parallel_for(
      "scalar_advection_diffusion",
      policy_type({box.lo[2], box.lo[1], box.lo[0]},
                  {box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}),
      KOKKOS_LAMBDA(int k, int j, int i) {
        const double cc = c(0, i, j, k);
        const double adv =
            0.5 * ux * (c(0, i + 1, j, k) - c(0, i - 1, j, k)) / hx +
            0.5 * uy * (c(0, i, j + 1, k) - c(0, i, j - 1, k)) / hy +
            0.5 * uz * (c(0, i, j, k + 1) - c(0, i, j, k - 1)) / hz;
        const double lap =
            (c(0, i + 1, j, k) - 2.0 * cc + c(0, i - 1, j, k)) * inv_hx2 +
            (c(0, i, j + 1, k) - 2.0 * cc + c(0, i, j - 1, k)) * inv_hy2 +
            (c(0, i, j, k + 1) - 2.0 * cc + c(0, i, j, k - 1)) * inv_hz2;
        rhs(0, i, j, k) = -adv + nu * lap;
      });
}

}  // namespace saturn
