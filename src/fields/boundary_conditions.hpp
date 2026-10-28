#pragma once

#include <Kokkos_Core.hpp>

#include "core/parallel.hpp"
#include "fields/patch_field.hpp"
#include "mesh/box.hpp"

#include <string>

namespace saturn {

// Ghost-cell boundary treatment for the compressible solver. `bc[d][0]` is
// the low side and `bc[d][1]` the high side of dimension d.
//
// Periodic ghosts are filled by wrapping the global coordinate. This routine
// is used on a single rank (or a rank whose local box is the whole domain);
// multi-rank periodic exchange is handled by BoxExchange. Transmissive and
// slip-wall boundaries are suitable for the first single-rank shock cases.
enum class BoundaryType {
  Periodic,
  Transmissive,
  SlipWall,
};

inline BoundaryType parse_boundary_type(const std::string& name) {
  if (name == "periodic") {
    return BoundaryType::Periodic;
  }
  if (name == "slip" || name == "wall" || name == "reflective") {
    return BoundaryType::SlipWall;
  }
  return BoundaryType::Transmissive;
}

// Fill all ghost cells of a five-variable compressible field. The kernel
// treats each coordinate independently, so edges and corners are filled too.
template <class ExecSpace>
void apply_boundary_conditions(PatchField<ExecSpace>& q,
                               const BoundaryType bc[3][2]) {
  const Box box = q.box();
  const int ng = q.ng();
  const int np0 = box.length(0) + 2 * ng;
  const int np1 = box.length(1) + 2 * ng;
  const int np2 = box.length(2) + 2 * ng;
  const auto policy = make_mdrange3<ExecSpace>(Int3{{0, 0, 0}},
                                               Int3{{np2, np1, np0}});

  Kokkos::parallel_for(
      "compressible_boundary", policy,
      KOKKOS_LAMBDA(int kp, int jp, int ip) {
        const int i = box.lo[0] - ng + ip;
        const int j = box.lo[1] - ng + jp;
        const int k = box.lo[2] - ng + kp;
        const bool outside = i < box.lo[0] || i > box.hi[0] ||
                             j < box.lo[1] || j > box.hi[1] ||
                             k < box.lo[2] || k > box.hi[2];
        if (!outside) {
          return;
        }

        int si = i;
        int sj = j;
        int sk = k;
        bool flip_x = false;
        bool flip_y = false;
        bool flip_z = false;

        const int coordinate[3] = {i, j, k};
        int source[3] = {i, j, k};
        bool flip[3] = {false, false, false};
        for (int d = 0; d < 3; ++d) {
          const int lo = box.lo[d];
          const int hi = box.hi[d];
          if (coordinate[d] >= lo && coordinate[d] <= hi) {
            continue;
          }
          const int side = coordinate[d] < lo ? 0 : 1;
          if (bc[d][side] == BoundaryType::Periodic) {
            const int n = hi - lo + 1;
            int wrapped = (coordinate[d] - lo) % n;
            if (wrapped < 0) {
              wrapped += n;
            }
            source[d] = lo + wrapped;
          } else if (bc[d][side] == BoundaryType::SlipWall) {
            if (coordinate[d] < lo) {
              source[d] = 2 * lo - coordinate[d] - 1;
            } else {
              source[d] = 2 * hi - coordinate[d] + 1;
            }
            flip[d] = true;
          } else {
            source[d] = coordinate[d] < lo ? lo : hi;
          }
        }
        si = source[0];
        sj = source[1];
        sk = source[2];
        flip_x = flip[0];
        flip_y = flip[1];
        flip_z = flip[2];

        const int source_ip = si - box.lo[0] + ng;
        const int source_jp = sj - box.lo[1] + ng;
        const int source_kp = sk - box.lo[2] + ng;
        auto data = q.data();
        for (int v = 0; v < q.nvar(); ++v) {
          double value = data(v, source_kp, source_jp, source_ip);
          if ((v == 1 && flip_x) || (v == 2 && flip_y) ||
              (v == 3 && flip_z)) {
            value = -value;
          }
          data(v, kp, jp, ip) = value;
        }
      });
}

}  // namespace saturn
