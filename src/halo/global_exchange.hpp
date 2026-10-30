#pragma once

#include <Kokkos_Core.hpp>

#include "core/parallel.hpp"
#include "fields/boundary_conditions.hpp"
#include "fields/patch_field.hpp"
#include "mesh/decomposition.hpp"

#include <cstdio>
#include <vector>

#ifdef SATURN_ENABLE_MPI
#include <mpi.h>
#endif

namespace saturn {

// Correctness-first global halo/BC fill for multi-rank non-periodic runs.
//
// Every rank all-gathers the interior fields, then each rank fills its ghost
// cells by mapping the global coordinate through the boundary conditions and
// looking up the owning rank's local block. This handles faces, edges, and
// corners and mixed periodic/transmissive/slip boundaries in one pass.
//
// It is O(N) communication per exchange and is intended as a transitional
// fallback; all-periodic runs use the scalable BoxExchange.
template <class ExecSpace>
class GlobalExchange {
 public:
  GlobalExchange(const Decomposition& decomposition, int nvar, int ng)
      : decomposition_(decomposition), nvar_(nvar), ng_(ng) {}

  void exchange(PatchField<ExecSpace>& q, const BoundaryType bc[3][2]) {
#ifndef SATURN_ENABLE_MPI
    applyBoundaryConditions(q, bc);
    (void)q;
#else
    if (decomposition_.size == 1) {
      applyBoundaryConditions(q, bc);
      return;
    }

    const Box local = decomposition_.local;
    const int local_count =
        nvar_ * local.length(0) * local.length(1) * local.length(2);

    int meta[6] = {local.lo[0], local.lo[1], local.lo[2],
                   local.hi[0], local.hi[1], local.hi[2]};
    std::vector<int> all_meta(static_cast<std::size_t>(6) *
                              decomposition_.size);
    MPI_Allgather(meta, 6, MPI_INT, all_meta.data(), 6, MPI_INT,
                  decomposition_.comm);

    std::vector<int> counts(static_cast<std::size_t>(decomposition_.size));
    std::vector<int> displacements(
        static_cast<std::size_t>(decomposition_.size));
    int total = 0;
    for (int rank = 0; rank < decomposition_.size; ++rank) {
      const int* m = &all_meta[static_cast<std::size_t>(rank) * 6];
      const int len0 = m[3] - m[0] + 1;
      const int len1 = m[4] - m[1] + 1;
      const int len2 = m[5] - m[2] + 1;
      counts[static_cast<std::size_t>(rank)] = nvar_ * len0 * len1 * len2;
      displacements[static_cast<std::size_t>(rank)] = total;
      total += counts[static_cast<std::size_t>(rank)];
    }

    auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q.data());
    std::vector<double> send(static_cast<std::size_t>(local_count));
    int index = 0;
    for (int v = 0; v < nvar_; ++v) {
      for (int k = local.lo[2]; k <= local.hi[2]; ++k) {
        for (int j = local.lo[1]; j <= local.hi[1]; ++j) {
          for (int i = local.lo[0]; i <= local.hi[0]; ++i) {
            send[static_cast<std::size_t>(index++)] =
                host(v, k - local.lo[2] + ng_, j - local.lo[1] + ng_,
                     i - local.lo[0] + ng_);
          }
        }
      }
    }
    std::vector<double> global(static_cast<std::size_t>(total));
    MPI_Allgatherv(send.data(), local_count, MPI_DOUBLE, global.data(),
                   counts.data(), displacements.data(), MPI_DOUBLE,
                   decomposition_.comm);

    const Box global_box = decomposition_.global;
    const int np0 = local.length(0) + 2 * ng_;
    const int np1 = local.length(1) + 2 * ng_;
    const int np2 = local.length(2) + 2 * ng_;
    for (int kp = 0; kp < np2; ++kp) {
      for (int jp = 0; jp < np1; ++jp) {
        for (int ip = 0; ip < np0; ++ip) {
          const int i = local.lo[0] - ng_ + ip;
          const int j = local.lo[1] - ng_ + jp;
          const int k = local.lo[2] - ng_ + kp;
          const bool outside = i < local.lo[0] || i > local.hi[0] ||
                               j < local.lo[1] || j > local.hi[1] ||
                               k < local.lo[2] || k > local.hi[2];
          if (!outside) {
            continue;
          }

          const int coordinate[3] = {i, j, k};
          int source[3] = {i, j, k};
          bool flip[3] = {false, false, false};
          for (int d = 0; d < 3; ++d) {
            const int lo = global_box.lo[d];
            const int hi = global_box.hi[d];
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
              source[d] = coordinate[d] < lo ? 2 * lo - coordinate[d] - 1
                                             : 2 * hi - coordinate[d] + 1;
              flip[d] = true;
            } else {
              source[d] = coordinate[d] < lo ? lo : hi;
            }
          }

          int owner = -1;
          for (int rank = 0; rank < decomposition_.size; ++rank) {
            const int* m = &all_meta[static_cast<std::size_t>(rank) * 6];
            if (source[0] >= m[0] && source[0] <= m[3] && source[1] >= m[1] &&
                source[1] <= m[4] && source[2] >= m[2] && source[2] <= m[5]) {
              owner = rank;
              break;
            }
          }
          if (owner < 0) {
            continue;  // should not happen for a valid BC mapping
          }
          const int* m = &all_meta[static_cast<std::size_t>(owner) * 6];
          const int len0 = m[3] - m[0] + 1;
          const int len1 = m[4] - m[1] + 1;
          const int base = displacements[static_cast<std::size_t>(owner)];
          for (int v = 0; v < nvar_; ++v) {
            const int source_index =
                base +
                ((v * (m[5] - m[2] + 1) + (source[2] - m[2])) * len1 +
                 (source[1] - m[1])) *
                    len0 +
                (source[0] - m[0]);
            double value = global[static_cast<std::size_t>(source_index)];
            if ((v == 1 && flip[0]) || (v == 2 && flip[1]) ||
                (v == 3 && flip[2])) {
              value = -value;
            }
            host(v, kp, jp, ip) = value;
          }
        }
      }
    }
    Kokkos::deep_copy(q.data(), host);
#endif
  }

 private:
  Decomposition decomposition_{};
  int nvar_ = 5;
  int ng_ = 3;
};

}  // namespace saturn
