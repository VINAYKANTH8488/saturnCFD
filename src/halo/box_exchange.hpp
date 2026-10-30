#pragma once

#include <Kokkos_Core.hpp>

#include "fields/patch_field.hpp"
#include "mesh/decomposition.hpp"

#include <vector>

#ifdef SATURN_ENABLE_MPI
#include <mpi.h>
#endif

namespace saturn {

// Full 3D halo exchange for cell-centered fields.
//
// Unlike FaceExchange, this fills edges and corners as well as faces, which is
// required by the fourth-order viscous stencil. It exchanges all 26 Cartesian
// neighbor blocks. The implementation is host-staged (MPI is not assumed to be
// GPU-aware) and currently uses one MPI_Sendrecv per block; it is correct for
// moderate grids and can be replaced by a fused neighbor exchange later.
//
// Non-periodic boundaries are skipped; apply the physical boundary conditions
// after this exchange.
template <class ExecSpace>
class BoxExchange {
 public:
  BoxExchange(const Decomposition& decomposition, int nvar, int ng)
      : decomposition_(decomposition), nvar_(nvar), ng_(ng) {}

  void exchange(PatchField<ExecSpace>& q) {
    const bool all_periodic = decomposition_.periodic[0] &&
                              decomposition_.periodic[1] &&
                              decomposition_.periodic[2];
#ifndef SATURN_ENABLE_MPI
    if (all_periodic) {
      q.fillPeriodic();
    }
    (void)q;
#else
    if (decomposition_.size == 1) {
      if (all_periodic) {
        q.fillPeriodic();
      }
      return;
    }
    (void)all_periodic;

    auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q.data());

    struct Pending {
      int neighbor = 0;
      int count = 0;
      int send_tag = 0;
      int recv_tag = 0;
      int offset[3] = {0, 0, 0};
      int extent[3] = {0, 0, 0};
      std::vector<double> send;
      std::vector<double> recv;
      MPI_Request requests[2]{MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    };

    std::vector<Pending> pending;
    pending.reserve(26);
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          if (ox == 0 && oy == 0 && oz == 0) {
            continue;
          }
          const Box box = decomposition_.local;
          const int off[3] = {ox, oy, oz};
          int extent[3];
          for (int d = 0; d < 3; ++d) {
            extent[d] = (off[d] == 0) ? box.length(d) : ng_;
          }

          int neighbor_coords[3];
          bool skip = false;
          for (int d = 0; d < 3; ++d) {
            int coordinate = decomposition_.coords[d] + off[d];
            if (coordinate < 0 || coordinate >= decomposition_.dims[d]) {
              if (!decomposition_.periodic[d]) {
                skip = true;
                break;
              }
              coordinate = (coordinate + decomposition_.dims[d]) %
                           decomposition_.dims[d];
            }
            neighbor_coords[d] = coordinate;
          }
          if (skip) {
            continue;
          }

          Pending block;
          int neighbor = decomposition_.rank;
          MPI_Cart_rank(decomposition_.comm, neighbor_coords, &neighbor);
          block.neighbor = neighbor;
          block.offset[0] = ox;
          block.offset[1] = oy;
          block.offset[2] = oz;
          block.extent[0] = extent[0];
          block.extent[1] = extent[1];
          block.extent[2] = extent[2];
          block.count = nvar_ * extent[0] * extent[1] * extent[2];
          block.send_tag = tagFor(ox, oy, oz);
          block.recv_tag = tagFor(-ox, -oy, -oz);
          block.send.resize(static_cast<std::size_t>(block.count));
          block.recv.resize(static_cast<std::size_t>(block.count));
          if (neighbor == decomposition_.rank) {
            // A periodic dimension with one rank maps a boundary to its own
            // opposite side, not to the same side. Pack the mirror block.
            const int mirror[3] = {-ox, -oy, -oz};
            pack(host, box, mirror, extent, block.send);
            unpack(host, box, off, extent, block.send);
            continue;
          }
          pack(host, box, off, extent, block.send);
          pending.push_back(std::move(block));
        }
      }
    }

    for (auto& block : pending) {
      MPI_Isend(block.send.data(), block.count, MPI_DOUBLE, block.neighbor,
                block.send_tag, decomposition_.comm, &block.requests[0]);
      MPI_Irecv(block.recv.data(), block.count, MPI_DOUBLE, block.neighbor,
                block.recv_tag, decomposition_.comm, &block.requests[1]);
    }
    for (auto& block : pending) {
      MPI_Waitall(2, block.requests, MPI_STATUSES_IGNORE);
    }
    for (auto& block : pending) {
      unpack(host, decomposition_.local, block.offset, block.extent,
             block.recv);
    }
    Kokkos::deep_copy(q.data(), host);
#endif
  }

 private:
  static int tagFor(int ox, int oy, int oz) {
    return 100 + (ox + 1) * 9 + (oy + 1) * 3 + (oz + 1);
  }

#ifdef SATURN_ENABLE_MPI
  template <class HostView>
  void pack(const HostView& host, const Box& box, const int off[3],
            const int extent[3], std::vector<double>& buffer) const {
    int index = 0;
    for (int v = 0; v < nvar_; ++v) {
      for (int kk = 0; kk < extent[2]; ++kk) {
        const int k = (off[2] > 0) ? box.hi[2] - ng_ + 1 + kk : box.lo[2] + kk;
        for (int jj = 0; jj < extent[1]; ++jj) {
          const int j =
              (off[1] > 0) ? box.hi[1] - ng_ + 1 + jj : box.lo[1] + jj;
          for (int ii = 0; ii < extent[0]; ++ii) {
            const int i =
                (off[0] > 0) ? box.hi[0] - ng_ + 1 + ii : box.lo[0] + ii;
            buffer[static_cast<std::size_t>(index++)] =
                host(v, k - box.lo[2] + ng_, j - box.lo[1] + ng_,
                     i - box.lo[0] + ng_);
          }
        }
      }
    }
  }

  template <class HostView>
  void unpack(HostView& host, const Box& box, const int off[3],
              const int extent[3], const std::vector<double>& buffer) const {
    int index = 0;
    for (int v = 0; v < nvar_; ++v) {
      for (int kk = 0; kk < extent[2]; ++kk) {
        const int k =
            (off[2] > 0) ? box.hi[2] + 1 + kk
                         : (off[2] < 0 ? box.lo[2] - ng_ + kk : box.lo[2] + kk);
        for (int jj = 0; jj < extent[1]; ++jj) {
          const int j = (off[1] > 0) ? box.hi[1] + 1 + jj
                                     : (off[1] < 0 ? box.lo[1] - ng_ + jj
                                                   : box.lo[1] + jj);
          for (int ii = 0; ii < extent[0]; ++ii) {
            const int i = (off[0] > 0) ? box.hi[0] + 1 + ii
                                       : (off[0] < 0 ? box.lo[0] - ng_ + ii
                                                     : box.lo[0] + ii);
            host(v, k - box.lo[2] + ng_, j - box.lo[1] + ng_,
                 i - box.lo[0] + ng_) =
                buffer[static_cast<std::size_t>(index++)];
          }
        }
      }
    }
  }
#endif

  Decomposition decomposition_{};
  int nvar_ = 1;
  int ng_ = 3;
};

}  // namespace saturn
