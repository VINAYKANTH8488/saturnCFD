#pragma once

#include <Kokkos_Core.hpp>

#include "core/parallel.hpp"
#include "fields/patch_field.hpp"
#include "mesh/decomposition.hpp"

#include <cstdio>

namespace saturn {

// Face-only halo exchange for cell-centered fields with ng ghost layers.
//
// Only the six face neighbors are needed by dimension-by-dimension stencils;
// edges and corners are not exchanged. The exchange is host-staged on device
// backends (MPI is not assumed to be GPU-aware). GPU-aware MPI can replace the
// staging later without changing the interface.
template <class ExecSpace>
class FaceExchange {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using buffer_type = Kokkos::View<double*, memory_space>;
  using host_buffer_type = Kokkos::View<double*, Kokkos::HostSpace>;

  FaceExchange(const Decomposition& decomposition, int nvar, int ng)
      : decomposition_(decomposition), nvar_(nvar), ng_(ng) {
    const Box& box = decomposition_.local;
    for (int d = 0; d < 3; ++d) {
      const int n1 = (d == 0) ? box.length(1) : box.length(0);
      const int n2 = (d == 0) ? box.length(2)
                              : ((d == 1) ? box.length(2) : box.length(1));
      n1_[d] = n1;
      n2_[d] = n2;
      size_[d] = nvar_ * ng_ * n1 * n2;
      for (int side = 0; side < 2; ++side) {
        send_[d][side] = buffer_type("send", size_[d]);
        recv_[d][side] = buffer_type("recv", size_[d]);
        host_send_[d][side] = Kokkos::create_mirror_view(send_[d][side]);
        host_recv_[d][side] = Kokkos::create_mirror_view(recv_[d][side]);
      }
    }
  }

  void exchange(PatchField<ExecSpace>& q) {
#ifndef SATURN_ENABLE_MPI
    q.fill_periodic();
    return;
#else
    for (int d = 0; d < 3; ++d) {
      int mpi_sides[2] = {0, 0};
      int nmpi = 0;

      // Handle self-periodic sides and pack all MPI sides first.
      for (int side_sign = -1; side_sign <= 1; side_sign += 2) {
        const int side = (side_sign < 0) ? 0 : 1;
        const int partner = decomposition_.partner(d, side_sign);
        if (partner == MPI_PROC_NULL) {
          continue;  // physical boundary; handled by BC kernels later
        }
        if (partner == decomposition_.rank) {
          // Periodic dimension with one rank: pack the opposite face and
          // unpack it into this side's ghosts.
          pack(q, d, -side_sign, send_[d][side]);
          Kokkos::deep_copy(recv_[d][side], send_[d][side]);
          unpack(q, d, side_sign, recv_[d][side]);
          continue;
        }
        pack(q, d, side_sign, send_[d][side]);
        Kokkos::deep_copy(host_send_[d][side], send_[d][side]);
        Kokkos::fence();
        mpi_sides[nmpi++] = side;
      }

      if (nmpi == 0) {
        continue;
      }

      // Post both sides before waiting: the receive tag of one side matches
      // the send tag of the opposite side on the neighbor.
      MPI_Request requests[4];
      int nreq = 0;
      for (int k = 0; k < nmpi; ++k) {
        const int side = mpi_sides[k];
        const int side_sign = (side == 0) ? -1 : 1;
        const int partner = decomposition_.partner(d, side_sign);
        const int send_tag = 100 + d * 2 + side;
        const int recv_tag = 100 + d * 2 + (1 - side);
        MPI_Isend(host_send_[d][side].data(), size_[d], MPI_DOUBLE, partner,
                  send_tag, decomposition_.comm, &requests[nreq++]);
        MPI_Irecv(host_recv_[d][side].data(), size_[d], MPI_DOUBLE, partner,
                  recv_tag, decomposition_.comm, &requests[nreq++]);
      }
      MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

      for (int k = 0; k < nmpi; ++k) {
        const int side = mpi_sides[k];
        const int side_sign = (side == 0) ? -1 : 1;
        Kokkos::deep_copy(recv_[d][side], host_recv_[d][side]);
        Kokkos::fence();
        unpack(q, d, side_sign, recv_[d][side]);
      }
    }
#endif
  }

 public:
  // pack/unpack must be public: nvcc's extended-lambda support rejects
  // __host__ __device__ lambdas inside private member functions.
  void pack(const PatchField<ExecSpace>& q, int d, int side_sign,
            buffer_type& buffer) {
    const Box box = decomposition_.local;
    const int ng = ng_;
    const int n1 = n1_[d];
    const int n2 = n2_[d];
    const int nvar = nvar_;
    const int lo0 = box.lo[0];
    const int lo1 = box.lo[1];
    const int lo2 = box.lo[2];
    const int hi0 = box.hi[0];
    const int hi1 = box.hi[1];
    const int hi2 = box.hi[2];

    const auto policy = make_mdrange4<ExecSpace>(
        Int4{{0, 0, 0, 0}}, Int4{{nvar, ng, n1, n2}});
    Kokkos::parallel_for(
        "halo_pack", policy,
        KOKKOS_LAMBDA(int v, int f, int a, int b) {
          int i = 0;
          int j = 0;
          int k = 0;
          if (d == 0) {
            i = (side_sign < 0) ? (lo0 + f) : (hi0 - ng + 1 + f);
            j = lo1 + a;
            k = lo2 + b;
          } else if (d == 1) {
            j = (side_sign < 0) ? (lo1 + f) : (hi1 - ng + 1 + f);
            i = lo0 + a;
            k = lo2 + b;
          } else {
            k = (side_sign < 0) ? (lo2 + f) : (hi2 - ng + 1 + f);
            i = lo0 + a;
            j = lo1 + b;
          }
          const int idx = ((v * ng + f) * n1 + a) * n2 + b;
          buffer(idx) = q(v, i, j, k);
        });
  }

  void unpack(PatchField<ExecSpace>& q, int d, int side_sign,
              const buffer_type& buffer) {
    const Box box = decomposition_.local;
    const int ng = ng_;
    const int n1 = n1_[d];
    const int n2 = n2_[d];
    const int nvar = nvar_;
    const int lo0 = box.lo[0];
    const int lo1 = box.lo[1];
    const int lo2 = box.lo[2];
    const int hi0 = box.hi[0];
    const int hi1 = box.hi[1];
    const int hi2 = box.hi[2];

    const auto policy = make_mdrange4<ExecSpace>(
        Int4{{0, 0, 0, 0}}, Int4{{nvar, ng, n1, n2}});
    Kokkos::parallel_for(
        "halo_unpack", policy,
        KOKKOS_LAMBDA(int v, int f, int a, int b) {
          int i = 0;
          int j = 0;
          int k = 0;
          if (d == 0) {
            i = (side_sign < 0) ? (lo0 - ng + f) : (hi0 + 1 + f);
            j = lo1 + a;
            k = lo2 + b;
          } else if (d == 1) {
            j = (side_sign < 0) ? (lo1 - ng + f) : (hi1 + 1 + f);
            i = lo0 + a;
            k = lo2 + b;
          } else {
            k = (side_sign < 0) ? (lo2 - ng + f) : (hi2 + 1 + f);
            i = lo0 + a;
            j = lo1 + b;
          }
          const int idx = ((v * ng + f) * n1 + a) * n2 + b;
          q(v, i, j, k) = buffer(idx);
        });
  }

 private:
  Decomposition decomposition_{};
  int nvar_ = 1;
  int ng_ = 1;
  int n1_[3]{1, 1, 1};
  int n2_[3]{1, 1, 1};
  int size_[3]{0, 0, 0};
  buffer_type send_[3][2]{};
  buffer_type recv_[3][2]{};
  host_buffer_type host_send_[3][2]{};
  host_buffer_type host_recv_[3][2]{};
};

}  // namespace saturn
