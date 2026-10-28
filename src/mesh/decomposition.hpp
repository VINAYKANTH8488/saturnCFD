#pragma once

#include "mesh/box.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>

#ifdef SATURN_ENABLE_MPI
#include <mpi.h>
#endif

namespace saturn {

// Cartesian MPI decomposition of a global box.
struct Decomposition {
  int rank = 0;
  int size = 1;
  int dims[3]{1, 1, 1};
  int coords[3]{0, 0, 0};
  bool periodic[3]{true, true, true};
  Box global{};
  Box local{};

#ifdef SATURN_ENABLE_MPI
  MPI_Comm comm = MPI_COMM_NULL;
#endif

  static Decomposition create(const Box& global_box,
                              const bool periodic_in[3]) {
    Decomposition dec;
    dec.global = global_box;
    for (int d = 0; d < 3; ++d) {
      dec.periodic[d] = periodic_in[d];
    }

#ifdef SATURN_ENABLE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &dec.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &dec.size);

    int dims[3] = {0, 0, 0};
    if (dec.size > 1) {
      MPI_Dims_create(dec.size, 3, dims);
      for (int d = 0; d < 3; ++d) {
        dims[d] = std::max(1, std::min(dims[d], global_box.length(d)));
      }
      if (dims[0] * dims[1] * dims[2] != dec.size) {
        // The grid is too thin for the MPI_Dims_create factorization; fall
        // back to a greedy factorization that respects each dimension.
        int remaining = dec.size;
        for (int d = 0; d < 3; ++d) {
          int f = std::min(remaining, global_box.length(d));
          while (f > 1 && remaining % f != 0) {
            --f;
          }
          dims[d] = f;
          remaining /= f;
        }
        if (remaining != 1) {
          std::fprintf(stderr,
                       "too many MPI ranks (%d) for grid %d x %d x %d\n",
                       dec.size, global_box.length(0), global_box.length(1),
                       global_box.length(2));
#ifdef SATURN_ENABLE_MPI
          MPI_Abort(MPI_COMM_WORLD, 2);
#else
          std::exit(2);
#endif
        }
      }
    } else {
      dims[0] = dims[1] = dims[2] = 1;
    }
    for (int d = 0; d < 3; ++d) {
      dec.dims[d] = dims[d];
    }

    int periods[3] = {dec.periodic[0] ? 1 : 0, dec.periodic[1] ? 1 : 0,
                      dec.periodic[2] ? 1 : 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dec.dims, periods, 0, &dec.comm);
    MPI_Cart_coords(dec.comm, dec.rank, 3, dec.coords);
#else
    dec.dims[0] = dec.dims[1] = dec.dims[2] = 1;
    dec.coords[0] = dec.coords[1] = dec.coords[2] = 0;
#endif

    for (int d = 0; d < 3; ++d) {
      const int n = global_box.length(d);
      const int base = n / dec.dims[d];
      const int rem = n % dec.dims[d];
      const int start = dec.coords[d] * base + std::min(dec.coords[d], rem);
      const int len = base + (dec.coords[d] < rem ? 1 : 0);
      dec.local.lo[d] = global_box.lo[d] + start;
      dec.local.hi[d] = dec.local.lo[d] + len - 1;
    }
    return dec;
  }

  // Box owned by the given Cartesian coordinates. The same split rule is
  // used for every rank, so this can reconstruct any rank's interior box.
  Box box_for_coords(const int coords_in[3]) const {
    Box result;
    for (int d = 0; d < 3; ++d) {
      const int n = global.length(d);
      const int base = n / dims[d];
      const int rem = n % dims[d];
      const int start = coords_in[d] * base + std::min(coords_in[d], rem);
      const int len = base + (coords_in[d] < rem ? 1 : 0);
      result.lo[d] = global.lo[d] + start;
      result.hi[d] = result.lo[d] + len - 1;
    }
    return result;
  }

  // Neighbor rank on the given side (-1 = low, +1 = high). Returns
  // MPI_PROC_NULL for a non-periodic physical boundary (or -1 without MPI).
  int partner(int d, int side) const {
#ifdef SATURN_ENABLE_MPI
    int source = MPI_PROC_NULL;
    int dest = MPI_PROC_NULL;
    MPI_Cart_shift(comm, d, side, &source, &dest);
    return source;
#else
    (void)d;
    (void)side;
    return -1;
#endif
  }
};

}  // namespace saturn
