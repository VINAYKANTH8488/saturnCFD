#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <string>

#include "mesh/box.hpp"

namespace saturn {

struct UniformGrid {
  Box global;
  std::array<double, 3> lo{{0.0, 0.0, 0.0}};
  std::array<double, 3> hi{{1.0, 1.0, 1.0}};
  std::array<double, 3> h{{1.0, 1.0, 1.0}};
  int ng = 2;

  static UniformGrid from_cells(int nx, int ny, int nz, double lx, double ly,
                                double lz, int ng) {
    UniformGrid grid;
    grid.global = Box::from_extents(nx, ny, nz);
    grid.lo = {0.0, 0.0, 0.0};
    grid.hi = {lx, ly, lz};
    grid.h = {lx / nx, ly / ny, lz / nz};
    grid.ng = ng;
    return grid;
  }

  // Single-rank local box. The Cartesian MPI decomposition will replace this
  // in the halo-exchange milestone; operators already take a Box.
  Box local_box() const { return global; }

  KOKKOS_INLINE_FUNCTION
  long long num_cells() const { return global.num_cells(); }
};

}  // namespace saturn
