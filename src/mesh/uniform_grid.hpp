#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <string>

#include "mesh/box.hpp"

namespace saturn {

struct UniformGrid {
  Box global;
  Box local;
  std::array<double, 3> lo{{0.0, 0.0, 0.0}};
  std::array<double, 3> hi{{1.0, 1.0, 1.0}};
  std::array<double, 3> h{{1.0, 1.0, 1.0}};
  int ng = 2;

  static UniformGrid from_cells(int nx, int ny, int nz, double lx, double ly,
                                double lz, int ng) {
    UniformGrid grid;
    grid.global = Box::from_extents(nx, ny, nz);
    grid.local = grid.global;
    grid.lo = {0.0, 0.0, 0.0};
    grid.hi = {lx, ly, lz};
    grid.h = {lx / nx, ly / ny, lz / nz};
    grid.ng = ng;
    return grid;
  }

  static UniformGrid from_cells_local(int nx, int ny, int nz, double lx,
                                      double ly, double lz, int ng,
                                      const Box& local) {
    UniformGrid grid = from_cells(nx, ny, nz, lx, ly, lz, ng);
    grid.local = local;
    return grid;
  }

  Box local_box() const { return local; }

  KOKKOS_INLINE_FUNCTION
  long long num_cells() const { return global.num_cells(); }
};

}  // namespace saturn
