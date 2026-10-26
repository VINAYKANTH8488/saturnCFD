#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>

namespace saturn {

// A 3D index box with inclusive bounds: [lo, hi].
struct Box {
  std::array<int, 3> lo{{0, 0, 0}};
  std::array<int, 3> hi{{-1, -1, -1}};

  KOKKOS_INLINE_FUNCTION
  bool empty() const {
    return hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2];
  }

  KOKKOS_INLINE_FUNCTION
  int length(int d) const { return hi[d] - lo[d] + 1; }

  KOKKOS_INLINE_FUNCTION
  long long num_cells() const {
    if (empty()) {
      return 0;
    }
    return static_cast<long long>(length(0)) * length(1) * length(2);
  }

  KOKKOS_INLINE_FUNCTION
  bool contains(int i, int j, int k) const {
    return i >= lo[0] && i <= hi[0] && j >= lo[1] && j <= hi[1] &&
           k >= lo[2] && k <= hi[2];
  }

  KOKKOS_INLINE_FUNCTION
  Box grow(int n) const {
    Box result;
    for (int d = 0; d < 3; ++d) {
      result.lo[d] = lo[d] - n;
      result.hi[d] = hi[d] + n;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Box shift(int di, int dj, int dk) const {
    Box result;
    result.lo = {lo[0] + di, lo[1] + dj, lo[2] + dk};
    result.hi = {hi[0] + di, hi[1] + dj, hi[2] + dk};
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Box intersect(const Box& other) const {
    Box result;
    for (int d = 0; d < 3; ++d) {
      result.lo[d] = std::max(lo[d], other.lo[d]);
      result.hi[d] = std::min(hi[d], other.hi[d]);
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const Box& other) const {
    return lo == other.lo && hi == other.hi;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const Box& other) const { return !(*this == other); }

  static Box from_extents(int nx, int ny, int nz) {
    Box result;
    result.lo = {0, 0, 0};
    result.hi = {nx - 1, ny - 1, nz - 1};
    return result;
  }
};

}  // namespace saturn
