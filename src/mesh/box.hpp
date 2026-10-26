#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>

namespace saturn {

// A 3D index box with inclusive bounds: [lo, hi].
// Plain C arrays are used instead of std::array so the accessors are safe in
// CUDA device code without relying on --expt-relaxed-constexpr.
struct Box {
  int lo[3]{0, 0, 0};
  int hi[3]{-1, -1, -1};

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
  Box shrink(int n) const { return grow(-n); }

  KOKKOS_INLINE_FUNCTION
  Box shift(int di, int dj, int dk) const {
    Box result;
    result.lo[0] = lo[0] + di;
    result.lo[1] = lo[1] + dj;
    result.lo[2] = lo[2] + dk;
    result.hi[0] = hi[0] + di;
    result.hi[1] = hi[1] + dj;
    result.hi[2] = hi[2] + dk;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Box intersect(const Box& other) const {
    Box result;
    for (int d = 0; d < 3; ++d) {
      result.lo[d] = (lo[d] > other.lo[d]) ? lo[d] : other.lo[d];
      result.hi[d] = (hi[d] < other.hi[d]) ? hi[d] : other.hi[d];
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const Box& other) const {
    return lo[0] == other.lo[0] && lo[1] == other.lo[1] &&
           lo[2] == other.lo[2] && hi[0] == other.hi[0] &&
           hi[1] == other.hi[1] && hi[2] == other.hi[2];
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const Box& other) const { return !(*this == other); }

  static Box from_extents(int nx, int ny, int nz) {
    Box result;
    result.lo[0] = 0;
    result.lo[1] = 0;
    result.lo[2] = 0;
    result.hi[0] = nx - 1;
    result.hi[1] = ny - 1;
    result.hi[2] = nz - 1;
    return result;
  }
};

}  // namespace saturn
