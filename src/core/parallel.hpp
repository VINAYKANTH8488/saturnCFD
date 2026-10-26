#pragma once

#include <Kokkos_Core.hpp>

#include <array>

namespace saturn {

// Inner-dimension tile for 3D MDRangePolicy kernels on device backends.
// Kokkos's default tile was poorly coalesced on the Quadro P600; 32 contiguous
// cells per warp fixed it (5 -> 28 GB/s). Host backends prefer Kokkos's own
// default, so the helper below only applies a tile on device spaces.
inline constexpr int kInnerTile = 32;

template <class ExecSpace>
inline constexpr bool is_host_execution_space =
    Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                               typename ExecSpace::memory_space>::accessible;

template <class ExecSpace>
using MDRange3 = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;

using Int3 = Kokkos::Array<int, 3>;

template <class ExecSpace>
MDRange3<ExecSpace> make_mdrange3(Int3 lower, Int3 upper) {
  if constexpr (is_host_execution_space<ExecSpace>) {
    return MDRange3<ExecSpace>(lower, upper);
  } else {
    return MDRange3<ExecSpace>(lower, upper, Int3{{1, 1, kInnerTile}});
  }
}

}  // namespace saturn
