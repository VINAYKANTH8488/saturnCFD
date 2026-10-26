// I/O test: write/read a binary checkpoint and write a VTK snapshot.

#include <Kokkos_Core.hpp>

#include "core/mpi.hpp"
#include "fields/patch_field.hpp"
#include "io/checkpoint.hpp"
#include "io/vtk_writer.hpp"
#include "mesh/decomposition.hpp"
#include "mesh/uniform_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

KOKKOS_INLINE_FUNCTION
double value(int i, int j, int k) {
  return static_cast<double>(i + 10 * j + 100 * k);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef SATURN_ENABLE_MPI
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int n = 16;
    const int ng = 2;
    const bool periodic[3] = {true, true, true};
    const auto dec = saturn::Decomposition::create(
        saturn::Box::from_extents(n, n, n), periodic);
    const auto grid = saturn::UniformGrid::from_cells_local(
        n, n, n, 1.0, 1.0, 1.0, ng, dec.local);

    using exec_space = Kokkos::DefaultExecutionSpace;
    using field_type = saturn::PatchField<exec_space>;

    const std::string base = "saturn_io_test";
    field_type q(dec.local, 1, ng, "q");
    auto hq = Kokkos::create_mirror_view(q.data());
    for (int k = dec.local.lo[2]; k <= dec.local.hi[2]; ++k) {
      for (int j = dec.local.lo[1]; j <= dec.local.hi[1]; ++j) {
        for (int i = dec.local.lo[0]; i <= dec.local.hi[0]; ++i) {
          hq(0, k - dec.local.lo[2] + ng, j - dec.local.lo[1] + ng,
             i - dec.local.lo[0] + ng) = value(i, j, k);
        }
      }
    }
    Kokkos::deep_copy(q.data(), hq);

    bool ok = saturn::write_checkpoint(base, q, 7, 1.25, dec);

    field_type q2(dec.local, 1, ng, "q2");
    Kokkos::deep_copy(q2.data(), 0.0);
    int step = -1;
    double time = -1.0;
    ok = ok && saturn::read_checkpoint(base, q2, step, time, dec);

    double local_error = 0.0;
    auto hq2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   q2.data());
    for (int k = dec.local.lo[2]; k <= dec.local.hi[2]; ++k) {
      for (int j = dec.local.lo[1]; j <= dec.local.hi[1]; ++j) {
        for (int i = dec.local.lo[0]; i <= dec.local.hi[0]; ++i) {
          const double expected = value(i, j, k);
          local_error = std::max(
              local_error,
              std::abs(hq2(0, k - dec.local.lo[2] + ng,
                           j - dec.local.lo[1] + ng,
                           i - dec.local.lo[0] + ng) -
                       expected));
        }
      }
    }
    const double max_error = saturn::mpi_max(local_error);
    ok = ok && (max_error < 1.0e-12) && (step == 7) &&
         (std::abs(time - 1.25) < 1.0e-12);

    const std::array<double, 3> origin{{0.0, 0.0, 0.0}};
    const std::array<double, 3> spacing{
        {grid.h[0], grid.h[1], grid.h[2]}};
    ok = ok && saturn::write_vti(base + "_field", q, origin, spacing,
                                 dec.global, 1.25);

    const std::string vti =
        (dec.size > 1) ? (base + "_field" + saturn::rank_suffix(dec.rank) +
                          ".vti")
                       : (base + "_field.vti");
    std::error_code ec;
    const auto vti_size = std::filesystem::file_size(vti, ec);
    ok = ok && !ec && vti_size > 0;

    if (dec.rank == 0) {
      const std::string pvti = base + "_field.pvti";
      if (dec.size > 1) {
        const auto pvti_size = std::filesystem::file_size(pvti, ec);
        ok = ok && !ec && pvti_size > 0;
      }
      std::printf(
          "io test: ranks=%d step=%d time=%.2f max_error=%.3e vti=%lld bytes\n",
          dec.size, step, time, max_error,
          static_cast<long long>(vti_size));
    }

    const bool global_ok = saturn::mpi_all(ok);
    if (dec.rank == 0) {
      std::printf("io test: %s\n", global_ok ? "passed" : "FAILED");
    }
    status = global_ok ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef SATURN_ENABLE_MPI
  MPI_Finalize();
#endif
  return status;
}
