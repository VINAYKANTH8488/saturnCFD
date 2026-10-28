// Full 3D halo-exchange test: every ghost cell, including edges and corners,
// must match the periodic global value.

#include <Kokkos_Core.hpp>

#include "core/mpi.hpp"
#include "fields/patch_field.hpp"
#include "halo/box_exchange.hpp"
#include "mesh/decomposition.hpp"
#include "mesh/uniform_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

KOKKOS_INLINE_FUNCTION
int wrap(int i, int n) { return ((i % n) + n) % n; }

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
    const int ng = 4;
    const bool periodic[3] = {true, true, true};
    const auto decomposition = saturn::Decomposition::create(
        saturn::Box::from_extents(n, n, n), periodic);
    const auto grid = saturn::UniformGrid::from_cells_local(
        n, n, n, 1.0, 1.0, 1.0, ng, decomposition.local);
    (void)grid;

    using exec_space = Kokkos::DefaultExecutionSpace;
    saturn::PatchField<exec_space> q(decomposition.local, 1, ng, "q");

    auto host_q = Kokkos::create_mirror_view(q.data());
    for (int k = decomposition.local.lo[2]; k <= decomposition.local.hi[2];
         ++k) {
      for (int j = decomposition.local.lo[1]; j <= decomposition.local.hi[1];
           ++j) {
        for (int i = decomposition.local.lo[0]; i <= decomposition.local.hi[0];
             ++i) {
          host_q(0, k - decomposition.local.lo[2] + ng,
                 j - decomposition.local.lo[1] + ng,
                 i - decomposition.local.lo[0] + ng) = value(i, j, k);
        }
      }
    }
    Kokkos::deep_copy(q.data(), host_q);

    saturn::BoxExchange<exec_space> exchange(decomposition, 1, ng);
    exchange.exchange(q);

    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    q.data());
    int local_errors = 0;
    double max_error = 0.0;
    for (int kp = 0; kp < decomposition.local.length(2) + 2 * ng; ++kp) {
      for (int jp = 0; jp < decomposition.local.length(1) + 2 * ng; ++jp) {
        for (int ip = 0; ip < decomposition.local.length(0) + 2 * ng; ++ip) {
          const int i = decomposition.local.lo[0] + ip - ng;
          const int j = decomposition.local.lo[1] + jp - ng;
          const int k = decomposition.local.lo[2] + kp - ng;
          const double expected =
              value(wrap(i, n), wrap(j, n), wrap(k, n));
          const double error = std::fabs(host(0, kp, jp, ip) - expected);
          max_error = std::max(max_error, error);
          if (error > 1.0e-12) {
            if (local_errors < 5) {
              std::printf("rank %d ghost (%d,%d,%d) actual %.3f expected "
                          "%.3f\n",
                          decomposition.rank, i, j, k,
                          host(0, kp, jp, ip), expected);
            }
            ++local_errors;
          }
        }
      }
    }

    const int errors = static_cast<int>(saturn::mpi_sum(local_errors));
    const double global_error = saturn::mpi_max(max_error);
    if (decomposition.rank == 0) {
      std::printf("box exchange: ranks=%d ng=%d errors=%d max_error=%.3e\n",
                  decomposition.size, ng, errors, global_error);
    }
    status = errors == 0 ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef SATURN_ENABLE_MPI
  MPI_Finalize();
#endif
  return status;
}
