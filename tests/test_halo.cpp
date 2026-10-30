// Halo-exchange unit test: fill a periodic field with a global linear index
// function, exchange, and verify every ghost cell against the wrapped value.

#include <Kokkos_Core.hpp>

#include "core/mpi.hpp"
#include "fields/patch_field.hpp"
#include "halo/face_exchange.hpp"
#include "mesh/decomposition.hpp"
#include "mesh/uniform_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    const int ng = 2;
    const bool periodic[3] = {true, true, true};
    const auto dec = saturn::Decomposition::create(
        saturn::Box::fromExtents(n, n, n), periodic);
    const auto grid = saturn::UniformGrid::fromCellsLocal(n, n, n, 1.0, 1.0,
                                                          1.0, ng, dec.local);
    (void)grid;

    using exec_space = Kokkos::DefaultExecutionSpace;
    saturn::PatchField<exec_space> q(dec.local, 1, ng, "q");

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

    saturn::FaceExchange<exec_space> exchange(dec, 1, ng);
    exchange.exchange(q);

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q.data());
    int local_errors = 0;
    int face_errors[3] = {0, 0, 0};
    double max_error = 0.0;
    for (int kp = 0; kp < dec.local.length(2) + 2 * ng; ++kp) {
      for (int jp = 0; jp < dec.local.length(1) + 2 * ng; ++jp) {
        for (int ip = 0; ip < dec.local.length(0) + 2 * ng; ++ip) {
          const int i = dec.local.lo[0] + ip - ng;
          const int j = dec.local.lo[1] + jp - ng;
          const int k = dec.local.lo[2] + kp - ng;
          const int outside_local =
              (i < dec.local.lo[0] || i > dec.local.hi[0] ? 1 : 0) +
              (j < dec.local.lo[1] || j > dec.local.hi[1] ? 1 : 0) +
              (k < dec.local.lo[2] || k > dec.local.hi[2] ? 1 : 0);
          // Face-only exchange: edges and corners are intentionally not
          // filled (the 7-point stencil never reads them).
          if (outside_local != 1) {
            continue;
          }
          int face = 0;
          if (i < dec.local.lo[0] || i > dec.local.hi[0]) {
            face = 0;
          } else if (j < dec.local.lo[1] || j > dec.local.hi[1]) {
            face = 1;
          } else {
            face = 2;
          }
          const double expected = value(wrap(i, n), wrap(j, n), wrap(k, n));
          const double error = std::abs(h(0, kp, jp, ip) - expected);
          if (error > 1.0e-12) {
            if (face_errors[face] < 3) {
              std::printf(
                  "rank %d face %d: ghost (%d,%d,%d) actual %.1f expected "
                  "%.1f\n",
                  dec.rank, face, i, j, k, h(0, kp, jp, ip), expected);
            }
            ++face_errors[face];
            ++local_errors;
            max_error = std::max(max_error, error);
          }
        }
      }
    }

    int global_errors = 0;
    double global_max = 0.0;
#ifdef SATURN_ENABLE_MPI
    MPI_Allreduce(&local_errors, &global_errors, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&max_error, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
#else
    global_errors = local_errors;
    global_max = max_error;
#endif

    if (dec.rank == 0) {
      std::printf(
          "halo test: ranks=%d local=%dx%dx%d errors=%d max_error=%.3e\n",
          dec.size, dec.local.length(0), dec.local.length(1),
          dec.local.length(2), global_errors, global_max);
    }
    status = (global_errors == 0) ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef SATURN_ENABLE_MPI
  MPI_Finalize();
#endif
  return status;
}
