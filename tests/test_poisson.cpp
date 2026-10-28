// Periodic cell-centered Poisson solver: manufactured-solution convergence.
//
// laplacian(p) = f with p = sin(2 pi x) sin(2 pi y) sin(2 pi z) and
// f = -3 (2 pi)^2 p. The 7-point Laplacian is second-order accurate, so the
// discrete solution error must decrease as h^2 once the multigrid residual is
// driven below the discretization error.

#include <Kokkos_Core.hpp>

#include "numerics/poisson.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

KOKKOS_INLINE_FUNCTION
double exact(double x, double y, double z) {
  return Kokkos::sin(2.0 * kPi * x) * Kokkos::sin(2.0 * kPi * y) *
         Kokkos::sin(2.0 * kPi * z);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using view_type =
        saturn::poisson::GeometricMultigrid<exec_space>::view_type;

    const std::vector<int> resolutions{16, 32, 48};
    double previous_error = 0.0;
    double previous_h = 0.0;
    double last_order = 0.0;
    bool all_converged = true;

    std::printf("periodic Poisson multigrid\n");
    std::printf("%6s %8s %8s %14s %14s %14s %8s\n", "N", "levels", "cycles",
                "residual0", "residual", "L2 error", "order");

    for (const int n : resolutions) {
      const double h = 1.0 / static_cast<double>(n);
      saturn::poisson::GeometricMultigrid<exec_space> solver(
          std::array<int, 3>{{n, n, n}}, std::array<double, 3>{{h, h, h}});

      view_type b("b", n, n, n);
      view_type x("x", n, n, n);
      const double source = -3.0 * (2.0 * kPi) * (2.0 * kPi);
      Kokkos::parallel_for(
          "fill_rhs",
          Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>({0, 0, 0},
                                                             {n, n, n}),
          KOKKOS_LAMBDA(int k, int j, int i) {
            const double px = (i + 0.5) * h;
            const double py = (j + 0.5) * h;
            const double pz = (k + 0.5) * h;
            b(k, j, i) = source * exact(px, py, pz);
            x(k, j, i) = 0.0;
          });

      saturn::poisson::Options options;
      options.max_cycles = 200;
      options.tolerance = 1.0e-10;
      const auto result = solver.solve(x, b, options);
      all_converged = all_converged && result.converged;

      auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
          for (int i = 0; i < n; ++i) {
            const double px = (i + 0.5) * h;
            const double py = (j + 0.5) * h;
            const double pz = (k + 0.5) * h;
            const double error = hx(k, j, i) - exact(px, py, pz);
            sum += error * error;
          }
        }
      }
      const double l2 = std::sqrt(sum / static_cast<double>(n) / n / n);
      const double order =
          previous_error > 0.0
              ? std::log(previous_error / l2) / std::log(previous_h / h)
              : 0.0;
      previous_error = l2;
      previous_h = h;
      last_order = order;

      std::printf("%6d %8d %8d %14.3e %14.3e %14.3e %8.2f\n", n,
                  solver.levels(), result.cycles, result.initial_residual,
                  result.final_residual, l2, order);
    }

    if (!all_converged) {
      std::fprintf(stderr, "Poisson solver failed to converge\n");
      status = 1;
    }
    if (last_order < 1.8) {
      std::fprintf(stderr, "Poisson convergence order %.2f is below 1.8\n",
                   last_order);
      status = 1;
    }
    std::printf("periodic Poisson test %s\n",
                status == 0 ? "passed" : "failed");
  }
  Kokkos::finalize();
  return status;
}
