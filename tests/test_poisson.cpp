// Poisson solver convergence on periodic and Neumann boxes.
//
// periodic: p = sin(2pi x) sin(2pi y) sin(2pi z), f = -3 (2pi)^2 p
// Neumann : p = cos(pi x) cos(pi y) cos(pi z),   f = -3 pi^2 p
//
// Both have zero mean (the Neumann cosine has zero integral on [0,1]), so
// the periodic/Neumann null space is compatible.

#include <Kokkos_Core.hpp>

#include "numerics/poisson.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

KOKKOS_INLINE_FUNCTION
double exactValue(int case_id, double x, double y, double z) {
  if (case_id == 1) {
    return Kokkos::cos(kPi * x) * Kokkos::cos(kPi * y) * Kokkos::cos(kPi * z);
  }
  if (case_id == 2) {
    return Kokkos::sin(2.0 * kPi * x) * Kokkos::cos(kPi * y) *
           Kokkos::sin(2.0 * kPi * z);
  }
  return Kokkos::sin(2.0 * kPi * x) * Kokkos::sin(2.0 * kPi * y) *
         Kokkos::sin(2.0 * kPi * z);
}

double exactValueHost(int case_id, double x, double y, double z) {
  if (case_id == 1) {
    return std::cos(kPi * x) * std::cos(kPi * y) * std::cos(kPi * z);
  }
  if (case_id == 2) {
    return std::sin(2.0 * kPi * x) * std::cos(kPi * y) *
           std::sin(2.0 * kPi * z);
  }
  return std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y) *
         std::sin(2.0 * kPi * z);
}

int runConfig(int case_id, const char* name,
              const std::array<saturn::poisson::Boundary, 3>& bc_low,
              const std::array<saturn::poisson::Boundary, 3>& bc_high,
              double source_coefficient) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_type = saturn::poisson::GeometricMultigrid<exec_space>::view_type;

  const std::vector<int> resolutions{16, 32, 48};
  double previous_error = 0.0;
  double previous_h = 0.0;
  double last_order = 0.0;
  bool all_converged = true;

  std::printf("\n%s\n", name);
  std::printf("%6s %8s %8s %14s %14s %14s %8s\n", "N", "levels", "cycles",
              "residual0", "residual", "L2 error", "order");

  for (const int n : resolutions) {
    const double h = 1.0 / static_cast<double>(n);
    saturn::poisson::GeometricMultigrid<exec_space> solver(
        std::array<int, 3>{{n, n, n}}, std::array<double, 3>{{h, h, h}}, bc_low,
        bc_high);

    view_type b("b", n, n, n);
    view_type x("x", n, n, n);
    const double source = source_coefficient;
    Kokkos::parallel_for(
        "fill_rhs",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>({0, 0, 0},
                                                           {n, n, n}),
        KOKKOS_LAMBDA(int k, int j, int i) {
          const double px = (i + 0.5) * h;
          const double py = (j + 0.5) * h;
          const double pz = (k + 0.5) * h;
          b(k, j, i) = source * exactValue(case_id, px, py, pz);
          x(k, j, i) = 0.0;
        });

    saturn::poisson::Options options;
    options.max_cycles = 300;
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
          const double error =
              hx(k, j, i) - exactValueHost(case_id, px, py, pz);
          sum += error * error;
        }
      }
    }
    const double l2 = std::sqrt(sum / static_cast<double>(n) / n / n);
    const double order = previous_error > 0.0 ? std::log(previous_error / l2) /
                                                    std::log(previous_h / h)
                                              : 0.0;
    previous_error = l2;
    previous_h = h;
    last_order = order;

    std::printf("%6d %8d %8d %14.3e %14.3e %14.3e %8.2f\n", n, solver.levels(),
                result.cycles, result.initial_residual, result.final_residual,
                l2, order);
  }

  int status = 0;
  if (!all_converged) {
    std::fprintf(stderr, "%s: solver failed to converge\n", name);
    status = 1;
  }
  if (last_order < 1.8) {
    std::fprintf(stderr, "%s: order %.2f is below 1.8\n", name, last_order);
    status = 1;
  }
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const std::array<saturn::poisson::Boundary, 3> periodic{
        {saturn::poisson::Boundary::Periodic,
         saturn::poisson::Boundary::Periodic,
         saturn::poisson::Boundary::Periodic}};
    const std::array<saturn::poisson::Boundary, 3> neumann{
        {saturn::poisson::Boundary::Neumann, saturn::poisson::Boundary::Neumann,
         saturn::poisson::Boundary::Neumann}};
    const std::array<saturn::poisson::Boundary, 3> mixed_low{
        {saturn::poisson::Boundary::Periodic,
         saturn::poisson::Boundary::Neumann,
         saturn::poisson::Boundary::Periodic}};
    const std::array<saturn::poisson::Boundary, 3> mixed_high{
        {saturn::poisson::Boundary::Periodic,
         saturn::poisson::Boundary::Neumann,
         saturn::poisson::Boundary::Periodic}};

    status |= runConfig(0, "periodic", periodic, periodic,
                        -3.0 * (2.0 * kPi) * (2.0 * kPi));
    status |= runConfig(1, "Neumann walls", neumann, neumann, -3.0 * kPi * kPi);
    status |= runConfig(2, "periodic x/z, Neumann y", mixed_low, mixed_high,
                        -9.0 * kPi * kPi);
    std::printf("\nPoisson test %s\n", status == 0 ? "passed" : "failed");
  }
  Kokkos::finalize();
  return status;
}
