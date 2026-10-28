// MAC projection test on a periodic box.
//
// 1. Project a smooth field with non-zero divergence and check that the
//    discrete divergence drops to solver accuracy.
// 2. Start from u = grad(psi); the projection must remove it (u_new ~ 0),
//    which verifies the discrete divergence/gradient adjoint pair.

#include <Kokkos_Core.hpp>

#include "numerics/poisson.hpp"
#include "numerics/projection.hpp"

#include <array>
#include <cmath>
#include <cstdio>

namespace {

constexpr double kPi = 3.14159265358979323846;

using exec_space = Kokkos::DefaultExecutionSpace;
using view_type = saturn::projection::ScalarView<exec_space>;

KOKKOS_INLINE_FUNCTION
double psi(double x, double y, double z) {
  return Kokkos::sin(2.0 * kPi * x) * Kokkos::sin(2.0 * kPi * y) *
         Kokkos::sin(2.0 * kPi * z);
}

double max_abs_divergence(const view_type& u, const view_type& v,
                          const view_type& w, const std::array<double, 3>& h) {
  view_type div("div", u.extent(0), u.extent(1), u.extent(2));
  saturn::projection::divergence<exec_space>(u, v, w, div, h);
  double maximum = 0.0;
  Kokkos::parallel_reduce(
      "max_div",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
          {0, 0, 0}, {u.extent(0), u.extent(1), u.extent(2)}),
      KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
        const double value = Kokkos::abs(div(k, j, i));
        acc = value > acc ? value : acc;
      },
      Kokkos::Max<double>{maximum});
  return maximum;
}

double max_abs(const view_type& a) {
  double maximum = 0.0;
  Kokkos::parallel_reduce(
      "max_abs",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
          {0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
      KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
        const double value = Kokkos::abs(a(k, j, i));
        acc = value > acc ? value : acc;
      },
      Kokkos::Max<double>{maximum});
  return maximum;
}

int check(int n) {
  const double h = 1.0 / static_cast<double>(n);
  const std::array<double, 3> spacing{{h, h, h}};
  saturn::poisson::GeometricMultigrid<exec_space> solver(
      std::array<int, 3>{{n, n, n}}, spacing);

  view_type u("u", n, n, n);
  view_type v("v", n, n, n);
  view_type w("w", n, n, n);
  view_type phi("phi", n, n, n);
  view_type scratch("scratch", n, n, n);

  saturn::poisson::Options options;
  options.max_cycles = 200;
  options.tolerance = 1.0e-10;

  // Smooth, divergent test field.
  Kokkos::parallel_for(
      "fill_divergent",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>({0, 0, 0}, {n, n, n}),
      KOKKOS_LAMBDA(int k, int j, int i) {
        const double x = (i + 0.5) * h;
        const double y = (j + 0.5) * h;
        const double z = (k + 0.5) * h;
        u(k, j, i) = Kokkos::sin(2.0 * kPi * y) * Kokkos::sin(2.0 * kPi * z) +
                     0.5 * Kokkos::cos(2.0 * kPi * x);
        v(k, j, i) = Kokkos::cos(2.0 * kPi * x) * Kokkos::sin(2.0 * kPi * z);
        w(k, j, i) = Kokkos::sin(2.0 * kPi * x) * Kokkos::sin(2.0 * kPi * y);
      });

  const double before = max_abs_divergence(u, v, w, spacing);
  auto result = saturn::projection::project<exec_space>(
      u, v, w, phi, scratch, spacing, 1.0e-3, solver, options);
  const double after = max_abs_divergence(u, v, w, spacing);

  // Pure gradient field: projection should cancel it.
  Kokkos::parallel_for(
      "fill_gradient",
      Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>({0, 0, 0}, {n, n, n}),
      KOKKOS_LAMBDA(int k, int j, int i) {
        const int im = (i == 0) ? n - 1 : i - 1;
        const int jm = (j == 0) ? n - 1 : j - 1;
        const int km = (k == 0) ? n - 1 : k - 1;
        const double x = (i + 0.5) * h;
        const double y = (j + 0.5) * h;
        const double z = (k + 0.5) * h;
        const double xm = (im + 0.5) * h;
        const double ym = (jm + 0.5) * h;
        const double zm = (km + 0.5) * h;
        u(k, j, i) = (psi(x, y, z) - psi(xm, y, z)) / h;
        v(k, j, i) = (psi(x, y, z) - psi(x, ym, z)) / h;
        w(k, j, i) = (psi(x, y, z) - psi(x, y, zm)) / h;
      });

  auto result_gradient = saturn::projection::project<exec_space>(
      u, v, w, phi, scratch, spacing, 1.0e-3, solver, options);
  const double gradient_left = max_abs(u);
  const double gradient_left_v = max_abs(v);
  const double gradient_left_w = max_abs(w);

  std::printf(
      "n=%d div=%.3e -> %.3e (cycles %d)  |grad| left %.3e %.3e %.3e "
      "(cycles %d)\n",
      n, before, after, result.cycles, gradient_left, gradient_left_v,
      gradient_left_w, result_gradient.cycles);

  int status = 0;
  if (!result.converged || !result_gradient.converged) {
    std::fprintf(stderr, "n=%d: Poisson solve did not converge\n", n);
    status = 1;
  }
  if (after > 1.0e-7 || before < 1.0e-2) {
    std::fprintf(stderr, "n=%d: divergence projection failed\n", n);
    status = 1;
  }
  if (gradient_left > 1.0e-6 || gradient_left_v > 1.0e-6 ||
      gradient_left_w > 1.0e-6) {
    std::fprintf(stderr, "n=%d: gradient field was not removed\n", n);
    status = 1;
  }
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    status |= check(16);
    status |= check(32);
    std::printf("MAC projection test %s\n", status == 0 ? "passed" : "failed");
  }
  Kokkos::finalize();
  return status;
}
