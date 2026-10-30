// FFT periodic Poisson solver: manufactured-solution convergence.

#include <Kokkos_Core.hpp>

#include <cstdio>

#ifdef SATURN_HAVE_FFTW

#include "numerics/incompressible/fft_poisson.hpp"

#include <array>
#include <cmath>
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
    using view_type = saturn::poisson::FftPoisson<exec_space>::view_type;

    const std::vector<int> resolutions{16, 32, 48};
    double previous_error = 0.0;
    double previous_h = 0.0;
    double last_order = 0.0;

    std::printf("periodic FFT Poisson\n");
    std::printf("%6s %14s %14s %8s\n", "N", "L2 error", "order", "time [s]");

    for (const int n : resolutions) {
      const double h = 1.0 / static_cast<double>(n);
      saturn::poisson::FftPoisson<exec_space> solver(
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

      Kokkos::Timer timer;
      saturn::poisson::Options options;
      const auto result = solver.solve(x, b, options);
      const double seconds = timer.seconds();

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
      if (!result.converged) {
        status = 1;
      }
      std::printf("%6d %14.4e %14.2f %8.3f\n", n, l2, order, seconds);
    }

    if (last_order < 1.8) {
      std::fprintf(stderr, "FFT Poisson order %.2f below 1.8\n", last_order);
      status = 1;
    }
    std::printf("FFT Poisson test %s\n", status == 0 ? "passed" : "failed");
  }
  Kokkos::finalize();
  return status;
}

#else

int main() {
  std::printf("FFT Poisson test skipped (FFTW3 not available)\n");
  return 0;
}

#endif
