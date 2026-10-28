// Unit tests for the face-reconstruction schemes:
// exactness on polynomials, smooth-flow convergence order, symmetry, and
// non-oscillatory behaviour across a discontinuity.

#include <Kokkos_Core.hpp>

#include "numerics/reconstruction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace {

using saturn::Reconstruction;
using saturn::recon::reconstruct_left;
using saturn::recon::reconstruct_right;

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr Reconstruction kAllSchemes[] = {
    Reconstruction::Central2, Reconstruction::Central4,
    Reconstruction::Central6, Reconstruction::Weno5Z, Reconstruction::Teno6};

int g_failures = 0;

const char* scheme_name(Reconstruction scheme) {
  switch (scheme) {
    case Reconstruction::Central2:
      return "central2";
    case Reconstruction::Central4:
      return "central4";
    case Reconstruction::Central6:
      return "central6";
    case Reconstruction::Weno5Z:
      return "weno5z";
    case Reconstruction::Teno6:
      return "teno6";
  }
  return "unknown";
}

void check_near(const char* what, const char* scheme, double actual,
                double expected, double tolerance) {
  const double scale = std::max(1.0, std::fabs(expected));
  const double error = std::fabs(actual - expected);
  if (!(error <= tolerance * scale)) {
    std::printf("FAIL %s [%s]: actual %.17g expected %.17g error %.3e\n", what,
                scheme, actual, expected, error);
    ++g_failures;
  }
}

std::array<double, 6> make_stencil(double vm2, double vm1, double v0,
                                   double vp1, double vp2, double vp3) {
  return {vm2, vm1, v0, vp1, vp2, vp3};
}

double left_value(Reconstruction scheme, const std::array<double, 6>& v) {
  return reconstruct_left(scheme, v[0], v[1], v[2], v[3], v[4], v[5]);
}

double right_value(Reconstruction scheme, const std::array<double, 6>& v) {
  return reconstruct_right(scheme, v[0], v[1], v[2], v[3], v[4], v[5]);
}

// Cell average of x^degree over [j h, (j+1) h].
double polynomial_average(int j, int degree, double h) {
  const double a = static_cast<double>(j) * h;
  const double b = static_cast<double>(j + 1) * h;
  return (std::pow(b, degree + 1) - std::pow(a, degree + 1)) /
         (static_cast<double>(degree + 1) * h);
}

// Cell average of sin(2 pi x + phase) over cell j, on a unit periodic box.
double sine_average(int j, int n, double phase) {
  const double h = 1.0 / static_cast<double>(n);
  const int wrapped = ((j % n) + n) % n;
  const double a = static_cast<double>(wrapped) * h;
  const double b = static_cast<double>(wrapped + 1) * h;
  return (std::cos(2.0 * kPi * a + phase) -
          std::cos(2.0 * kPi * b + phase)) /
         (2.0 * kPi * h);
}

template <class Cell>
std::array<double, 6> stencil_at(const Cell& cell, int i) {
  return make_stencil(cell(i - 2), cell(i - 1), cell(i), cell(i + 1),
                      cell(i + 2), cell(i + 3));
}

void test_constant() {
  const auto v = make_stencil(2.5, 2.5, 2.5, 2.5, 2.5, 2.5);
  for (const auto scheme : kAllSchemes) {
    check_near("constant left", scheme_name(scheme), left_value(scheme, v),
               2.5, 1.0e-15);
    check_near("constant right", scheme_name(scheme), right_value(scheme, v),
               2.5, 1.0e-15);
  }
}

void test_polynomial_exactness(int degree,
                               std::initializer_list<Reconstruction> schemes) {
  constexpr double h = 0.1;
  const auto cell = [&](int j) {
    return polynomial_average(j, degree, h);
  };
  const int face_index = 3;
  const auto v = stencil_at(cell, face_index);
  const double exact = std::pow(static_cast<double>(face_index + 1) * h, degree);

  char label[64];
  std::snprintf(label, sizeof(label), "degree-%d", degree);
  for (const auto scheme : schemes) {
    check_near(label, scheme_name(scheme), left_value(scheme, v), exact,
               1.0e-12);
    check_near(label, scheme_name(scheme), right_value(scheme, v), exact,
               1.0e-12);
  }
}

double smooth_face_error(Reconstruction scheme, int n) {
  constexpr double phase = 0.3;
  const auto cell = [&](int j) { return sine_average(j, n, phase); };
  const double h = 1.0 / static_cast<double>(n);
  double error = 0.0;
  for (int i = 0; i < n; ++i) {
    const auto v = stencil_at(cell, i);
    const double exact =
        std::sin(2.0 * kPi * static_cast<double>(i + 1) * h + phase);
    error += std::fabs(left_value(scheme, v) - exact) * h;
  }
  return error;
}

void test_convergence() {
  struct Expectation {
    Reconstruction scheme;
    double order;
  };
  const Expectation expectations[] = {
      {Reconstruction::Central2, 2.0}, {Reconstruction::Central4, 4.0},
      {Reconstruction::Central6, 6.0}, {Reconstruction::Weno5Z, 5.0},
      {Reconstruction::Teno6, 6.0}};

  for (const auto& expectation : expectations) {
    const double coarse = smooth_face_error(expectation.scheme, 32);
    const double fine = smooth_face_error(expectation.scheme, 64);
    const double observed = std::log(coarse / fine) / std::log(2.0);
    std::printf("  %-8s L1 error %.3e -> %.3e, order %.2f (expected %.1f)\n",
                scheme_name(expectation.scheme), coarse, fine, observed,
                expectation.order);
    if (observed < expectation.order - 0.2) {
      std::printf("FAIL convergence [%s]: observed order %.2f\n",
                  scheme_name(expectation.scheme), observed);
      ++g_failures;
    }
  }
}

void test_symmetry() {
  // Symmetric cell values around the face: left and right states must agree.
  const auto v = make_stencil(-0.3, 1.7, 4.2, 4.2, 1.7, -0.3);
  for (const auto scheme : kAllSchemes) {
    check_near("symmetry", scheme_name(scheme), left_value(scheme, v),
               right_value(scheme, v), 1.0e-14);
  }
}

void test_shock_bounded() {
  const auto cell = [](int j) { return j < 0 ? 0.0 : 1.0; };
  const Reconstruction bounded[] = {Reconstruction::Weno5Z,
                                    Reconstruction::Teno6};
  for (const auto scheme : bounded) {
    for (int i = -5; i <= 4; ++i) {
      const auto v = stencil_at(cell, i);
      const double left = left_value(scheme, v);
      const double right = right_value(scheme, v);
      if (!(left >= -1.0e-12 && left <= 1.0 + 1.0e-12)) {
        std::printf("FAIL shock left [%s] at face %d: %.17g\n",
                    scheme_name(scheme), i, left);
        ++g_failures;
      }
      if (!(right >= -1.0e-12 && right <= 1.0 + 1.0e-12)) {
        std::printf("FAIL shock right [%s] at face %d: %.17g\n",
                    scheme_name(scheme), i, right);
        ++g_failures;
      }
    }
  }
}

void test_teno6_falls_back_at_shock() {
  // At the face immediately left of the jump the smooth six-point candidate
  // overshoots (central6 gives 0.5 above the exact 0). TENO6 must reject it
  // and return the upwind state.
  const auto v = make_stencil(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
  const double teno = left_value(Reconstruction::Teno6, v);
  check_near("TENO6 upwind at shock", "teno6", teno, 0.0, 1.0e-12);
  const double central = left_value(Reconstruction::Central6, v);
  if (!(central > 0.4)) {
    std::printf("FAIL TENO6 fallback check: central6 unexpectedly %.17g\n",
                central);
    ++g_failures;
  }
}

void test_device_kernel() {
  constexpr int kSchemeCount = 5;
  Kokkos::View<double*> values("reconstruction_values", 6);
  Kokkos::View<double*> output("reconstruction_output", 2 * kSchemeCount);
  auto host_values = Kokkos::create_mirror_view(values);
  host_values(0) = -0.4;
  host_values(1) = 0.9;
  host_values(2) = 1.3;
  host_values(3) = 1.8;
  host_values(4) = 2.4;
  host_values(5) = 3.7;
  Kokkos::deep_copy(values, host_values);

  Kokkos::parallel_for(
      "reconstruction_device", kSchemeCount, KOKKOS_LAMBDA(int index) {
        const auto scheme = static_cast<Reconstruction>(index);
        output(2 * index) =
            reconstruct_left(scheme, values(0), values(1), values(2),
                             values(3), values(4), values(5));
        output(2 * index + 1) =
            reconstruct_right(scheme, values(0), values(1), values(2),
                              values(3), values(4), values(5));
      });

  auto host_output =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
  const std::array<double, 6> v = {-0.4, 0.9, 1.3, 1.8, 2.4, 3.7};
  for (int index = 0; index < kSchemeCount; ++index) {
    const auto scheme = static_cast<Reconstruction>(index);
    check_near("device left", scheme_name(scheme), host_output(2 * index),
               left_value(scheme, v), 1.0e-14);
    check_near("device right", scheme_name(scheme), host_output(2 * index + 1),
               right_value(scheme, v), 1.0e-14);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Reconstruction exactness:\n");
    test_constant();
    test_polynomial_exactness(1, {Reconstruction::Central2,
                                  Reconstruction::Central4,
                                  Reconstruction::Central6,
                                  Reconstruction::Weno5Z,
                                  Reconstruction::Teno6});
    test_polynomial_exactness(2, {Reconstruction::Central4,
                                  Reconstruction::Central6,
                                  Reconstruction::Weno5Z,
                                  Reconstruction::Teno6});
    test_polynomial_exactness(3, {Reconstruction::Central4,
                                  Reconstruction::Central6,
                                  Reconstruction::Teno6});
    test_polynomial_exactness(5, {Reconstruction::Central6,
                                  Reconstruction::Teno6});

    std::printf("Smooth-flow convergence (periodic sine):\n");
    test_convergence();

    test_symmetry();
    test_shock_bounded();
    test_teno6_falls_back_at_shock();
    test_device_kernel();

    if (g_failures == 0) {
      std::printf("saturnCFD reconstruction tests passed\n");
    } else {
      std::printf("saturnCFD reconstruction tests FAILED: %d failure(s)\n",
                  g_failures);
    }
  }
  Kokkos::finalize();
  return g_failures == 0 ? 0 : 1;
}
