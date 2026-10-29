// Variable-coefficient Poisson with Neumann walls: manufactured solution.

#include <Kokkos_Core.hpp>

#ifdef SATURN_HAVE_FFTW

#include "numerics/twophase2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr double kPi = 3.14159265358979323846;
double beta(double x, double y) {
  return 1.0 + 0.25 * std::sin(kPi * x) * std::sin(kPi * y);
}
}  // namespace

int main() {
  {
    // Sanity: constant coefficient, periodic sine.
    const int m = 32;
    const double hm = 1.0 / m;
    std::vector<double> bx0(static_cast<std::size_t>(m) * m, 1.0);
    std::vector<double> by0(static_cast<std::size_t>(m) * m, 1.0);
    std::vector<double> f0(static_cast<std::size_t>(m) * m);
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        const double x = (i + 0.5) * hm;
        const double y = (j + 0.5) * hm;
        f0[static_cast<std::size_t>(j) * m + i] = -8.0 * kPi * kPi *
                                                  std::sin(2.0 * kPi * x) *
                                                  std::sin(2.0 * kPi * y);
      }
    }
    std::vector<double> p0;
    saturn::twophase::VariablePoisson2D solver0(m, m, hm, hm, bx0, by0);
    solver0.solve(f0, p0, 1.0e-12, 2000);
    double e0 = 0.0;
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        const double x = (i + 0.5) * hm;
        const double y = (j + 0.5) * hm;
        const double exact = std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
        e0 = std::max(
            e0, std::abs(p0[static_cast<std::size_t>(j) * m + i] - exact));
      }
    }
    std::printf("periodic PCG sanity: Linf=%.3e\n", e0);
  }

  {
    // Constant coefficient, Neumann walls, p=cos(pi x)cos(pi y).
    const int m = 32;
    const double hm = 1.0 / m;
    std::vector<double> bx0(static_cast<std::size_t>(m) * m, 1.0);
    std::vector<double> by0(static_cast<std::size_t>(m) * m, 1.0);
    std::vector<double> f0(static_cast<std::size_t>(m) * m);
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        const double x = (i + 0.5) * hm;
        const double y = (j + 0.5) * hm;
        f0[static_cast<std::size_t>(j) * m + i] =
            -2.0 * kPi * kPi * std::cos(kPi * x) * std::cos(kPi * y);
      }
    }
    std::vector<double> p0;
    saturn::twophase::VariablePoisson2D solver0(m, m, hm, hm, bx0, by0, false,
                                                false);
    solver0.solve(f0, p0, 1.0e-12, 2000);
    double e0 = 0.0;
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        const double x = (i + 0.5) * hm;
        const double y = (j + 0.5) * hm;
        const double exact = std::cos(kPi * x) * std::cos(kPi * y);
        e0 = std::max(
            e0, std::abs(p0[static_cast<std::size_t>(j) * m + i] - exact));
      }
    }
    std::printf("Neumann constant PCG: Linf=%.3e\n", e0);
  }

  const int n = 64;
  const double h = 1.0 / n;
  std::vector<double> bx(static_cast<std::size_t>(n) * n);
  std::vector<double> by(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double xf = i * h;
      const double yc = (j + 0.5) * h;
      const double xc = (i + 0.5) * h;
      const double yf = j * h;
      bx[static_cast<std::size_t>(j) * n + i] = beta(xf, yc);
      by[static_cast<std::size_t>(j) * n + i] = beta(xc, yf);
    }
  }
  std::vector<double> f(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) * h;
      const double y = (j + 0.5) * h;
      const double sx = std::sin(kPi * x);
      const double cx = std::cos(kPi * x);
      const double sy = std::sin(kPi * y);
      const double cy = std::cos(kPi * y);
      f[static_cast<std::size_t>(j) * n + i] =
          -2.0 * kPi * kPi * cx * cy * (1.0 + 0.25 * sx * sy) -
          0.5 * kPi * kPi * sx * cx * sy * cy;
    }
  }
  std::vector<double> p;
  saturn::twophase::VariablePoisson2D solver(n, n, h, h, bx, by, false, false);
  {
    std::vector<double> pexact(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) * h;
        const double y = (j + 0.5) * h;
        pexact[static_cast<std::size_t>(j) * n + i] =
            std::cos(kPi * x) * std::cos(kPi * y);
      }
    }
    std::vector<double> ap(pexact.size(), 0.0);
    solver.apply(pexact, ap);
    double op_error = 0.0;
    for (std::size_t q = 0; q < ap.size(); ++q) {
      op_error = std::max(op_error, std::abs(ap[q] - f[q]));
    }
    std::printf("operator(p_exact)-f: Linf=%.3e\n", op_error);
  }
  solver.solve(f, p, 1.0e-12, 2000);
  std::vector<double> ap(p.size(), 0.0);
  solver.apply(p, ap);
  double residual = 0.0;
  double pmax = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    residual += (ap[i] - f[i]) * (ap[i] - f[i]);
    pmax = std::max(pmax, std::abs(p[i]));
  }
  std::printf("residual=%.3e |p|max=%.3e\n",
              std::sqrt(residual / static_cast<double>(p.size())), pmax);
  double sum = 0.0;
  double max_error = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) * h;
      const double y = (j + 0.5) * h;
      const double exact = std::cos(kPi * x) * std::cos(kPi * y);
      const double error = p[static_cast<std::size_t>(j) * n + i] - exact;
      sum += error * error;
      max_error = std::max(max_error, std::abs(error));
    }
  }
  const double l2 = std::sqrt(sum / static_cast<double>(n) / n);
  std::printf("variable Neumann Poisson: L2=%.3e Linf=%.3e\n", l2, max_error);
  const bool ok = l2 < 1.0e-3 && max_error < 5.0e-3;
  std::printf("variable Neumann Poisson test %s\n", ok ? "passed" : "failed");
  return ok ? 0 : 1;
}

#else
int main() {
  std::printf("variable Neumann Poisson test skipped (FFTW3 unavailable)\n");
  return 0;
}
#endif
