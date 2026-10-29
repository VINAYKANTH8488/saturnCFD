// Balanced CSF + 2D projection: static bubble Laplace pressure.

#include <Kokkos_Core.hpp>

#ifdef SATURN_HAVE_FFTW

#include "numerics/twophase2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  const int n = 64;
  const double sigma = 0.1;
  const double radius = 0.2;
  saturn::twophase::Solver2D solver(n, n);
  auto& c = solver.c();
  const double h = solver.hx();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double x = (i + 0.5) * solver.hx();
      const double y = (j + 0.5) * solver.hy();
      const double dist =
          std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
      c(i, j) = std::min(1.0, std::max(0.0, 0.5 + (radius - dist) / h));
    }
  }
  std::vector<double> kappa(static_cast<std::size_t>(n) * n, 1.0 / radius);
  std::vector<double> fx;
  std::vector<double> fy;
  solver.csf_faces(kappa, sigma, fx, fy);
  solver.project_force(fx, fy, 1.0e-3);

  const double speed = solver.max_speed();
  const double div = solver.max_divergence();
  const auto& p = solver.p();
  const double p_inside = p[static_cast<std::size_t>(n / 2) * n + n / 2];
  const double p_outside = p[0];
  const double jump = p_inside - p_outside;
  const double expected = sigma / radius;
  std::printf("static bubble: |u|max=%.3e div=%.3e dp=%.6f expected=%.6f\n",
              speed, div, jump, expected);
  const double mass0 = c.mass();
  solver.advect(1.0e-3);
  const double mass_drift = std::abs(c.mass() - mass0) / mass0;
  int status = 0;
  if (speed > 1.0e-8 || div > 1.0e-8) {
    status = 1;
  }
  if (std::abs(jump - expected) > 0.02) {
    status = 1;
  }
  if (mass_drift > 1.0e-12) {
    status = 1;
  }
  std::printf("two-phase static bubble test %s\n",
              status == 0 ? "passed" : "failed");
  return status;
}

#else

int main() {
  std::printf("two-phase test skipped (FFTW3 not available)\n");
  return 0;
}

#endif
