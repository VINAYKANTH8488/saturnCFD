// Wall-bounded two-phase dam-break and equilibrium tests.

#include <Kokkos_Core.hpp>

#ifdef SATURN_HAVE_FFTW

#include "numerics/dambreak2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool bounds_ok(const std::vector<double>& c, double& cmin, double& cmax) {
  cmin = 1.0;
  cmax = 0.0;
  for (double value : c) {
    cmin = std::min(cmin, value);
    cmax = std::max(cmax, value);
  }
  return cmin >= -1.0e-10 && cmax <= 1.0 + 1.0e-10;
}

void fill_column(saturn::twophase::DamBreak2D& solver) {
  const int n = solver.nx();
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      solver.c()[static_cast<std::size_t>(j) * n + i] =
          (i < n * 3 / 10) ? 1.0 : 0.0;
    }
  }
}

bool check_case(const char* label, double sigma, std::vector<double>* final_c) {
  const int n = 64;
  saturn::twophase::DamBreak2D solver(n, n);
  fill_column(solver);
  const double mass0 = solver.mass();
  const double dt = 1.0e-3;
  for (int s = 0; s < 50; ++s) {
    solver.step(dt, 1000.0, 1.0, 1.0e-3, -9.81, true, sigma);
  }
  double cmin = 0.0;
  double cmax = 0.0;
  const bool bounded = bounds_ok(solver.c(), cmin, cmax);
  const double drift = std::abs(solver.mass() - mass0) / mass0;
  const double speed = solver.max_speed();
  std::printf("%s: |u|max=%.3e mass_drift=%.3e range=[%.3e,%.3e]\n", label,
              speed, drift, cmin, cmax);
  if (final_c != nullptr) {
    *final_c = solver.c();
  }
  return std::isfinite(speed) && speed < 1.0e3 && drift < 1.0e-8 && bounded;
}

// Two layers at rest must stay at rest: gravity is balanced exactly by the
// wall-bounded variable-coefficient pressure solve.
bool check_hydrostatic() {
  const int n = 64;
  saturn::twophase::DamBreak2D solver(n, n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      solver.c()[static_cast<std::size_t>(j) * n + i] = (j < n / 2) ? 1.0 : 0.0;
    }
  }
  const double mass0 = solver.mass();
  for (int s = 0; s < 200; ++s) {
    solver.step(1.0e-3, 1000.0, 1.0, 1.0e-3, -9.81);
  }
  const double drift = std::abs(solver.mass() - mass0) / mass0;
  const double speed = solver.max_speed();
  std::printf("hydrostatic: |u|max=%.3e mass_drift=%.3e\n", speed, drift);
  return speed < 1.0e-10 && drift < 1.0e-8;
}

// A wall-bounded static drop with surface tension: curvature error leaves a
// small residual current, but it must stay bounded and mass must be conserved.
bool check_static_drop() {
  const int n = 64;
  saturn::twophase::DamBreak2D solver(n, n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n;
      const double y = (j + 0.5) / n;
      const double d = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
      solver.c()[static_cast<std::size_t>(j) * n + i] =
          std::min(1.0, std::max(0.0, 0.5 + (0.2 - d) * n));
    }
  }
  const double mass0 = solver.mass();
  for (int s = 0; s < 200; ++s) {
    solver.step(1.0e-3, 1000.0, 1.0, 1.0e-3, 0.0, true, 0.072);
  }
  double cmin = 0.0;
  double cmax = 0.0;
  const bool bounded = bounds_ok(solver.c(), cmin, cmax);
  const double drift = std::abs(solver.mass() - mass0) / mass0;
  const double speed = solver.max_speed();
  std::printf("static drop: |u|max=%.3e mass_drift=%.3e range=[%.3e,%.3e]\n",
              speed, drift, cmin, cmax);
  return speed < 5.0e-3 && drift < 1.0e-8 && bounded;
}

}  // namespace

int main() {
  std::vector<double> plain;
  std::vector<double> csf;
  bool ok = check_case("dam break", 0.0, &plain);
  ok = check_case("dam break + CSF", 0.072, &csf) && ok;
  double difference = 0.0;
  for (std::size_t q = 0; q < plain.size(); ++q) {
    difference = std::max(difference, std::abs(plain[q] - csf[q]));
  }
  std::printf("CSF/plain interface difference: %.3e\n", difference);
  ok = difference > 1.0e-8 && ok;
  ok = check_hydrostatic() && ok;
  ok = check_static_drop() && ok;
  std::printf("dam break test %s\n", ok ? "passed" : "failed");
  return ok ? 0 : 1;
}

#else
int main() {
  std::printf("dam break test skipped (FFTW3 unavailable)\n");
  return 0;
}
#endif
