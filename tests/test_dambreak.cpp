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
  double pressure_in = 0.0;
  double pressure_out = 0.0;
  int inside = 0;
  int outside = 0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t q = static_cast<std::size_t>(j) * n + i;
      if (solver.c()[q] > 0.9) {
        pressure_in += solver.p()[q];
        ++inside;
      } else if (solver.c()[q] < 0.1) {
        pressure_out += solver.p()[q];
        ++outside;
      }
    }
  }
  pressure_in /= inside;
  pressure_out /= outside;
  const double pressure_jump = pressure_in - pressure_out;
  const double expected_jump = 0.072 / 0.2;
  std::printf(
      "static drop: |u|max=%.3e mass_drift=%.3e range=[%.3e,%.3e] "
      "dp=%.6f expected=%.6f\n",
      speed, drift, cmin, cmax, pressure_jump, expected_jump);
  return speed < 5.0e-3 && drift < 1.0e-8 && bounded &&
         std::abs(pressure_jump / expected_jump - 1.0) < 0.02;
}

// Rayleigh-Taylor linear growth at Atwood 1/3: the centre-column vertical
// velocity should grow as exp(sigma t) with sigma = sqrt(A g k) to within the
// accuracy of the first-order upwind momentum flux at n=64.
bool check_rti() {
  const int n = 64;
  const double h = 1.0 / n;
  const double amplitude = 1.0e-3;
  const double k = 4.0 * 3.141592653589793;
  const double atwood = 1.0 / 3.0;
  saturn::twophase::DamBreak2D solver(n, n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) * h;
      const double y = (j + 0.5) * h;
      const double interface_y = 0.5 + amplitude * std::cos(k * x);
      solver.c()[static_cast<std::size_t>(j) * n + i] =
          std::min(1.0, std::max(0.0, 0.5 + (y - interface_y) / h));
    }
  }
  const double mass0 = solver.mass();
  const int centre = n / 2;
  double v_early = 0.0;
  double v_late = 0.0;
  for (int step = 0; step < 400; ++step) {
    solver.step(1.0e-3, 2.0, 1.0, 1.0e-3, -9.81);
    if (step + 1 == 200) {
      v_early = solver.v()[static_cast<std::size_t>(centre) * n + centre];
    } else if (step + 1 == 400) {
      v_late = solver.v()[static_cast<std::size_t>(centre) * n + centre];
    }
  }
  double cmin = 0.0;
  double cmax = 0.0;
  const bool bounded = bounds_ok(solver.c(), cmin, cmax);
  const double drift = std::abs(solver.mass() - mass0) / mass0;
  const double measured = std::log(std::abs(v_late / v_early)) / 0.2;
  const double theory = std::sqrt(atwood * 9.81 * k);
  std::printf("RTI: sigma=%.4f theory=%.4f mass_drift=%.3e\n", measured, theory,
              drift);
  const bool growth_ok = std::abs(measured / theory - 1.0) < 0.15;
  return growth_ok && drift < 1.0e-8 && bounded &&
         std::isfinite(solver.max_speed());
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
  ok = check_rti() && ok;
  std::printf("dam break test %s\n", ok ? "passed" : "failed");
  return ok ? 0 : 1;
}

#else
int main() {
  std::printf("dam break test skipped (FFTW3 unavailable)\n");
  return 0;
}
#endif
