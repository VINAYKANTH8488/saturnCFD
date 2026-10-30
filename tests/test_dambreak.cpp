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
bool check_static_drop(const char* label, double mu_l, double mu_g) {
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
    solver.step(1.0e-3, 1000.0, 1.0, mu_l, mu_g, 0.0, true, 0.072);
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
      "%s: |u|max=%.3e mass_drift=%.3e range=[%.3e,%.3e] "
      "dp=%.6f expected=%.6f\n",
      label, speed, drift, cmin, cmax, pressure_jump, expected_jump);
  return speed < 5.0e-3 && drift < 1.0e-8 && bounded &&
         std::abs(pressure_jump / expected_jump - 1.0) < 0.02;
}

// Buoyant rise with a viscosity contrast: a light bubble in a heavier, more
// viscous liquid must rise while mass stays conserved and the field bounded.
// Parameters follow the Hysing et al. case-1 setup; this is a stability and
// qualitative check, not yet a quantitative benchmark comparison.
bool check_rising_bubble() {
  const int nx = 64;
  const int ny = 128;
  const double h = 1.0 / nx;
  saturn::twophase::DamBreak2D solver(nx, ny, 1.0, 2.0);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double x = (i + 0.5) * h;
      const double y = (j + 0.5) * h;
      const double d = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
      solver.c()[static_cast<std::size_t>(j) * nx + i] =
          std::min(1.0, std::max(0.0, 0.5 + (0.25 - d) / h));
    }
  }
  const double mass0 = solver.mass();
  const auto centroid = [&]() {
    double sum = 0.0;
    double weight = 0.0;
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const double c = solver.c()[static_cast<std::size_t>(j) * nx + i];
        sum += c * (j + 0.5) * h;
        weight += c;
      }
    }
    return sum / weight;
  };
  const double y0 = centroid();
  for (int step = 0; step < 300; ++step) {
    solver.step(1.0e-3, 100.0, 1000.0, 1.0, 10.0, -0.98, true, 24.5);
  }
  const double y1 = centroid();
  double cmin = 0.0;
  double cmax = 0.0;
  const bool bounded = bounds_ok(solver.c(), cmin, cmax);
  const double drift = std::abs(solver.mass() - mass0) / mass0;
  const double speed = solver.max_speed();
  std::printf("rising bubble: rise=%.4f |u|max=%.3e mass_drift=%.3e\n", y1 - y0,
              speed, drift);
  return y1 - y0 > 0.01 && speed < 1.0 && drift < 1.0e-8 && bounded &&
         std::isfinite(speed);
}

// Martin & Moyce (1952) dry-bed dam break: the surge front Z = x/a must reach
// the reference positions at the published dimensionless times
// T = t sqrt(2g/a) within 15%. The experiment has a finite gate-release time,
// so an inviscid simulation is expected to be a little early.
bool check_martin_moyce() {
  const int nx = 128;
  const int ny = 64;
  const double lx = 0.8;
  const double ly = 0.4;
  const double a = 0.1;
  const double hx = lx / nx;
  const double hy = ly / ny;
  saturn::twophase::DamBreak2D solver(nx, ny, lx, ly);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double x = (i + 0.5) * hx;
      const double y = (j + 0.5) * hy;
      solver.c()[static_cast<std::size_t>(j) * nx + i] =
          (x < a && y < 2.0 * a) ? 1.0 : 0.0;
    }
  }
  const double mass0 = solver.mass();
  const double targets[3] = {2.11, 3.00, 3.89};
  const double reference[3] = {1.83, 2.51, 3.11};
  double arrival[3] = {-1.0, -1.0, -1.0};
  for (int step = 0; step < 300; ++step) {
    solver.step(1.0e-3, 1000.0, 1.0, 1.0e-3, 1.0e-3, -9.81, true, 0.0);
    double front = 0.0;
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < nx; ++i) {
        if (solver.c()[static_cast<std::size_t>(j) * nx + i] > 0.5) {
          front = std::max(front, (i + 0.5) * hx);
        }
      }
    }
    const double z = front / a;
    const double t = (step + 1) * 1.0e-3;
    const double dimensionless = t * std::sqrt(2.0 * 9.81 / a);
    for (int q = 0; q < 3; ++q) {
      if (arrival[q] < 0.0 && z >= targets[q]) {
        arrival[q] = dimensionless;
      }
    }
  }
  double cmin = 0.0;
  double cmax = 0.0;
  const bool bounded = bounds_ok(solver.c(), cmin, cmax);
  const double drift = std::abs(solver.mass() - mass0) / mass0;
  std::printf(
      "Martin-Moyce: T=%.2f/%.2f/%.2f (ref %.2f/%.2f/%.2f) "
      "drift=%.2e\n",
      arrival[0], arrival[1], arrival[2], reference[0], reference[1],
      reference[2], drift);
  bool ok = bounded && drift < 1.0e-8 && std::isfinite(solver.max_speed());
  for (int q = 0; q < 3; ++q) {
    if (arrival[q] < 0.0 || std::abs(arrival[q] / reference[q] - 1.0) > 0.15) {
      ok = false;
    }
  }
  return ok;
}

// Rayleigh-Taylor linear growth at Atwood 1/3: the centre-column vertical
// velocity should grow as exp(sigma t) with sigma = sqrt(A g k) to within the
// accuracy of the face-staggered conservative momentum flux at n=64.
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
  ok = check_static_drop("static drop", 1.0e-3, 1.0e-3) && ok;
  ok = check_static_drop("static drop mu 10:1", 1.0e-3, 1.0e-2) && ok;
  ok = check_rising_bubble() && ok;
  ok = check_martin_moyce() && ok;
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
