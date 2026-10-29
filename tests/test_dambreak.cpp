// Wall-bounded two-phase dam-break smoke test.

#include <Kokkos_Core.hpp>

#ifdef SATURN_HAVE_FFTW

#include "numerics/dambreak2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

bool run_case(const char* label, double sigma,
              std::vector<double>* final_c = nullptr) {
  const int n = 64;
  saturn::twophase::DamBreak2D solver(n, n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      solver.c()[static_cast<std::size_t>(j) * n + i] =
          (i < n * 3 / 10) ? 1.0 : 0.0;
    }
  }
  const double mass0 = solver.mass();
  const double dt = 1.0e-3;
  for (int s = 0; s < 50; ++s) {
    solver.step(dt, 1000.0, 1.0, 1.0e-3, -9.81, true, sigma);
  }
  double cmin = 1.0;
  double cmax = 0.0;
  for (double c : solver.c()) {
    cmin = std::min(cmin, c);
    cmax = std::max(cmax, c);
  }
  const double drift = std::abs(solver.mass() - mass0) / mass0;
  const double speed = solver.max_speed();
  std::printf("%s: |u|max=%.3e mass_drift=%.3e range=[%.3e,%.3e]\n", label,
              speed, drift, cmin, cmax);
  if (final_c != nullptr) {
    *final_c = solver.c();
  }
  return std::isfinite(speed) && speed < 1.0e3 && drift < 1.0e-8 &&
         cmin >= -1.0e-10 && cmax <= 1.0 + 1.0e-10;
}

}  // namespace

int main() {
  std::vector<double> plain;
  std::vector<double> csf;
  bool ok = run_case("dam break", 0.0, &plain);
  ok = run_case("dam break + CSF", 0.072, &csf) && ok;
  double difference = 0.0;
  for (std::size_t q = 0; q < plain.size(); ++q) {
    difference = std::max(difference, std::abs(plain[q] - csf[q]));
  }
  std::printf("CSF/plain interface difference: %.3e\n", difference);
  if (difference < 1.0e-8) {
    ok = false;
  }
  std::printf("dam break test %s\n", ok ? "passed" : "failed");
  return ok ? 0 : 1;
}

#else
int main() {
  std::printf("dam break test skipped (FFTW3 unavailable)\n");
  return 0;
}
#endif
