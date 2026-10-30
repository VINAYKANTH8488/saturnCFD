// Bounded redistribution: mass conservation and [0,1] bounds.

#include "numerics/vof/vof_redistribute.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
double perturb(int i, int j, int k) {
  const double s = std::sin(0.7 * i + 0.3 * j + 0.11 * k);
  return 0.5 + 0.55 * s;
}
}  // namespace

int main() {
  int status = 0;
  {
    saturn::vof::Grid2D g(32, 32);
    for (int i = 0; i < g.nx(); ++i) {
      for (int j = 0; j < g.ny(); ++j) {
        g(i, j) = perturb(i, j, 0);
      }
    }
    const double target = g.mass();
    saturn::vof::redistribute(g);
    double cmin = 1.0;
    double cmax = 0.0;
    for (double c : g.data()) {
      cmin = std::min(cmin, c);
      cmax = std::max(cmax, c);
    }
    const double error = std::abs(g.mass() - target) / target;
    std::printf("2D redistribute: range=[%.3e,%.3e] mass_error=%.3e\n", cmin,
                cmax, error);
    if (cmin < -1e-14 || cmax > 1.0 + 1e-14 || error > 1e-12) {
      status = 1;
    }
  }
  {
    saturn::vof::Grid3D g(16, 16, 16);
    for (int i = 0; i < g.nx(); ++i) {
      for (int j = 0; j < g.ny(); ++j) {
        for (int k = 0; k < g.nz(); ++k) {
          g(i, j, k) = perturb(i, j, k);
        }
      }
    }
    const double target = g.mass();
    saturn::vof::redistribute(g);
    double cmin = 1.0;
    double cmax = 0.0;
    for (double c : g.data()) {
      cmin = std::min(cmin, c);
      cmax = std::max(cmax, c);
    }
    const double error = std::abs(g.mass() - target) / target;
    std::printf("3D redistribute: range=[%.3e,%.3e] mass_error=%.3e\n", cmin,
                cmax, error);
    if (cmin < -1e-14 || cmax > 1.0 + 1e-14 || error > 1e-12) {
      status = 1;
    }
  }
  std::printf("VOF redistribute test %s\n", status == 0 ? "passed" : "failed");
  return status;
}
