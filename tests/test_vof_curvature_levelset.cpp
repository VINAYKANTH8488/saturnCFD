// Level-set curvature on a circular interface.

#include "numerics/vof_curvature_levelset2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

int main() {
  const int n = 96;
  const double radius = 0.2;
  saturn::vof::Grid2D g(n, n);
  const double h = g.hx();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double x = (i + 0.5) * g.hx();
      const double y = (j + 0.5) * g.hy();
      const double dist =
          std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
      g(i, j) = std::min(1.0, std::max(0.0, 0.5 + (radius - dist) / h));
    }
  }
  for (int passes = 1; passes <= 16; ++passes) {
    saturn::vof::Grid2D smoothed(n, n);
    saturn::vof::smoothBinomial(g, smoothed, passes);
    double max_error = 0.0;
    double sum_sq = 0.0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (g(i, j) <= 0.0 || g(i, j) >= 1.0) {
          continue;
        }
        const auto k = saturn::vof::curvatureLevelset(smoothed, i, j);
        if (!k.valid) {
          continue;
        }
        const double error = std::abs(k.kappa - 1.0 / radius);
        max_error = std::max(max_error, error);
        sum_sq += error * error;
        ++count;
      }
    }
    const double rms = std::sqrt(sum_sq / std::max(count, 1));
    std::printf("level-set passes=%d: max=%.3e rms=%.3e samples=%d\n", passes,
                max_error, rms, count);
  }
  // Use 16 smoothing passes for the pass/fail decision (the smoothing width
  // trades grid noise against curvature bias).
  saturn::vof::Grid2D smoothed(n, n);
  saturn::vof::smoothBinomial(g, smoothed, 16);
  double max_error = 0.0;
  int count = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (g(i, j) <= 0.0 || g(i, j) >= 1.0) {
        continue;
      }
      const auto k = saturn::vof::curvatureLevelset(smoothed, i, j);
      if (k.valid) {
        max_error = std::max(max_error, std::abs(k.kappa - 1.0 / radius));
        ++count;
      }
    }
  }
  const bool ok = max_error < 0.3 && count > 50;
  std::printf("VOF level-set curvature test %s\n", ok ? "passed" : "failed");
  return ok ? 0 : 1;
}
