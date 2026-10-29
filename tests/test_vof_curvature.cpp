// Height-function curvature on a smooth periodic interface.

#include "numerics/vof_curvature2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

int main() {
  const int n = 128;
  const double lx = 1.0;
  const double amplitude = 0.1;
  const double k = 2.0 * kPi;
  saturn::vof::Grid2D grid(n, n, lx, 1.0);

  for (int i = 0; i < n; ++i) {
    const double x = (i + 0.5) * grid.hx();
    const double interface = 0.5 + amplitude * std::sin(k * x);
    for (int j = 0; j < n; ++j) {
      const double lower = j * grid.hy();
      const double fraction = (interface - lower) / grid.hy();
      grid(i, j) = std::min(1.0, std::max(0.0, fraction));
    }
  }

  double max_error = 0.0;
  double sum_sq = 0.0;
  int count = 0;
  for (int i = 2; i < n - 2; ++i) {
    const double x = (i + 0.5) * grid.hx();
    const double slope = amplitude * k * std::cos(k * x);
    const double second = -amplitude * k * k * std::sin(k * x);
    const double exact = second / std::pow(1.0 + slope * slope, 1.5);
    const auto curvature = saturn::vof::curvature_y(grid, i, 2);
    if (!curvature.valid) {
      continue;
    }
    const double error = std::abs(curvature.kappa - exact);
    max_error = std::max(max_error, error);
    sum_sq += error * error;
    ++count;
  }
  const double rms = std::sqrt(sum_sq / std::max(count, 1));
  std::printf("curvature: max=%.3e rms=%.3e samples=%d\n", max_error, rms,
              count);
  const bool ok = max_error < 0.05 && count > n / 2;
  std::printf("VOF curvature test %s\n", ok ? "passed" : "failed");
  return ok ? 0 : 1;
}
