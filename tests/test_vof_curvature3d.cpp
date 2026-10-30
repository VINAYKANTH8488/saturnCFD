// Height-function curvature on a smooth periodic 3D interface.

#include "numerics/vof/vof_curvature3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

int main() {
  const int n = 48;
  const double amplitude = 0.1;
  const double k = 2.0 * kPi;
  saturn::vof::Grid3D grid(n, n, n);

  for (int i = 0; i < n; ++i) {
    const double x = (i + 0.5) * grid.hx();
    for (int j = 0; j < n; ++j) {
      const double y = (j + 0.5) * grid.hy();
      const double interface =
          0.5 + amplitude * std::sin(k * x) * std::sin(k * y);
      for (int m = 0; m < n; ++m) {
        const double lower = m * grid.hz();
        const double fraction = (interface - lower) / grid.hz();
        grid(i, j, m) = std::min(1.0, std::max(0.0, fraction));
      }
    }
  }

  double max_error = 0.0;
  double sum_sq = 0.0;
  int count = 0;
  for (int i = 2; i < n - 2; ++i) {
    const double x = (i + 0.5) * grid.hx();
    for (int j = 2; j < n - 2; ++j) {
      const double y = (j + 0.5) * grid.hy();
      const double sx = std::sin(k * x);
      const double cx = std::cos(k * x);
      const double sy = std::sin(k * y);
      const double cy = std::cos(k * y);
      const double hx = amplitude * k * cx * sy;
      const double hy = amplitude * k * sx * cy;
      const double hxx = -amplitude * k * k * sx * sy;
      const double hyy = -amplitude * k * k * sx * sy;
      const double hxy = amplitude * k * k * cx * cy;
      const double gx2 = hx * hx;
      const double gy2 = hy * hy;
      const double exact =
          (hxx * (1.0 + gy2) + hyy * (1.0 + gx2) - 2.0 * hx * hy * hxy) /
          std::pow(1.0 + gx2 + gy2, 1.5);
      const auto curvature = saturn::vof::curvatureZ(grid, i, j);
      if (!curvature.valid) {
        continue;
      }
      const double error = std::abs(curvature.kappa - exact);
      max_error = std::max(max_error, error);
      sum_sq += error * error;
      ++count;
    }
  }
  const double rms = std::sqrt(sum_sq / std::max(count, 1));
  std::printf("3D curvature: max=%.3e rms=%.3e samples=%d\n", max_error, rms,
              count);
  const bool ok = max_error < 0.05 && count > n * n / 2;
  std::printf("VOF 3D curvature test %s\n", ok ? "passed" : "failed");
  return ok ? 0 : 1;
}
