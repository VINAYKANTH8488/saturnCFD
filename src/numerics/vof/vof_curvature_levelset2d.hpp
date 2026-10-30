#pragma once

#include "numerics/vof/vof2d.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace saturn {
namespace vof {

// Level-set curvature on a smoothed volume fraction:
//
//   phi = smooth(c) - 1/2
//   n   = grad(phi) / |grad(phi)|
//   kappa = -div(n)
//
// A short separable binomial smoothing removes the grid-scale staircase of a
// sharp PLIC field. This works for closed interfaces (bubbles) where the
// column height function is not single-valued.

inline void smoothBinomial(const Grid2D& in, Grid2D& out, int passes) {
  const int nx = in.nx();
  const int ny = in.ny();
  std::vector<double> buffer(in.data().size(), 0.0);
  Grid2D work = in;
  for (int pass = 0; pass < passes; ++pass) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        buffer[static_cast<std::size_t>(j) * nx + i] =
            0.25 * work(i - 1, j) + 0.5 * work(i, j) + 0.25 * work(i + 1, j);
      }
    }
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        work(i, j) =
            0.25 *
                buffer[static_cast<std::size_t>((j - 1 + ny) % ny) * nx + i] +
            0.5 * buffer[static_cast<std::size_t>(j) * nx + i] +
            0.25 * buffer[static_cast<std::size_t>((j + 1) % ny) * nx + i];
      }
    }
  }
  out = work;
}

struct LevelSetCurvature {
  double kappa = 0.0;
  bool valid = false;
};

inline LevelSetCurvature curvatureLevelset(const Grid2D& smoothed, int i,
                                           int j) {
  LevelSetCurvature result;
  const double hx = smoothed.hx();
  const double hy = smoothed.hy();
  const double gx = 0.5 * (smoothed(i + 1, j) - smoothed(i - 1, j)) / hx;
  const double gy = 0.5 * (smoothed(i, j + 1) - smoothed(i, j - 1)) / hy;
  const double norm = std::sqrt(gx * gx + gy * gy);
  if (norm < 1.0e-8) {
    return result;
  }
  // Divergence of the normalized gradient.
  auto normalX = [&](int ii, int jj) {
    const double x = 0.5 * (smoothed(ii + 1, jj) - smoothed(ii - 1, jj)) / hx;
    const double y = 0.5 * (smoothed(ii, jj + 1) - smoothed(ii, jj - 1)) / hy;
    const double n = std::sqrt(x * x + y * y);
    return n > 1.0e-10 ? x / n : 0.0;
  };
  auto normalY = [&](int ii, int jj) {
    const double x = 0.5 * (smoothed(ii + 1, jj) - smoothed(ii - 1, jj)) / hx;
    const double y = 0.5 * (smoothed(ii, jj + 1) - smoothed(ii, jj - 1)) / hy;
    const double n = std::sqrt(x * x + y * y);
    return n > 1.0e-10 ? y / n : 0.0;
  };
  const double div = 0.5 * (normalX(i + 1, j) - normalX(i - 1, j)) / hx +
                     0.5 * (normalY(i, j + 1) - normalY(i, j - 1)) / hy;
  result.kappa = -div;
  result.valid = true;
  return result;
}

}  // namespace vof
}  // namespace saturn
