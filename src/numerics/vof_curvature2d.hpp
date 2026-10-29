#pragma once

#include "numerics/vof2d.hpp"

#include <algorithm>
#include <cmath>

namespace saturn {
namespace vof {

// Height-function curvature for a mostly horizontal interface y = f(x).
//
// The height function is the column integral of the volume fraction,
//   h_i = sum_j c_ij * hy,
// and the curvature is
//   kappa = h'' / (1 + h'^2)^(3/2).
// A five-point stencil is used for h' and h'' by default.
struct Curvature2D {
  double kappa = 0.0;
  double slope = 0.0;
  bool valid = false;
};

inline double column_height(const Grid2D& g, int i) {
  double sum = 0.0;
  for (int j = 0; j < g.ny(); ++j) {
    sum += g(i, j) * g.hy();
  }
  return sum;
}

inline double row_height(const Grid2D& g, int j) {
  double sum = 0.0;
  for (int i = 0; i < g.nx(); ++i) {
    sum += g(i, j) * g.hx();
  }
  return sum;
}

// Curvature of a y = f(x) interface at column i.
inline Curvature2D curvature_y(const Grid2D& g, int i, int stencil = 2) {
  Curvature2D result;
  if (i - stencil < 0 || i + stencil >= g.nx()) {
    return result;
  }
  const int s = stencil;
  const double hx = g.hx();
  double first = 0.0;
  double second = 0.0;
  if (s == 1) {
    const double hm = column_height(g, i - 1);
    const double h0 = column_height(g, i);
    const double hp = column_height(g, i + 1);
    first = (hp - hm) / (2.0 * hx);
    second = (hp - 2.0 * h0 + hm) / (hx * hx);
  } else {
    // 5-point first and second derivatives.
    const double hm2 = column_height(g, i - 2);
    const double hm1 = column_height(g, i - 1);
    const double h0 = column_height(g, i);
    const double hp1 = column_height(g, i + 1);
    const double hp2 = column_height(g, i + 2);
    first = (hm2 - 8.0 * hm1 + 8.0 * hp1 - hp2) / (12.0 * hx);
    second =
        (-hp2 + 16.0 * hp1 - 30.0 * h0 + 16.0 * hm1 - hm2) / (12.0 * hx * hx);
  }
  result.slope = first;
  result.kappa = second / std::pow(1.0 + first * first, 1.5);
  result.valid = true;
  return result;
}

// Curvature of an x = f(y) interface at row j (transpose of curvature_y).
inline Curvature2D curvature_x(const Grid2D& g, int j, int stencil = 2) {
  Curvature2D result;
  if (j - stencil < 0 || j + stencil >= g.ny()) {
    return result;
  }
  const int s = stencil;
  const double hy = g.hy();
  double first = 0.0;
  double second = 0.0;
  if (s == 1) {
    const double hm = row_height(g, j - 1);
    const double h0 = row_height(g, j);
    const double hp = row_height(g, j + 1);
    first = (hp - hm) / (2.0 * hy);
    second = (hp - 2.0 * h0 + hm) / (hy * hy);
  } else {
    const double hm2 = row_height(g, j - 2);
    const double hm1 = row_height(g, j - 1);
    const double h0 = row_height(g, j);
    const double hp1 = row_height(g, j + 1);
    const double hp2 = row_height(g, j + 2);
    first = (hm2 - 8.0 * hm1 + 8.0 * hp1 - hp2) / (12.0 * hy);
    second =
        (-hp2 + 16.0 * hp1 - 30.0 * h0 + 16.0 * hm1 - hm2) / (12.0 * hy * hy);
  }
  result.slope = first;
  result.kappa = second / std::pow(1.0 + first * first, 1.5);
  result.valid = true;
  return result;
}

}  // namespace vof
}  // namespace saturn
