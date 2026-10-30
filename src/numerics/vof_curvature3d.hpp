#pragma once

#include "numerics/vof3d.hpp"

#include <algorithm>
#include <cmath>

namespace saturn {
namespace vof {

// Height-function curvature for a mostly horizontal interface z = f(x, y).
//
//   h_ij = sum_k c_ijk hz
//   kappa = [h_xx (1+h_y^2) + h_yy (1+h_x^2) - 2 h_x h_y h_xy]
//           / (1 + h_x^2 + h_y^2)^(3/2)
//
// Derivatives use 5-point stencils in x and y.

struct Curvature3D {
  double kappa = 0.0;
  double hx = 0.0;
  double hy = 0.0;
  bool valid = false;
};

inline double columnHeight(const Grid3D& g, int i, int j) {
  double sum = 0.0;
  for (int k = 0; k < g.nz(); ++k) {
    sum += g(i, j, k) * g.hz();
  }
  return sum;
}

inline Curvature3D curvatureZ(const Grid3D& g, int i, int j) {
  Curvature3D result;
  const int s = 2;
  if (i - s < 0 || i + s >= g.nx() || j - s < 0 || j + s >= g.ny()) {
    return result;
  }
  const double hx = g.hx();
  const double hy = g.hy();
  const double hm2 = columnHeight(g, i - 2, j);
  const double hm1 = columnHeight(g, i - 1, j);
  const double h0 = columnHeight(g, i, j);
  const double hp1 = columnHeight(g, i + 1, j);
  const double hp2 = columnHeight(g, i + 2, j);

  const double jm2 = columnHeight(g, i, j - 2);
  const double jm1 = columnHeight(g, i, j - 1);
  const double jp1 = columnHeight(g, i, j + 1);
  const double jp2 = columnHeight(g, i, j + 2);

  const double hx_der = (hm2 - 8.0 * hm1 + 8.0 * hp1 - hp2) / (12.0 * hx);
  const double hy_der = (jm2 - 8.0 * jm1 + 8.0 * jp1 - jp2) / (12.0 * hy);
  const double hxx =
      (-hp2 + 16.0 * hp1 - 30.0 * h0 + 16.0 * hm1 - hm2) / (12.0 * hx * hx);
  const double hyy =
      (-jp2 + 16.0 * jp1 - 30.0 * h0 + 16.0 * jm1 - jm2) / (12.0 * hy * hy);
  // Mixed derivative from the 4-point central difference of h_x in y.
  const double hx_jm =
      (columnHeight(g, i + 1, j - 1) - columnHeight(g, i - 1, j - 1)) /
      (2.0 * hx);
  const double hx_jp =
      (columnHeight(g, i + 1, j + 1) - columnHeight(g, i - 1, j + 1)) /
      (2.0 * hx);
  const double hxy = (hx_jp - hx_jm) / (2.0 * hy);

  const double gx2 = hx_der * hx_der;
  const double gy2 = hy_der * hy_der;
  const double denom = std::pow(1.0 + gx2 + gy2, 1.5);
  result.kappa =
      (hxx * (1.0 + gy2) + hyy * (1.0 + gx2) - 2.0 * hx_der * hy_der * hxy) /
      denom;
  result.hx = hx_der;
  result.hy = hy_der;
  result.valid = true;
  return result;
}

}  // namespace vof
}  // namespace saturn
