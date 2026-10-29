#pragma once

#include "numerics/vof2d.hpp"
#include "numerics/vof3d.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace saturn {
namespace vof {

// Conservative bounded redistribution.
//
// Advection can leave small overshoots (c < 0 or c > 1) at sharp interfaces.
// This pass clamps the field and redistributes the resulting mass error to
// cells with available capacity, iterating until the target mass is restored
// (or capacity is exhausted). It is a safety net, not a replacement for a
// bounded advection scheme.

inline void redistribute(Grid2D& g, int sweeps = 20) {
  const double target = g.mass();
  const double volume = g.hx() * g.hy();
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    for (double& c : g.data()) {
      c = std::min(1.0, std::max(0.0, c));
    }
    const double current = g.mass();
    const double delta = target - current;
    if (std::abs(delta) <= 1.0e-15 * std::max(1.0, std::abs(target))) {
      break;
    }
    double capacity = 0.0;
    for (double c : g.data()) {
      capacity += (delta > 0.0) ? (1.0 - c) : c;
    }
    if (capacity <= 0.0) {
      break;
    }
    const double amount = delta / (capacity * volume);
    for (double& c : g.data()) {
      const double available = (delta > 0.0) ? (1.0 - c) : c;
      c += amount * available;
      c = std::min(1.0, std::max(0.0, c));
    }
  }
}

inline void redistribute(Grid3D& g, int sweeps = 20) {
  const double target = g.mass();
  const double volume = g.cell_volume();
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    for (double& c : g.data()) {
      c = std::min(1.0, std::max(0.0, c));
    }
    const double current = g.mass();
    const double delta = target - current;
    if (std::abs(delta) <= 1.0e-15 * std::max(1.0, std::abs(target))) {
      break;
    }
    double capacity = 0.0;
    for (double c : g.data()) {
      capacity += (delta > 0.0) ? (1.0 - c) : c;
    }
    if (capacity <= 0.0) {
      break;
    }
    const double amount = delta / (capacity * volume);
    for (double& c : g.data()) {
      const double available = (delta > 0.0) ? (1.0 - c) : c;
      c += amount * available;
      c = std::min(1.0, std::max(0.0, c));
    }
  }
}

}  // namespace vof
}  // namespace saturn
