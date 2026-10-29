// 3D PLIC VOF prototype: clipping check, translation, and rotation.

#include "numerics/vof3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

double sphere(double x, double y, double z, double x0, double y0, double z0,
              double r) {
  return (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0) <=
                 r * r
             ? 1.0
             : 0.0;
}

void fill_sphere(saturn::vof::Grid3D& g, double x0, double y0, double z0,
                 double r) {
  for (int i = 0; i < g.nx(); ++i) {
    for (int j = 0; j < g.ny(); ++j) {
      for (int k = 0; k < g.nz(); ++k) {
        g(i, j, k) = sphere((i + 0.5) * g.hx(), (j + 0.5) * g.hy(),
                            (k + 0.5) * g.hz(), x0, y0, z0, r);
      }
    }
  }
}

double l1_error(const saturn::vof::Grid3D& a, const saturn::vof::Grid3D& b) {
  double sum = 0.0;
  for (int i = 0; i < a.nx(); ++i) {
    for (int j = 0; j < a.ny(); ++j) {
      for (int k = 0; k < a.nz(); ++k) {
        sum += std::abs(a(i, j, k) - b(i, j, k));
      }
    }
  }
  return sum / static_cast<double>(a.nx()) / a.ny() / a.nz();
}

int check_bounds(const saturn::vof::Grid3D& g) {
  for (double c : g.data()) {
    if (c < -1.0e-7 || c > 1.0 + 1.0e-7) {
      return 1;
    }
  }
  return 0;
}

}  // namespace

int main() {
  int status = 0;
  {
    const auto cube = saturn::vof::unit_cube();
    const double full = saturn::vof::polyhedron_volume(cube);
    const auto half =
        saturn::vof::clip_half_plane(cube, saturn::vof::Vec3{1, 0, 0}, 0.5);
    const double clipped = saturn::vof::polyhedron_volume(half);
    std::printf("clip: full=%.6f half=%.6f\n", full, clipped);
    if (std::abs(full - 1.0) > 1e-12 || std::abs(clipped - 0.5) > 1e-12) {
      status = 1;
    }
  }

  {
    // 1D step translation in x (ny = nz = 1).
    const int n = 32;
    saturn::vof::Grid3D grid(n, 1, 1);
    for (int i = 0; i < n; ++i) {
      grid(i, 0, 0) = (i < n / 2) ? 1.0 : 0.0;
    }
    const double mass0 = grid.mass();
    const double speed = 0.3;
    const double tend = 2.0 * grid.hx() / speed;  // exactly two cells
    const double dt = 0.25 * grid.hx() / speed;
    const int steps = static_cast<int>(std::ceil(tend / dt));
    std::vector<double> u(static_cast<std::size_t>(n), speed);
    std::vector<double> v(static_cast<std::size_t>(n), 0.0);
    std::vector<double> w(static_cast<std::size_t>(n), 0.0);
    for (int s = 0; s < steps; ++s) {
      grid.advance(u, v, w, std::min(dt, tend - s * dt));
    }
    saturn::vof::Grid3D exact(n, 1, 1);
    const double shift = speed * tend / grid.hx();
    for (int i = 0; i < n; ++i) {
      const double xs = i - shift;
      exact(i, 0, 0) = (xs >= 0.0 && xs < n / 2.0) ? 1.0 : 0.0;
    }
    const double error = l1_error(grid, exact);
    const double drift = std::abs(grid.mass() - mass0) / mass0;
    std::printf("step 1D: L1=%.3e mass_drift=%.3e bounds=%s\n", error, drift,
                check_bounds(grid) == 0 ? "ok" : "bad");
    if (error > 5.0e-3 || drift > 1.0e-12 || check_bounds(grid) != 0) {
      status = 1;
    }
  }

  {
    const int n = 16;
    saturn::vof::Grid3D grid(n, n, n);
    fill_sphere(grid, 0.4, 0.5, 0.5, 0.15);
    const double mass0 = grid.mass();
    const double speed = 0.2;
    const double tend = 0.2;
    const double dt = 0.25 * grid.hx() / speed;
    const int steps = static_cast<int>(std::ceil(tend / dt));
    const std::size_t total = static_cast<std::size_t>(n) * n * n;
    std::vector<double> u(total, speed);
    std::vector<double> v(total, 0.0);
    std::vector<double> w(total, 0.0);
    for (int s = 0; s < steps; ++s) {
      grid.advance(u, v, w, std::min(dt, tend - s * dt));
    }
    saturn::vof::Grid3D exact(n, n, n);
    fill_sphere(exact, 0.4 + speed * tend, 0.5, 0.5, 0.15);
    const double error = l1_error(grid, exact);
    const double drift = std::abs(grid.mass() - mass0) / mass0;
    std::printf("translation: L1=%.3e mass_drift=%.3e bounds=%s\n", error,
                drift, check_bounds(grid) == 0 ? "ok" : "bad");
    if (error > 2.0e-2 || drift > 1.0e-10 || check_bounds(grid) != 0) {
      status = 1;
    }
  }

  {
    const int n = 16;
    saturn::vof::Grid3D grid(n, n, n);
    fill_sphere(grid, 0.5, 0.7, 0.5, 0.15);
    const double mass0 = grid.mass();
    saturn::vof::Grid3D initial = grid;
    const double omega = 1.0;
    const std::size_t total = static_cast<std::size_t>(n) * n * n;
    std::vector<double> u(total), v(total), w(total, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
          const std::size_t idx = (static_cast<std::size_t>(i) * n + j) * n + k;
          u[idx] = -omega * ((j + 0.5) * grid.hy() - 0.5);
          v[idx] = omega * ((i + 0.5) * grid.hx() - 0.5);
        }
      }
    }
    const double dt = 0.25 * grid.hx() / (omega * 0.5);
    const double period = 2.0 * kPi / omega;
    const int steps = static_cast<int>(std::ceil(period / dt));
    for (int s = 0; s < steps; ++s) {
      grid.advance(u, v, w, std::min(dt, period - s * dt));
    }
    const double error = l1_error(grid, initial);
    const double drift = std::abs(grid.mass() - mass0) / mass0;
    double cmin = 0.0;
    double cmax = 1.0;
    for (double c : grid.data()) {
      cmin = std::min(cmin, c);
      cmax = std::max(cmax, c);
    }
    std::printf("rotation: L1=%.3e mass_drift=%.3e range=[%.3e,%.3e]\n", error,
                drift, cmin, cmax);
    if (error > 5.0e-2 || drift > 1.0e-10 || check_bounds(grid) != 0) {
      status = 1;
    }
  }

  std::printf("VOF 3D test %s\n", status == 0 ? "passed" : "failed");
  return status;
}
