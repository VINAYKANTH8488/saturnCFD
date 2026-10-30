// 2D PLIC VOF prototype: translation and solid-body rotation.

#include "numerics/vof2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

double circle(double x, double y, double x0, double y0, double r) {
  return (x - x0) * (x - x0) + (y - y0) * (y - y0) <= r * r ? 1.0 : 0.0;
}

void fillCircle(saturn::vof::Grid2D& g, double x0, double y0, double r) {
  for (int i = 0; i < g.nx(); ++i) {
    for (int j = 0; j < g.ny(); ++j) {
      const double x = (i + 0.5) * g.hx();
      const double y = (j + 0.5) * g.hy();
      g(i, j) = circle(x, y, x0, y0, r);
    }
  }
}

std::vector<double> faceUTranslation(const saturn::vof::Grid2D& g,
                                     double speed) {
  return std::vector<double>(static_cast<std::size_t>(g.nx()) * g.ny(), speed);
}

std::vector<double> faceURotation(const saturn::vof::Grid2D& g, double omega) {
  std::vector<double> u(static_cast<std::size_t>(g.nx()) * g.ny(), 0.0);
  for (int i = 0; i < g.nx(); ++i) {
    for (int j = 0; j < g.ny(); ++j) {
      const double y = (j + 0.5) * g.hy();
      u[static_cast<std::size_t>(i) * g.ny() + j] = -omega * (y - 0.5);
    }
  }
  return u;
}

std::vector<double> faceVRotation(const saturn::vof::Grid2D& g, double omega) {
  std::vector<double> v(static_cast<std::size_t>(g.nx()) * g.ny(), 0.0);
  for (int i = 0; i < g.nx(); ++i) {
    for (int j = 0; j < g.ny(); ++j) {
      const double x = (i + 0.5) * g.hx();
      v[static_cast<std::size_t>(i) * g.ny() + j] = omega * (x - 0.5);
    }
  }
  return v;
}

double l1Error(const saturn::vof::Grid2D& a, const saturn::vof::Grid2D& b) {
  double sum = 0.0;
  for (int i = 0; i < a.nx(); ++i) {
    for (int j = 0; j < a.ny(); ++j) {
      sum += std::abs(a(i, j) - b(i, j));
    }
  }
  return sum / static_cast<double>(a.nx()) / a.ny();
}

int checkBounds(const saturn::vof::Grid2D& g) {
  for (double c : g.data()) {
    if (c < -1.0e-8 || c > 1.0 + 1.0e-8) {
      return 1;
    }
  }
  return 0;
}

}  // namespace

int main() {
  int status = 0;
  {
    // One-dimensional step translation: the PLIC flux should be exact.
    const int n = 64;
    saturn::vof::Grid2D grid(n, 1);
    for (int i = 0; i < n; ++i) {
      grid(i, 0) = (i < n / 2) ? 1.0 : 0.0;
    }
    const double mass0 = grid.mass();
    const double speed = 0.3;
    const double tend = 0.1;
    const double dt = 0.25 * grid.hx() / speed;
    const int steps = static_cast<int>(std::ceil(tend / dt));
    const auto u = std::vector<double>(static_cast<std::size_t>(n), speed);
    const std::vector<double> v(static_cast<std::size_t>(n), 0.0);
    for (int s = 0; s < steps; ++s) {
      grid.advance(u, v, std::min(dt, tend - s * dt));
    }
    saturn::vof::Grid2D exact(n, 1);
    const double shift = speed * tend / grid.hx();
    for (int i = 0; i < n; ++i) {
      const double xs = i - shift;
      exact(i, 0) = (xs >= 0.0 && xs < n / 2.0) ? 1.0 : 0.0;
    }
    const double error = l1Error(grid, exact);
    const double drift = std::abs(grid.mass() - mass0) / mass0;
    std::printf("step 1D:     L1=%.3e mass_drift=%.3e bounds=%s\n", error,
                drift, checkBounds(grid) == 0 ? "ok" : "bad");
    if (error > 5.0e-3 || drift > 1.0e-12 || checkBounds(grid) != 0) {
      status = 1;
    }
  }

  {
    const int n = 32;
    saturn::vof::Grid2D grid(n, n);
    fillCircle(grid, 0.4, 0.5, 0.15);
    const double mass0 = grid.mass();

    const double speed = 0.2;
    const double tend = 0.3;
    const double h = grid.hx();
    const double dt = 0.25 * h / speed;
    const int steps = static_cast<int>(std::ceil(tend / dt));
    const auto u = faceUTranslation(grid, speed);
    const std::vector<double> v(grid.data().size(), 0.0);
    for (int s = 0; s < steps; ++s) {
      const double local_dt = std::min(dt, tend - static_cast<double>(s) * dt);
      grid.advance(u, v, local_dt);
    }

    saturn::vof::Grid2D exact(n, n);
    fillCircle(exact, 0.4 + speed * tend, 0.5, 0.15);
    const double error = l1Error(grid, exact);
    const double drift = std::abs(grid.mass() - mass0) / mass0;
    std::printf("translation: L1=%.3e mass_drift=%.3e bounds=%s\n", error,
                drift, checkBounds(grid) == 0 ? "ok" : "bad");
    if (error > 0.02 || drift > 1.0e-10 || checkBounds(grid) != 0) {
      status = 1;
    }
  }

  {
    const int n = 32;
    saturn::vof::Grid2D grid(n, n);
    fillCircle(grid, 0.5, 0.7, 0.15);
    const double mass0 = grid.mass();
    saturn::vof::Grid2D initial = grid;

    const double omega = 1.0;
    const auto u = faceURotation(grid, omega);
    const auto v = faceVRotation(grid, omega);
    const double maxSpeed = omega * 0.5;
    const double dt = 0.25 * grid.hx() / maxSpeed;
    const double period = 2.0 * kPi / omega;
    const int steps = static_cast<int>(std::ceil(period / dt));
    for (int s = 0; s < steps; ++s) {
      const double local_dt =
          std::min(dt, period - static_cast<double>(s) * dt);
      grid.advance(u, v, local_dt);
    }
    const double error = l1Error(grid, initial);
    const double drift = std::abs(grid.mass() - mass0) / mass0;
    std::printf("rotation:    L1=%.3e mass_drift=%.3e bounds=%s\n", error,
                drift, checkBounds(grid) == 0 ? "ok" : "bad");
    if (error > 2.0e-2 || drift > 1.0e-10 || checkBounds(grid) != 0) {
      status = 1;
    }
  }
  std::printf("VOF 2D test %s\n", status == 0 ? "passed" : "failed");
  return status;
}
