#pragma once

#ifdef SATURN_HAVE_FFTW

#include "numerics/multiphase/twophase2d.hpp"
#include "numerics/vof/vof2d.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace saturn {
namespace twophase {

// Wall-bounded two-phase solver prototype.
//
// Closed box, no-slip/no-penetration walls, gravity in -y, variable density
// rho(c) and viscosity mu(c), wall-aware PLIC transport, balanced CSF, and a
// face-staggered conservative momentum update. The volume fraction and
// momentum fluxes are built from the same PLIC face fluxes, so the discrete
// density and momentum updates are consistent. The pressure projection uses
// the variable-coefficient Neumann PCG solver. The scheme is first order in
// time; it is a validation prototype, not yet a production solver.
class DamBreak2D {
 public:
  DamBreak2D(int nx, int ny, double lx = 1.0, double ly = 1.0)
      : nx_(nx),
        ny_(ny),
        hx_(lx / nx),
        hy_(ly / ny),
        c_(static_cast<std::size_t>(nx) * ny, 0.0),
        u_(static_cast<std::size_t>(nx) * ny, 0.0),
        v_(static_cast<std::size_t>(nx) * ny, 0.0),
        p_(static_cast<std::size_t>(nx) * ny, 0.0) {}

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  double hx() const { return hx_; }
  double hy() const { return hy_; }
  std::vector<double>& c() { return c_; }
  const std::vector<double>& c() const { return c_; }
  std::vector<double>& u() { return u_; }
  std::vector<double>& v() { return v_; }
  const std::vector<double>& p() const { return p_; }

  double mass() const {
    double sum = 0.0;
    for (double value : c_) {
      sum += value;
    }
    return sum * hx_ * hy_;
  }

  void step(double dt, double rho_l, double rho_g, double mu, double gy,
            bool project = true, double sigma = 0.0) {
    step(dt, rho_l, rho_g, mu, mu, gy, project, sigma);
  }

  // Face-staggered conservative step. c and the momentum use the same PLIC
  // mass fluxes; momentum lives on the MAC faces, so no cell-centred
  // velocity re-averaging is needed before the projection.
  void step(double dt, double rho_l, double rho_g, double mu_l, double mu_g,
            double gy, bool project, double sigma) {
    const std::size_t total = static_cast<std::size_t>(nx_) * ny_;
    std::vector<double> rho_old(total, 0.0);
    for (std::size_t q = 0; q < total; ++q) {
      rho_old[q] = rho_l * c_[q] + rho_g * (1.0 - c_[q]);
    }

    std::vector<double> vof_x;
    std::vector<double> vof_y;
    plicFluxX(dt, vof_x);
    plicFluxY(dt, vof_y);

    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double right = (i + 1 == nx_) ? 0.0 : vof_x[index + 1];
        const double high = (j + 1 == ny_) ? 0.0 : vof_y[index + nx_];
        c_[index] -= (right - vof_x[index]) / (hx_ * hy_) +
                     (high - vof_y[index]) / (hx_ * hy_);
      }
    }
    restoreBounds();

    std::vector<double> rho_new(total, 0.0);
    std::vector<double> mu_cell(total, 0.0);
    for (std::size_t q = 0; q < total; ++q) {
      rho_new[q] = rho_l * c_[q] + rho_g * (1.0 - c_[q]);
      mu_cell[q] = mu_l * c_[q] + mu_g * (1.0 - c_[q]);
    }

    std::vector<double> rho_u_old(total, 0.0);
    std::vector<double> rho_v_old(total, 0.0);
    std::vector<double> rho_u_new(total, 0.0);
    std::vector<double> rho_v_new(total, 0.0);
    std::vector<double> m_u(total, 0.0);
    std::vector<double> m_v(total, 0.0);
    std::vector<double> mass_x(total, 0.0);
    std::vector<double> mass_y(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        rho_u_old[index] = (i == 0)
                               ? rho_old[index]
                               : 0.5 * (rho_old[index - 1] + rho_old[index]);
        rho_v_old[index] = (j == 0)
                               ? rho_old[index]
                               : 0.5 * (rho_old[index - nx_] + rho_old[index]);
        rho_u_new[index] = (i == 0)
                               ? rho_new[index]
                               : 0.5 * (rho_new[index - 1] + rho_new[index]);
        rho_v_new[index] = (j == 0)
                               ? rho_new[index]
                               : 0.5 * (rho_new[index - nx_] + rho_new[index]);
        m_u[index] = rho_u_old[index] * u_[index];
        m_v[index] = rho_v_old[index] * v_[index];
        if (i > 0) {
          mass_x[index] = rho_l * vof_x[index] +
                          rho_g * (u_[index] * dt * hy_ - vof_x[index]);
        }
        if (j > 0) {
          mass_y[index] = rho_l * vof_y[index] +
                          rho_g * (v_[index] * dt * hx_ - vof_y[index]);
        }
      }
    }

    std::vector<double> bx_new(total, 0.0);
    std::vector<double> by_new(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        if (i > 0) {
          const double rl = rho_new[index - 1];
          bx_new[index] = (rl + rho_new[index]) / (2.0 * rl * rho_new[index]);
        }
        if (j > 0) {
          const double rd = rho_new[index - nx_];
          by_new[index] = (rd + rho_new[index]) / (2.0 * rd * rho_new[index]);
        }
      }
    }

    // u-momentum on x-faces.
    std::vector<double> u_new(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double mass_left =
            0.5 * (((i > 1) ? mass_x[index - 1] : 0.0) + mass_x[index]);
        const double mass_right =
            0.5 * (mass_x[index] + ((i + 1 < nx_) ? mass_x[index + 1] : 0.0));
        const double u_left_boundary =
            0.5 * (((i > 1) ? u_[index - 1] : 0.0) + u_[index]);
        const double u_right_boundary =
            0.5 * (u_[index] + ((i + 1 < nx_) ? u_[index + 1] : 0.0));
        const double f_left =
            mass_left * (u_left_boundary > 0.0 ? u_[index - 1] : u_[index]);
        const double f_right =
            mass_right * (u_right_boundary > 0.0
                              ? u_[index]
                              : ((i + 1 < nx_) ? u_[index + 1] : 0.0));
        const double v_bottom = 0.5 * (v_[index - 1] + v_[index]);
        const double v_top =
            0.5 * (((j + 1 < ny_) ? v_[index - 1 + nx_] : 0.0) +
                   ((j + 1 < ny_) ? v_[index + nx_] : 0.0));
        const double mass_bottom = 0.5 * (((j > 0) ? mass_y[index - 1] : 0.0) +
                                          ((j > 0) ? mass_y[index] : 0.0));
        const double mass_top =
            0.5 * (((j + 1 < ny_) ? mass_y[index - 1 + nx_] : 0.0) +
                   ((j + 1 < ny_) ? mass_y[index + nx_] : 0.0));
        const double g_bottom =
            mass_bottom *
            (v_bottom > 0.0 ? ((j > 0) ? u_[index - nx_] : 0.0) : u_[index]);
        const double g_top =
            mass_top *
            (v_top > 0.0 ? u_[index] : ((j + 1 < ny_) ? u_[index + nx_] : 0.0));
        const double u_left = (i > 1) ? u_[index - 1] : 0.0;
        const double u_right = (i + 1 < nx_) ? u_[index + 1] : 0.0;
        const double u_low = (j > 0) ? u_[index - nx_] : -u_[index];
        const double u_high = (j + 1 < ny_) ? u_[index + nx_] : -u_[index];
        const double lap = (u_right - 2.0 * u_[index] + u_left) / (hx_ * hx_) +
                           (u_high - 2.0 * u_[index] + u_low) / (hy_ * hy_);
        const double mu_face = 0.5 * (mu_cell[index - 1] + mu_cell[index]);
        const double m_new =
            m_u[index] -
            ((f_right - f_left) + (g_top - g_bottom)) / (hx_ * hy_);
        u_new[index] =
            m_new / rho_u_new[index] + dt * (mu_face * bx_new[index] * lap);
      }
      u_new[static_cast<std::size_t>(j) * nx_] = 0.0;
    }

    // v-momentum on y-faces.
    std::vector<double> v_new(total, 0.0);
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double mass_left = mass_x[index];
        const double mass_right = (i + 1 < nx_) ? mass_x[index + 1] : 0.0;
        const double f_left =
            mass_left *
            (u_[index] > 0.0 ? ((i > 0) ? v_[index - 1] : 0.0) : v_[index]);
        const double f_right =
            mass_right * (((i + 1 < nx_) && u_[index + 1] > 0.0)
                              ? v_[index]
                              : ((i + 1 < nx_) ? v_[index + 1] : 0.0));
        const double mass_bottom =
            0.5 * (((j > 1) ? mass_y[index - nx_] : 0.0) + mass_y[index]);
        const double mass_top =
            0.5 * (mass_y[index] + ((j + 1 < ny_) ? mass_y[index + nx_] : 0.0));
        const double v_bottom =
            0.5 * (((j > 1) ? v_[index - nx_] : 0.0) + v_[index]);
        const double v_top =
            0.5 * (v_[index] + ((j + 1 < ny_) ? v_[index + nx_] : 0.0));
        const double g_bottom =
            mass_bottom *
            (v_bottom > 0.0 ? ((j > 1) ? v_[index - nx_] : 0.0) : v_[index]);
        const double g_top =
            mass_top *
            (v_top > 0.0 ? v_[index] : ((j + 1 < ny_) ? v_[index + nx_] : 0.0));
        const double v_low = (j > 1) ? v_[index - nx_] : 0.0;
        const double v_high = (j + 1 < ny_) ? v_[index + nx_] : 0.0;
        const double v_left = (i > 0) ? v_[index - 1] : 0.0;
        const double v_right = (i + 1 < nx_) ? v_[index + 1] : 0.0;
        const double lap = (v_right - 2.0 * v_[index] + v_left) / (hx_ * hx_) +
                           (v_high - 2.0 * v_[index] + v_low) / (hy_ * hy_);
        const double mu_face = 0.5 * (mu_cell[index - nx_] + mu_cell[index]);
        const double m_new =
            m_v[index] -
            ((f_right - f_left) + (g_top - g_bottom)) / (hx_ * hy_);
        v_new[index] = m_new / rho_v_new[index] +
                       dt * (mu_face * by_new[index] * lap + gy);
      }
    }

    // CSF acts on faces.
    std::vector<double> csf_x;
    std::vector<double> csf_y;
    csfForces(sigma, csf_x, csf_y);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        u_new[index] += dt * bx_new[index] * csf_x[index];
      }
    }
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        v_new[index] += dt * by_new[index] * csf_y[index];
      }
    }

    u_ = u_new;
    v_ = v_new;

    if (project) {
      std::vector<double> div(total, 0.0);
      for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
          const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
          const double u_right = (i + 1 == nx_) ? 0.0 : u_[index + 1];
          const double v_high = (j + 1 == ny_) ? 0.0 : v_[index + nx_];
          div[index] = (u_right - u_[index]) / hx_ + (v_high - v_[index]) / hy_;
        }
      }
      for (double& value : div) {
        value /= dt;
      }
      VariablePoisson2D solver(nx_, ny_, hx_, hy_, bx_new, by_new, false,
                               false);
      solver.solve(div, p_);
      for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
          const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
          u_[index] -= dt * bx_new[index] * (p_[index] - p_[index - 1]) / hx_;
        }
        u_[static_cast<std::size_t>(j) * nx_] = 0.0;
      }
      for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
          const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
          v_[index] -= dt * by_new[index] * (p_[index] - p_[index - nx_]) / hy_;
        }
      }
      for (int i = 0; i < nx_; ++i) {
        v_[i] = 0.0;
      }
    }
  }

  // Wall-aware PLIC advection: Strang split x/y with geometric face fluxes.
  void advectPlic(double dt) {
    sweepXPlic(0.5 * dt);
    sweepYPlic(dt);
    sweepXPlic(0.5 * dt);
    restoreBounds();
  }

 private:
  std::size_t cellIndex(int i, int j) const {
    const int ii = std::min(std::max(i, 0), nx_ - 1);
    const int jj = std::min(std::max(j, 0), ny_ - 1);
    return static_cast<std::size_t>(jj) * nx_ + ii;
  }

  // Balanced CSF force on faces: sigma * kappa * grad(c), with a smoothed
  // volume fraction and mirror (zero-normal-gradient) ghosts at the walls.
  void csfForces(double sigma, std::vector<double>& fx,
                 std::vector<double>& fy) const {
    const std::size_t total = static_cast<std::size_t>(nx_) * ny_;
    fx.assign(total, 0.0);
    fy.assign(total, 0.0);
    if (sigma <= 0.0) {
      return;
    }
    std::vector<double> smoothed = c_;
    for (int pass = 0; pass < 8; ++pass) {
      std::vector<double> buffer(smoothed);
      for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
          buffer[cellIndex(i, j)] = 0.25 * smoothed[cellIndex(i - 1, j)] +
                                    0.5 * smoothed[cellIndex(i, j)] +
                                    0.25 * smoothed[cellIndex(i + 1, j)];
        }
      }
      std::vector<double> next(buffer);
      for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
          next[cellIndex(i, j)] = 0.25 * buffer[cellIndex(i, j - 1)] +
                                  0.5 * buffer[cellIndex(i, j)] +
                                  0.25 * buffer[cellIndex(i, j + 1)];
        }
      }
      smoothed = next;
    }
    auto grad = [&](int i, int j, double& gx, double& gy) {
      gx = 0.5 *
           (smoothed[cellIndex(i + 1, j)] - smoothed[cellIndex(i - 1, j)]) /
           hx_;
      gy = 0.5 *
           (smoothed[cellIndex(i, j + 1)] - smoothed[cellIndex(i, j - 1)]) /
           hy_;
    };
    auto normalX = [&](int i, int j) {
      double gx = 0.0;
      double gy = 0.0;
      grad(i, j, gx, gy);
      const double norm = std::sqrt(gx * gx + gy * gy);
      return norm > 1.0e-10 ? gx / norm : 0.0;
    };
    auto normalY = [&](int i, int j) {
      double gx = 0.0;
      double gy = 0.0;
      grad(i, j, gx, gy);
      const double norm = std::sqrt(gx * gx + gy * gy);
      return norm > 1.0e-10 ? gy / norm : 0.0;
    };
    std::vector<double> kappa(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const double div = 0.5 * (normalX(i + 1, j) - normalX(i - 1, j)) / hx_ +
                           0.5 * (normalY(i, j + 1) - normalY(i, j - 1)) / hy_;
        kappa[cellIndex(i, j)] = -div;
      }
    }
    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = cellIndex(i, j);
        const std::size_t left = cellIndex(i - 1, j);
        const double kx = 0.5 * (kappa[index] + kappa[left]);
        fx[index] = sigma * kx * (c_[index] - c_[left]) / hx_;
      }
    }
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = cellIndex(i, j);
        const std::size_t low = cellIndex(i, j - 1);
        const double ky = 0.5 * (kappa[index] + kappa[low]);
        fy[index] = sigma * ky * (c_[index] - c_[low]) / hy_;
      }
    }
  }

  // Strang splitting can leave small overshoots at sharp interfaces even for a
  // discretely divergence-free velocity. Clamp and redistribute the resulting
  // mass error over available capacity so the update stays conservative and
  // bounded (the same construction as vof::redistribute, for a raw vector).
  void restoreBounds(int sweeps = 8) {
    const double cellVolume = hx_ * hy_;
    double target = 0.0;
    for (double value : c_) {
      target += value * cellVolume;
    }
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      for (double& value : c_) {
        value = std::min(1.0, std::max(0.0, value));
      }
      double current = 0.0;
      for (double value : c_) {
        current += value * cellVolume;
      }
      const double delta = target - current;
      if (std::abs(delta) <= 1.0e-14 * std::max(1.0, std::abs(target))) {
        break;
      }
      double capacity = 0.0;
      for (double value : c_) {
        const bool eligible = (delta > 0.0) ? (value > 0.0) : (value < 1.0);
        if (eligible) {
          capacity += (delta > 0.0) ? (1.0 - value) : value;
        }
      }
      if (capacity * cellVolume < std::abs(delta)) {
        // Interface cells cannot absorb the correction; fall back to the whole
        // domain.
        capacity = 0.0;
        for (double value : c_) {
          capacity += (delta > 0.0) ? (1.0 - value) : value;
        }
      }
      if (capacity * cellVolume <= 0.0) {
        break;
      }
      const double weight = delta / (capacity * cellVolume);
      for (double& value : c_) {
        const bool eligible = (delta > 0.0) ? (value > 0.0) : (value < 1.0);
        if (!eligible) {
          continue;
        }
        const double available = (delta > 0.0) ? (1.0 - value) : value;
        value += weight * available;
        value = std::min(1.0, std::max(0.0, value));
      }
    }
  }

  double cval(int i, int j) const {
    const int ii = std::min(std::max(i, 0), nx_ - 1);
    const int jj = std::min(std::max(j, 0), ny_ - 1);
    return c_[static_cast<std::size_t>(jj) * nx_ + ii];
  }

  // Geometric PLIC flux of fluid 1 through each x-face (indexed like u_).
  void plicFluxX(double dt, std::vector<double>& flux) const {
    const auto rec = wallReconstructions();
    flux.assign(c_.size(), 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double vel = u_[index];
        if (vel == 0.0) {
          continue;
        }
        const int up = vel > 0.0 ? i - 1 : i;
        const auto& r = rec[static_cast<std::size_t>(j) * nx_ + up];
        const double width = std::min(std::abs(vel) * dt / hx_, 1.0);
        const double x0 = vel > 0.0 ? 1.0 - width : 0.0;
        const double x1 = vel > 0.0 ? 1.0 : width;
        std::vector<vof::Point> poly = vof::fluidPolygon(r);
        poly = vof::clipAxis(poly, 0, x0, -1.0);
        poly = vof::clipAxis(poly, 0, x1, 1.0);
        flux[index] =
            (vel > 0.0 ? 1.0 : -1.0) * vof::polygonArea(poly) * hx_ * hy_;
      }
    }
  }

  // Geometric PLIC flux of fluid 1 through each y-face (indexed like v_).
  void plicFluxY(double dt, std::vector<double>& flux) const {
    const auto rec = wallReconstructions();
    flux.assign(c_.size(), 0.0);
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double vel = v_[index];
        if (vel == 0.0) {
          continue;
        }
        const int up = vel > 0.0 ? j - 1 : j;
        const auto& r = rec[static_cast<std::size_t>(up) * nx_ + i];
        const double width = std::min(std::abs(vel) * dt / hy_, 1.0);
        const double y0 = vel > 0.0 ? 1.0 - width : 0.0;
        const double y1 = vel > 0.0 ? 1.0 : width;
        std::vector<vof::Point> poly = vof::fluidPolygon(r);
        poly = vof::clipAxis(poly, 1, y0, -1.0);
        poly = vof::clipAxis(poly, 1, y1, 1.0);
        flux[index] =
            (vel > 0.0 ? 1.0 : -1.0) * vof::polygonArea(poly) * hx_ * hy_;
      }
    }
  }

  std::vector<vof::Reconstruction> wallReconstructions() const {
    std::vector<vof::Reconstruction> rec(static_cast<std::size_t>(nx_) * ny_);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const double nx = -0.5 * (cval(i + 1, j) - cval(i - 1, j));
        const double ny = -0.5 * (cval(i, j + 1) - cval(i, j - 1));
        rec[static_cast<std::size_t>(j) * nx_ + i] =
            vof::reconstruct(c_[static_cast<std::size_t>(j) * nx_ + i], nx, ny);
      }
    }
    return rec;
  }

  void sweepXPlic(double dt) {
    const auto rec = wallReconstructions();
    std::vector<double> flux(c_.size(), 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double vel = u_[index];
        if (vel == 0.0) {
          continue;
        }
        const int up = vel > 0.0 ? i - 1 : i;
        const auto& r = rec[static_cast<std::size_t>(j) * nx_ + up];
        const double width = std::min(std::abs(vel) * dt / hx_, 1.0);
        const double x0 = vel > 0.0 ? 1.0 - width : 0.0;
        const double x1 = vel > 0.0 ? 1.0 : width;
        std::vector<vof::Point> poly = vof::fluidPolygon(r);
        poly = vof::clipAxis(poly, 0, x0, -1.0);
        poly = vof::clipAxis(poly, 0, x1, 1.0);
        flux[index] =
            (vel > 0.0 ? 1.0 : -1.0) * vof::polygonArea(poly) * hx_ * hy_;
      }
    }
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double left = flux[index];
        const double right = (i + 1 == nx_) ? 0.0 : flux[index + 1];
        c_[index] -= (right - left) / (hx_ * hy_);
      }
    }
  }

  void sweepYPlic(double dt) {
    const auto rec = wallReconstructions();
    std::vector<double> flux(c_.size(), 0.0);
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double vel = v_[index];
        if (vel == 0.0) {
          continue;
        }
        const int up = vel > 0.0 ? j - 1 : j;
        const auto& r = rec[static_cast<std::size_t>(up) * nx_ + i];
        const double width = std::min(std::abs(vel) * dt / hy_, 1.0);
        const double y0 = vel > 0.0 ? 1.0 - width : 0.0;
        const double y1 = vel > 0.0 ? 1.0 : width;
        std::vector<vof::Point> poly = vof::fluidPolygon(r);
        poly = vof::clipAxis(poly, 1, y0, -1.0);
        poly = vof::clipAxis(poly, 1, y1, 1.0);
        flux[index] =
            (vel > 0.0 ? 1.0 : -1.0) * vof::polygonArea(poly) * hx_ * hy_;
      }
    }
    for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double low = flux[index];
        const double high = (j + 1 == ny_) ? 0.0 : flux[index + nx_];
        c_[index] -= (high - low) / (hx_ * hy_);
      }
    }
  }

 public:
  double maxSpeed() const {
    double result = 0.0;
    for (double x : u_) {
      result = std::max(result, std::abs(x));
    }
    for (double x : v_) {
      result = std::max(result, std::abs(x));
    }
    return result;
  }

 private:
  int nx_;
  int ny_;
  double hx_;
  double hy_;
  std::vector<double> c_;
  std::vector<double> u_;
  std::vector<double> v_;
  std::vector<double> p_;
};

}  // namespace twophase
}  // namespace saturn

#endif  // SATURN_HAVE_FFTW
