#pragma once

#ifdef SATURN_HAVE_FFTW

#include "numerics/twophase2d.hpp"
#include "numerics/vof2d.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace saturn {
namespace twophase {

// Wall-bounded two-phase solver prototype.
//
// Closed box [0,1]^2, no-slip/no-penetration walls on all sides, gravity in
// -y, variable density rho(c), wall-aware PLIC transport, balanced CSF, and a
// conservative first-order upwind momentum flux. The pressure projection uses
// the variable-coefficient Neumann PCG solver. The momentum flux is
// deliberately dissipative to remain stable at high density ratios; a
// conservative variable-density (rho u) formulation is the production step.
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
    const std::size_t total = static_cast<std::size_t>(nx_) * ny_;
    std::vector<double> rho(total, 0.0);
    for (std::size_t q = 0; q < total; ++q) {
      rho[q] = rho_l * c_[q] + rho_g * (1.0 - c_[q]);
    }
    std::vector<double> bx(total, 0.0);
    std::vector<double> by(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        if (i > 0) {
          const double rl = rho[static_cast<std::size_t>(j) * nx_ + i - 1];
          bx[index] = (rl + rho[index]) / (2.0 * rl * rho[index]);
        }
        if (j > 0) {
          const double rd = rho[static_cast<std::size_t>(j - 1) * nx_ + i];
          by[index] = (rd + rho[index]) / (2.0 * rd * rho[index]);
        }
      }
    }

    // Cell-centred velocities for a conservative first-order upwind flux.
    std::vector<double> uc(total, 0.0);
    std::vector<double> vc(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double u_right = (i + 1 == nx_) ? 0.0 : u_[index + 1];
        const double v_high = (j + 1 == ny_) ? 0.0 : v_[index + nx_];
        uc[index] = 0.5 * (u_[index] + u_right);
        vc[index] = 0.5 * (v_[index] + v_high);
      }
    }
    auto upwind = [](double velocity, double left, double right) {
      return velocity * (velocity > 0.0 ? left : right);
    };
    std::vector<double> du(total, 0.0);
    std::vector<double> dv(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const std::size_t xm = index - 1;
        const std::size_t xp = (i + 1 == nx_) ? index : index + 1;
        const double u_here = u_[index];
        const double u_left = u_[xm];
        const double u_right = (i + 1 == nx_) ? 0.0 : u_[xp];
        const double u_low = (j == 0) ? -u_here : u_[index - nx_];
        const double u_high = (j + 1 == ny_) ? -u_here : u_[index + nx_];
        const std::size_t ymb = (j == 0) ? index : index - nx_;
        const std::size_t yab = (j + 1 == ny_) ? index : index + nx_;
        const double u_below = uc[ymb];
        const double u_above = uc[yab];
        const double f_left =
            (i == 0) ? 0.0
                     : upwind(0.5 * (uc[xm] + uc[index]), uc[xm], uc[index]);
        const double f_right =
            (i + 1 == nx_)
                ? 0.0
                : upwind(0.5 * (uc[index] + uc[xp]), uc[index], uc[xp]);
        const double g_low =
            (j == 0) ? 0.0
                     : upwind(0.5 * (vc[ymb] + vc[index]), u_below, uc[index]);
        const double g_high =
            (j + 1 == ny_)
                ? 0.0
                : upwind(0.5 * (vc[index] + vc[yab]), uc[index], u_above);
        const double lap = (u_right - 2.0 * u_here + u_left) / (hx_ * hx_) +
                           (u_high - 2.0 * u_here + u_low) / (hy_ * hy_);
        du[index] = -((f_right - f_left) / hx_ + (g_high - g_low) / hy_) +
                    mu * bx[index] * lap;
      }
    }
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const std::size_t ym = index - nx_;
        const std::size_t yp = (j + 1 == ny_) ? index : index + nx_;
        const double v_here = v_[index];
        const double v_low = v_[ym];
        const double v_high = (j + 1 == ny_) ? 0.0 : v_[yp];
        const double v_left = (i == 0) ? -v_here : v_[index - 1];
        const double v_right = (i + 1 == nx_) ? -v_here : v_[index + 1];
        const std::size_t xmb = (i == 0) ? index : index - 1;
        const std::size_t xab = (i + 1 == nx_) ? index : index + 1;
        const double f_left =
            (i == 0) ? 0.0
                     : upwind(0.5 * (uc[xmb] + uc[index]), vc[xmb], vc[index]);
        const double f_right =
            (i + 1 == nx_)
                ? 0.0
                : upwind(0.5 * (uc[index] + uc[xab]), vc[index], vc[xab]);
        const double g_low =
            (j == 0) ? 0.0
                     : upwind(0.5 * (vc[ym] + vc[index]), vc[ym], vc[index]);
        const double g_high =
            (j + 1 == ny_)
                ? 0.0
                : upwind(0.5 * (vc[index] + vc[yp]), vc[index], vc[yp]);
        const double lap = (v_right - 2.0 * v_here + v_left) / (hx_ * hx_) +
                           (v_high - 2.0 * v_here + v_low) / (hy_ * hy_);
        dv[index] = -((f_right - f_left) / hx_ + (g_high - g_low) / hy_) +
                    mu * by[index] * lap + gy;
      }
    }

    std::vector<double> fx;
    std::vector<double> fy;
    csf_forces(sigma, fx, fy);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        u_[index] += dt * (du[index] + bx[index] * fx[index]);
      }
      u_[static_cast<std::size_t>(j) * nx_] = 0.0;
    }
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        v_[index] += dt * (dv[index] + by[index] * fy[index]);
      }
    }
    for (int i = 0; i < nx_; ++i) {
      v_[i] = 0.0;
    }

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
      VariablePoisson2D solver(nx_, ny_, hx_, hy_, bx, by, false, false);
      solver.solve(div, p_);
      for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
          const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
          u_[index] -= dt * bx[index] * (p_[index] - p_[index - 1]) / hx_;
        }
        u_[static_cast<std::size_t>(j) * nx_] = 0.0;
      }
      for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
          const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
          v_[index] -= dt * by[index] * (p_[index] - p_[index - nx_]) / hy_;
        }
      }
      for (int i = 0; i < nx_; ++i) {
        v_[i] = 0.0;
      }
    }

    advect_plic(dt);
  }

  // Wall-aware PLIC advection: Strang split x/y with geometric face fluxes.
  void advect_plic(double dt) {
    sweep_x_plic(0.5 * dt);
    sweep_y_plic(dt);
    sweep_x_plic(0.5 * dt);
    restore_bounds();
  }

 private:
  std::size_t cell_index(int i, int j) const {
    const int ii = std::min(std::max(i, 0), nx_ - 1);
    const int jj = std::min(std::max(j, 0), ny_ - 1);
    return static_cast<std::size_t>(jj) * nx_ + ii;
  }

  // Balanced CSF force on faces: sigma * kappa * grad(c), with a smoothed
  // volume fraction and mirror (zero-normal-gradient) ghosts at the walls.
  void csf_forces(double sigma, std::vector<double>& fx,
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
          buffer[cell_index(i, j)] = 0.25 * smoothed[cell_index(i - 1, j)] +
                                     0.5 * smoothed[cell_index(i, j)] +
                                     0.25 * smoothed[cell_index(i + 1, j)];
        }
      }
      std::vector<double> next(buffer);
      for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
          next[cell_index(i, j)] = 0.25 * buffer[cell_index(i, j - 1)] +
                                   0.5 * buffer[cell_index(i, j)] +
                                   0.25 * buffer[cell_index(i, j + 1)];
        }
      }
      smoothed = next;
    }
    auto grad = [&](int i, int j, double& gx, double& gy) {
      gx = 0.5 *
           (smoothed[cell_index(i + 1, j)] - smoothed[cell_index(i - 1, j)]) /
           hx_;
      gy = 0.5 *
           (smoothed[cell_index(i, j + 1)] - smoothed[cell_index(i, j - 1)]) /
           hy_;
    };
    auto normal_x = [&](int i, int j) {
      double gx = 0.0;
      double gy = 0.0;
      grad(i, j, gx, gy);
      const double norm = std::sqrt(gx * gx + gy * gy);
      return norm > 1.0e-10 ? gx / norm : 0.0;
    };
    auto normal_y = [&](int i, int j) {
      double gx = 0.0;
      double gy = 0.0;
      grad(i, j, gx, gy);
      const double norm = std::sqrt(gx * gx + gy * gy);
      return norm > 1.0e-10 ? gy / norm : 0.0;
    };
    std::vector<double> kappa(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const double div =
            0.5 * (normal_x(i + 1, j) - normal_x(i - 1, j)) / hx_ +
            0.5 * (normal_y(i, j + 1) - normal_y(i, j - 1)) / hy_;
        kappa[cell_index(i, j)] = -div;
      }
    }
    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = cell_index(i, j);
        const std::size_t left = cell_index(i - 1, j);
        const double kx = 0.5 * (kappa[index] + kappa[left]);
        fx[index] = sigma * kx * (c_[index] - c_[left]) / hx_;
      }
    }
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = cell_index(i, j);
        const std::size_t low = cell_index(i, j - 1);
        const double ky = 0.5 * (kappa[index] + kappa[low]);
        fy[index] = sigma * ky * (c_[index] - c_[low]) / hy_;
      }
    }
  }

  // Strang splitting can leave small overshoots at sharp interfaces even for a
  // discretely divergence-free velocity. Clamp and redistribute the resulting
  // mass error over available capacity so the update stays conservative and
  // bounded (the same construction as vof::redistribute, for a raw vector).
  void restore_bounds(int sweeps = 8) {
    const double cell_volume = hx_ * hy_;
    double target = 0.0;
    for (double value : c_) {
      target += value * cell_volume;
    }
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      for (double& value : c_) {
        value = std::min(1.0, std::max(0.0, value));
      }
      double current = 0.0;
      for (double value : c_) {
        current += value * cell_volume;
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
      if (capacity * cell_volume < std::abs(delta)) {
        // Interface cells cannot absorb the correction; fall back to the whole
        // domain.
        capacity = 0.0;
        for (double value : c_) {
          capacity += (delta > 0.0) ? (1.0 - value) : value;
        }
      }
      if (capacity * cell_volume <= 0.0) {
        break;
      }
      const double weight = delta / (capacity * cell_volume);
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

  std::vector<vof::Reconstruction> wall_reconstructions() const {
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

  void sweep_x_plic(double dt) {
    const auto rec = wall_reconstructions();
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
        std::vector<vof::Point> poly = vof::fluid_polygon(r);
        poly = vof::clip_axis(poly, 0, x0, -1.0);
        poly = vof::clip_axis(poly, 0, x1, 1.0);
        flux[index] =
            (vel > 0.0 ? 1.0 : -1.0) * vof::polygon_area(poly) * hx_ * hy_;
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

  void sweep_y_plic(double dt) {
    const auto rec = wall_reconstructions();
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
        std::vector<vof::Point> poly = vof::fluid_polygon(r);
        poly = vof::clip_axis(poly, 1, y0, -1.0);
        poly = vof::clip_axis(poly, 1, y1, 1.0);
        flux[index] =
            (vel > 0.0 ? 1.0 : -1.0) * vof::polygon_area(poly) * hx_ * hy_;
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
  double max_speed() const {
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
