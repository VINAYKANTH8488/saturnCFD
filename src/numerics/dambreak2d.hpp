#pragma once

#ifdef SATURN_HAVE_FFTW

#include "numerics/twophase2d.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace saturn {
namespace twophase {

// Wall-bounded two-phase dam-break prototype.
//
// Closed box [0,1]^2, no-slip/no-penetration walls on all sides, gravity in
// -y, variable density rho(c), and a first-order upwind VOF transport. The
// pressure projection uses the variable-coefficient Neumann PCG solver. This
// is a deliberately simple first wall-bounded step; PLIC/CSF/wall-aware
// momentum on the production path come next.
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
            bool project = true) {
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
        const double v_high = (j + 1 == ny_) ? 0.0 : v_[index + nx_];
        const double v_high_m = (j + 1 == ny_) ? 0.0 : v_[xm + nx_];
        const double dudx = (u_right - u_left) / (2.0 * hx_);
        const double dudy = (u_high - u_low) / (2.0 * hy_);
        const double v_at_u = 0.25 * (v_[xm] + v_[index] + v_high_m + v_high);
        const double lap = (u_right - 2.0 * u_here + u_left) / (hx_ * hx_) +
                           (u_high - 2.0 * u_here + u_low) / (hy_ * hy_);
        du[index] = -(u_here * dudx + v_at_u * dudy) + mu * bx[index] * lap;
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
        const double u_here_i = u_[index];
        const double u_ip = (i + 1 == nx_) ? 0.0 : u_[index + 1];
        const double u_jm_i = u_[ym];
        const double u_jm_ip = (i + 1 == nx_) ? 0.0 : u_[ym + 1];
        const double dvdx = (v_right - v_left) / (2.0 * hx_);
        const double dvdy = (v_high - v_low) / (2.0 * hy_);
        const double u_at_v = 0.25 * (u_jm_i + u_here_i + u_jm_ip + u_ip);
        const double lap = (v_right - 2.0 * v_here + v_left) / (hx_ * hx_) +
                           (v_high - 2.0 * v_here + v_low) / (hy_ * hy_);
        dv[index] =
            -(u_at_v * dvdx + v_here * dvdy) + mu * by[index] * lap + gy;
      }
    }

    for (int j = 0; j < ny_; ++j) {
      for (int i = 1; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        u_[index] += dt * du[index];
      }
      u_[static_cast<std::size_t>(j) * nx_] = 0.0;
    }
    for (int j = 1; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        v_[index] += dt * dv[index];
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

    // First-order upwind VOF transport with zero wall fluxes.
    std::vector<double> flux(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double vel = u_[index];
        if (i == 0 || vel == 0.0) {
          flux[index] = 0.0;
        } else {
          const double upwind = vel > 0.0 ? c_[index - 1] : c_[index];
          flux[index] = vel * upwind;
        }
      }
    }
    std::vector<double> dc(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double right = (i + 1 == nx_) ? 0.0 : flux[index + 1];
        dc[index] -= (right - flux[index]) / hx_;
      }
    }
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double vel = v_[index];
        if (j == 0 || vel == 0.0) {
          flux[index] = 0.0;
        } else {
          const double upwind = vel > 0.0 ? c_[index - nx_] : c_[index];
          flux[index] = vel * upwind;
        }
      }
    }
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double high = (j + 1 == ny_) ? 0.0 : flux[index + nx_];
        dc[index] -= (high - flux[index]) / hy_;
      }
    }
    for (std::size_t q = 0; q < total; ++q) {
      c_[q] = std::min(1.0, std::max(0.0, c_[q] + dt * dc[q]));
    }
  }

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
