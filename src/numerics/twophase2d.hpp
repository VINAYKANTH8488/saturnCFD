#pragma once

#ifdef SATURN_HAVE_FFTW

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "numerics/vof2d.hpp"

namespace saturn {
namespace twophase {

// Two-dimensional periodic MAC projection solver with VOF and a balanced
// CSF surface-tension force. Constant density in this prototype.
//
//   u(i,j): x-face at x = i hx, cell-centered in y
//   v(i,j): y-face at y = j hy, cell-centered in x
//   p(i,j): cell center
//
// The CSF force is evaluated on faces from the jump in c,
//   f_x = sigma kappa_face (c_R - c_L)/hx,
// which is a discrete gradient of sigma*kappa*c when kappa is constant. The
// projection then balances it with pressure and leaves a static bubble with
// zero velocity up to solver tolerance (balanced-force CSF).

class FftPoisson2D {
 public:
  FftPoisson2D(int nx, int ny, double hx, double hy)
      : nx_(nx), ny_(ny), hx_(hx), hy_(hy) {
    in_ = fftw_alloc_complex(static_cast<std::size_t>(nx) * ny);
    out_ = fftw_alloc_complex(static_cast<std::size_t>(nx) * ny);
    forward_ = fftw_plan_dft_2d(ny, nx, in_, out_, FFTW_FORWARD, FFTW_ESTIMATE);
    inverse_ =
        fftw_plan_dft_2d(ny, nx, in_, out_, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  ~FftPoisson2D() {
    if (forward_ != nullptr) {
      fftw_destroy_plan(forward_);
    }
    if (inverse_ != nullptr) {
      fftw_destroy_plan(inverse_);
    }
    if (in_ != nullptr) {
      fftw_free(in_);
    }
    if (out_ != nullptr) {
      fftw_free(out_);
    }
  }
  FftPoisson2D(const FftPoisson2D&) = delete;
  FftPoisson2D& operator=(const FftPoisson2D&) = delete;

  // Solve laplacian(p) = rhs with zero mean; periodic boundaries.
  void solve(const std::vector<double>& rhs, std::vector<double>& p) {
    const int total = nx_ * ny_;
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int index = j * nx_ + i;
        in_[index][0] = rhs[static_cast<std::size_t>(index)];
        in_[index][1] = 0.0;
      }
    }
    fftw_execute_dft(forward_, in_, out_);
    for (int j = 0; j < ny_; ++j) {
      const double ky = 2.0 * std::sin(kPi * j / ny_) / hy_;
      for (int i = 0; i < nx_; ++i) {
        const double kx = 2.0 * std::sin(kPi * i / nx_) / hx_;
        const int index = j * nx_ + i;
        const double lambda = -(kx * kx + ky * ky);
        double re = 0.0;
        double im = 0.0;
        if (lambda != 0.0) {
          re = out_[index][0] / lambda;
          im = out_[index][1] / lambda;
        }
        in_[index][0] = re;
        in_[index][1] = im;
      }
    }
    fftw_execute_dft(inverse_, in_, out_);
    p.assign(static_cast<std::size_t>(total), 0.0);
    const double scale = 1.0 / static_cast<double>(total);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int index = j * nx_ + i;
        p[static_cast<std::size_t>(index)] = out_[index][0] * scale;
      }
    }
  }

 private:
  static constexpr double kPi = 3.14159265358979323846;
  int nx_;
  int ny_;
  double hx_;
  double hy_;
  fftw_complex* in_ = nullptr;
  fftw_complex* out_ = nullptr;
  fftw_plan forward_ = nullptr;
  fftw_plan inverse_ = nullptr;
};

// Variable-coefficient periodic Poisson solver:
//
//   div(beta grad(p)) = rhs,   beta = 1/rho at faces.
//
// Matrix-free PCG with a Jacobi preconditioner. Used for the variable-density
// projection.
class VariablePoisson2D {
 public:
  VariablePoisson2D(int nx, int ny, double hx, double hy,
                    const std::vector<double>& beta_x,
                    const std::vector<double>& beta_y)
      : nx_(nx), ny_(ny), hx_(hx), hy_(hy), bx_(beta_x), by_(beta_y) {}

  void apply(const std::vector<double>& p, std::vector<double>& out) const {
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int im = (i == 0) ? nx_ - 1 : i - 1;
        const int ip = (i + 1 == nx_) ? 0 : i + 1;
        const int jm = (j == 0) ? ny_ - 1 : j - 1;
        const int jp = (j + 1 == ny_) ? 0 : j + 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double xflux =
            bx_[static_cast<std::size_t>(j) * nx_ + ip] *
                (p[static_cast<std::size_t>(j) * nx_ + ip] - p[index]) -
            bx_[index] * (p[index] - p[static_cast<std::size_t>(j) * nx_ + im]);
        const double yflux =
            by_[static_cast<std::size_t>(jp) * nx_ + i] *
                (p[static_cast<std::size_t>(jp) * nx_ + i] - p[index]) -
            by_[index] * (p[index] - p[static_cast<std::size_t>(jm) * nx_ + i]);
        out[index] = xflux / (hx_ * hx_) + yflux / (hy_ * hy_);
      }
    }
  }

  void solve(const std::vector<double>& rhs, std::vector<double>& p,
             double tolerance = 1.0e-10, int max_iterations = 500) const {
    const std::size_t total = static_cast<std::size_t>(nx_) * ny_;
    p.assign(total, 0.0);
    std::vector<double> r = rhs;
    std::vector<double> diag(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int ip = (i + 1 == nx_) ? 0 : i + 1;
        const int jp = (j + 1 == ny_) ? 0 : j + 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        diag[index] =
            -(bx_[index] + bx_[static_cast<std::size_t>(j) * nx_ + ip]) /
                (hx_ * hx_) -
            (by_[index] + by_[static_cast<std::size_t>(jp) * nx_ + i]) /
                (hy_ * hy_);
      }
    }
    std::vector<double> z(total, 0.0);
    std::vector<double> dir(total, 0.0);
    std::vector<double> ap(total, 0.0);
    auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
      double sum = 0.0;
      for (std::size_t i = 0; i < total; ++i) {
        sum += a[i] * b[i];
      }
      return sum;
    };
    auto precondition = [&]() {
      for (std::size_t i = 0; i < total; ++i) {
        z[i] = r[i] / diag[i];
      }
    };
    auto residual_norm = [&]() {
      double sum = 0.0;
      for (double value : r) {
        sum += value * value;
      }
      return std::sqrt(sum / static_cast<double>(total));
    };
    const double target =
        tolerance * std::max(1.0, std::sqrt(dot(rhs, rhs) / total));
    precondition();
    dir = z;
    double rz = dot(r, z);
    for (int it = 0; it < max_iterations; ++it) {
      if (residual_norm() <= target) {
        break;
      }
      apply(dir, ap);
      const double denom = dot(dir, ap);
      if (std::abs(denom) < 1e-300) {
        break;
      }
      const double alpha = rz / denom;
      for (std::size_t i = 0; i < total; ++i) {
        p[i] += alpha * dir[i];
        r[i] -= alpha * ap[i];
      }
      precondition();
      const double rz_new = dot(r, z);
      const double beta = rz_new / rz;
      for (std::size_t i = 0; i < total; ++i) {
        dir[i] = z[i] + beta * dir[i];
      }
      rz = rz_new;
    }
  }

 private:
  int nx_;
  int ny_;
  double hx_;
  double hy_;
  std::vector<double> bx_;
  std::vector<double> by_;
};

class Solver2D {
 public:
  Solver2D(int nx, int ny, double lx = 1.0, double ly = 1.0)
      : nx_(nx),
        ny_(ny),
        hx_(lx / nx),
        hy_(ly / ny),
        c_(nx, ny, lx, ly),
        u_(static_cast<std::size_t>(nx) * ny, 0.0),
        v_(static_cast<std::size_t>(nx) * ny, 0.0),
        p_(static_cast<std::size_t>(nx) * ny, 0.0),
        poisson_(nx, ny, hx_, hy_) {}

  vof::Grid2D& c() { return c_; }
  const vof::Grid2D& c() const { return c_; }
  std::vector<double>& u() { return u_; }
  std::vector<double>& v() { return v_; }
  const std::vector<double>& p() const { return p_; }
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  double hx() const { return hx_; }
  double hy() const { return hy_; }

  // CSF force faces from a cell-centered curvature field. The force is
  // sigma*kappa_face*(c_R-c_L)/h.
  void csf_faces(const std::vector<double>& kappa, double sigma,
                 std::vector<double>& fx, std::vector<double>& fy) const {
    fx.assign(u_.size(), 0.0);
    fy.assign(u_.size(), 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int index = j * nx_ + i;
        const int im = (i == 0) ? nx_ - 1 : i - 1;
        const int jm = (j == 0) ? ny_ - 1 : j - 1;
        const double kx = 0.5 * (kappa[static_cast<std::size_t>(j) * nx_ + im] +
                                 kappa[static_cast<std::size_t>(index)]);
        const double ky = 0.5 * (kappa[static_cast<std::size_t>(jm) * nx_ + i] +
                                 kappa[static_cast<std::size_t>(index)]);
        fx[static_cast<std::size_t>(index)] =
            sigma * kx * (c_(i, j) - c_(im, j)) / hx_;
        fy[static_cast<std::size_t>(index)] =
            sigma * ky * (c_(i, j) - c_(i, jm)) / hy_;
      }
    }
  }

  // One projection step from the CSF force, with an optional explicit
  // advection/viscous term omitted in this static-bubble prototype.
  void project_force(const std::vector<double>& fx,
                     const std::vector<double>& fy, double dt,
                     double rho = 1.0) {
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int index = j * nx_ + i;
        u_[static_cast<std::size_t>(index)] +=
            dt * fx[static_cast<std::size_t>(index)] / rho;
        v_[static_cast<std::size_t>(index)] +=
            dt * fy[static_cast<std::size_t>(index)] / rho;
      }
    }
    std::vector<double> div(static_cast<std::size_t>(nx_) * ny_, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int ip = (i + 1 == nx_) ? 0 : i + 1;
        const int jp = (j + 1 == ny_) ? 0 : j + 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        div[index] =
            (u_[static_cast<std::size_t>(j) * nx_ + ip] - u_[index]) / hx_ +
            (v_[static_cast<std::size_t>(jp) * nx_ + i] - v_[index]) / hy_;
      }
    }
    for (double& d : div) {
      d = d / dt;
    }
    poisson_.solve(div, p_);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int im = (i == 0) ? nx_ - 1 : i - 1;
        const int jm = (j == 0) ? ny_ - 1 : j - 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        u_[index] -= dt *
                     (p_[index] - p_[static_cast<std::size_t>(j) * nx_ + im]) /
                     (rho * hx_);
        v_[index] -= dt *
                     (p_[index] - p_[static_cast<std::size_t>(jm) * nx_ + i]) /
                     (rho * hy_);
      }
    }
  }

  // Variable-density projection. rho(c) is interpolated arithmetically to
  // cell centers and harmonically to faces; the CSF force and the pressure
  // gradient use the same face density, which keeps a static bubble balanced.
  void project_force_variable(const std::vector<double>& fx,
                              const std::vector<double>& fy, double dt,
                              double rho_l, double rho_g) {
    const std::size_t total = static_cast<std::size_t>(nx_) * ny_;
    std::vector<double> rho_cell(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        rho_cell[index] = rho_l * c_(i, j) + rho_g * (1.0 - c_(i, j));
      }
    }
    std::vector<double> beta_x(total, 0.0);
    std::vector<double> beta_y(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int im = (i == 0) ? nx_ - 1 : i - 1;
        const int jm = (j == 0) ? ny_ - 1 : j - 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double rl = rho_cell[static_cast<std::size_t>(j) * nx_ + im];
        const double rr = rho_cell[index];
        const double rd = rho_cell[static_cast<std::size_t>(jm) * nx_ + i];
        const double ru = rho_cell[index];
        const double rho_x = 2.0 * rl * rr / (rl + rr);
        const double rho_y = 2.0 * rd * ru / (rd + ru);
        beta_x[index] = 1.0 / rho_x;
        beta_y[index] = 1.0 / rho_y;
      }
    }
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        u_[index] += dt * fx[index] * beta_x[index];
        v_[index] += dt * fy[index] * beta_y[index];
      }
    }
    std::vector<double> div(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int ip = (i + 1 == nx_) ? 0 : i + 1;
        const int jp = (j + 1 == ny_) ? 0 : j + 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        div[index] =
            (u_[static_cast<std::size_t>(j) * nx_ + ip] - u_[index]) / hx_ +
            (v_[static_cast<std::size_t>(jp) * nx_ + i] - v_[index]) / hy_;
      }
    }
    for (double& d : div) {
      d = d / dt;
    }
    VariablePoisson2D solver(nx_, ny_, hx_, hy_, beta_x, beta_y);
    solver.solve(div, p_);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int im = (i == 0) ? nx_ - 1 : i - 1;
        const int jm = (j == 0) ? ny_ - 1 : j - 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        u_[index] -= dt * beta_x[index] *
                     (p_[index] - p_[static_cast<std::size_t>(j) * nx_ + im]) /
                     hx_;
        v_[index] -= dt * beta_y[index] *
                     (p_[index] - p_[static_cast<std::size_t>(jm) * nx_ + i]) /
                     hy_;
      }
    }
  }

  // Full explicit step with variable density, advection, diffusion, gravity,
  // CSF, and projection. First-order in time; adequate for the prototype.
  void step(double dt, double sigma, const std::vector<double>& kappa,
            double rho_l, double rho_g, double mu, double gx, double gy) {
    const std::size_t total = static_cast<std::size_t>(nx_) * ny_;
    std::vector<double> rho_cell(total, 0.0);
    std::vector<double> beta_x(total, 0.0);
    std::vector<double> beta_y(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        rho_cell[index] = rho_l * c_(i, j) + rho_g * (1.0 - c_(i, j));
      }
    }
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int im = (i == 0) ? nx_ - 1 : i - 1;
        const int jm = (j == 0) ? ny_ - 1 : j - 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const double rl = rho_cell[static_cast<std::size_t>(j) * nx_ + im];
        const double rr = rho_cell[index];
        const double rd = rho_cell[static_cast<std::size_t>(jm) * nx_ + i];
        const double ru = rho_cell[index];
        beta_x[index] = (rl + rr) / (2.0 * rl * rr);
        beta_y[index] = (rd + ru) / (2.0 * rd * ru);
      }
    }

    std::vector<double> du(total, 0.0);
    std::vector<double> dv(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int im = (i == 0) ? nx_ - 1 : i - 1;
        const int ip = (i + 1 == nx_) ? 0 : i + 1;
        const int jm = (j == 0) ? ny_ - 1 : j - 1;
        const int jp = (j + 1 == ny_) ? 0 : j + 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        const std::size_t xm = static_cast<std::size_t>(j) * nx_ + im;
        const std::size_t xp = static_cast<std::size_t>(j) * nx_ + ip;
        const std::size_t ym = static_cast<std::size_t>(jm) * nx_ + i;
        const std::size_t yp = static_cast<std::size_t>(jp) * nx_ + i;
        const double u_here = u_[index];
        const double v_here = v_[index];
        const double nu_x = mu * beta_x[index];
        const double nu_y = mu * beta_y[index];
        const double dudx = (u_[xp] - u_[xm]) / (2.0 * hx_);
        const double dudy = (u_[yp] - u_[ym]) / (2.0 * hy_);
        const double v_at_u =
            0.25 *
            (v_[xm] + v_[index] + v_[static_cast<std::size_t>(jp) * nx_ + im] +
             v_[static_cast<std::size_t>(jp) * nx_ + i]);
        const double lap_u = (u_[xp] - 2.0 * u_here + u_[xm]) / (hx_ * hx_) +
                             (u_[yp] - 2.0 * u_here + u_[ym]) / (hy_ * hy_);
        du[index] = -(u_here * dudx + v_at_u * dudy) + nu_x * lap_u + gx;

        const double dvdx = (v_[xp] - v_[xm]) / (2.0 * hx_);
        const double dvdy = (v_[yp] - v_[ym]) / (2.0 * hy_);
        const double u_at_v =
            0.25 * (u_[ym] + u_[index] +
                    u_[static_cast<std::size_t>(jm) * nx_ + ip] + u_[xp]);
        const double lap_v = (v_[xp] - 2.0 * v_here + v_[xm]) / (hx_ * hx_) +
                             (v_[yp] - 2.0 * v_here + v_[ym]) / (hy_ * hy_);
        dv[index] = -(u_at_v * dvdx + v_here * dvdy) + nu_y * lap_v + gy;
      }
    }

    std::vector<double> fx;
    std::vector<double> fy;
    if (sigma > 0.0 && !kappa.empty()) {
      csf_faces(kappa, sigma, fx, fy);
    } else {
      fx.assign(total, 0.0);
      fy.assign(total, 0.0);
    }
    for (std::size_t index = 0; index < total; ++index) {
      u_[index] += dt * (du[index] + fx[index] * beta_x[index]);
      v_[index] += dt * (dv[index] + fy[index] * beta_y[index]);
    }

    std::vector<double> div(total, 0.0);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int ip = (i + 1 == nx_) ? 0 : i + 1;
        const int jp = (j + 1 == ny_) ? 0 : j + 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        div[index] =
            (u_[static_cast<std::size_t>(j) * nx_ + ip] - u_[index]) / hx_ +
            (v_[static_cast<std::size_t>(jp) * nx_ + i] - v_[index]) / hy_;
      }
    }
    for (double& d : div) {
      d = d / dt;
    }
    VariablePoisson2D solver(nx_, ny_, hx_, hy_, beta_x, beta_y);
    solver.solve(div, p_);
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int im = (i == 0) ? nx_ - 1 : i - 1;
        const int jm = (j == 0) ? ny_ - 1 : j - 1;
        const std::size_t index = static_cast<std::size_t>(j) * nx_ + i;
        u_[index] -= dt * beta_x[index] *
                     (p_[index] - p_[static_cast<std::size_t>(j) * nx_ + im]) /
                     hx_;
        v_[index] -= dt * beta_y[index] *
                     (p_[index] - p_[static_cast<std::size_t>(jm) * nx_ + i]) /
                     hy_;
      }
    }
    c_.advance(u_, v_, dt);
  }

  // Advect the volume fraction with the current face velocities.
  void advect(double dt) { c_.advance(u_, v_, dt); }

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

  double max_divergence() const {
    double result = 0.0;
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        const int ip = (i + 1 == nx_) ? 0 : i + 1;
        const int jp = (j + 1 == ny_) ? 0 : j + 1;
        const double div = (u_[static_cast<std::size_t>(j) * nx_ + ip] -
                            u_[static_cast<std::size_t>(j) * nx_ + i]) /
                               hx_ +
                           (v_[static_cast<std::size_t>(jp) * nx_ + i] -
                            v_[static_cast<std::size_t>(j) * nx_ + i]) /
                               hy_;
        result = std::max(result, std::abs(div));
      }
    }
    return result;
  }

 private:
  int nx_;
  int ny_;
  double hx_;
  double hy_;
  vof::Grid2D c_;
  std::vector<double> u_;
  std::vector<double> v_;
  std::vector<double> p_;
  FftPoisson2D poisson_;
};

}  // namespace twophase
}  // namespace saturn

#endif  // SATURN_HAVE_FFTW
