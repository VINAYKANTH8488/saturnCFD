#pragma once

#ifdef SATURN_HAVE_FFTW

#include <fftw3.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "numerics/poisson.hpp"

namespace saturn {
namespace poisson {

// Fast Poisson solver for the channel configuration:
//
//   x, z: periodic        y: homogeneous Neumann walls
//
// The x/z directions are diagonalized with 2D FFTs; for every Fourier mode a
// tridiagonal system in y is solved with the Thomas algorithm. This is much
// faster than multigrid for the wall-bounded channel and makes long turbulent
// runs practical. It is host-side (FFTW plus a serial tridiagonal solve).

template <class ExecSpace>
class ChannelFftPoisson {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using view_type = Kokkos::View<double***, Kokkos::LayoutRight, memory_space>;

  ChannelFftPoisson(const std::array<int, 3>& cells,
                    const std::array<double, 3>& spacing)
      : n0_(cells[0]), n1_(cells[1]), n2_(cells[2]), h_(spacing) {
    plane_ = n0_ * n2_;
    in_ = fftw_alloc_complex(static_cast<std::size_t>(plane_));
    out_ = fftw_alloc_complex(static_cast<std::size_t>(plane_));
    forward_ =
        fftw_plan_dft_2d(n2_, n0_, in_, out_, FFTW_FORWARD, FFTW_ESTIMATE);
    inverse_ =
        fftw_plan_dft_2d(n2_, n0_, in_, out_, FFTW_BACKWARD, FFTW_ESTIMATE);
    spectrum_.resize(static_cast<std::size_t>(n1_) * plane_);
  }

  ~ChannelFftPoisson() {
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

  ChannelFftPoisson(const ChannelFftPoisson&) = delete;
  ChannelFftPoisson& operator=(const ChannelFftPoisson&) = delete;

  Result solve(view_type x, view_type b, const Options& options) {
    (void)options;
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

    // Forward transform each y plane.
    for (int j = 0; j < n1_; ++j) {
      for (int k = 0; k < n2_; ++k) {
        for (int i = 0; i < n0_; ++i) {
          const int index = k * n0_ + i;
          in_[index][0] = hb(k, j, i);
          in_[index][1] = 0.0;
        }
      }
      fftw_execute_dft(forward_, in_, out_);
      for (int m = 0; m < plane_; ++m) {
        spectrum_[static_cast<std::size_t>(j) * plane_ + m] =
            std::complex<double>(out_[m][0], out_[m][1]);
      }
    }

    const double hy2 = h_[1] * h_[1];
    std::vector<double> lower(n1_), diag(n1_), upper(n1_), cp(n1_);
    std::vector<std::complex<double>> rhs(n1_), sol(n1_);

    for (int kz = 0; kz < n2_; ++kz) {
      const double kzz = 2.0 * std::sin(kPi * kz / n2_) / h_[2];
      for (int kx = 0; kx < n0_; ++kx) {
        const double kxx = 2.0 * std::sin(kPi * kx / n0_) / h_[0];
        const double lam = -(kxx * kxx + kzz * kzz);
        const int mode = kz * n0_ + kx;
        if (kx == 0 && kz == 0) {
          for (int j = 0; j < n1_; ++j) {
            spectrum_[static_cast<std::size_t>(j) * plane_ + mode] = 0.0;
          }
          continue;
        }
        for (int j = 0; j < n1_; ++j) {
          lower[j] = (j == 0) ? 0.0 : 1.0 / hy2;
          upper[j] = (j == n1_ - 1) ? 0.0 : 1.0 / hy2;
          diag[j] = -2.0 / hy2 + lam;
          if (j == 0 || j == n1_ - 1) {
            diag[j] += 1.0 / hy2;
          }
          rhs[j] = spectrum_[static_cast<std::size_t>(j) * plane_ + mode];
        }
        cp[0] = upper[0] / diag[0];
        for (int j = 1; j < n1_; ++j) {
          const double denom = diag[j] - lower[j] * cp[j - 1];
          cp[j] = upper[j] / denom;
        }
        sol[0] = rhs[0] / diag[0];
        for (int j = 1; j < n1_; ++j) {
          const double denom = diag[j] - lower[j] * cp[j - 1];
          sol[j] = (rhs[j] - lower[j] * sol[j - 1]) / denom;
        }
        for (int j = n1_ - 2; j >= 0; --j) {
          sol[j] -= cp[j] * sol[j + 1];
        }
        for (int j = 0; j < n1_; ++j) {
          spectrum_[static_cast<std::size_t>(j) * plane_ + mode] = sol[j];
        }
      }
    }

    // Inverse transform each y plane.
    auto hx = Kokkos::create_mirror_view(x);
    const double scale = 1.0 / static_cast<double>(plane_);
    for (int j = 0; j < n1_; ++j) {
      for (int m = 0; m < plane_; ++m) {
        const auto value = spectrum_[static_cast<std::size_t>(j) * plane_ + m];
        in_[m][0] = value.real();
        in_[m][1] = value.imag();
      }
      fftw_execute_dft(inverse_, in_, out_);
      for (int k = 0; k < n2_; ++k) {
        for (int i = 0; i < n0_; ++i) {
          const int m = k * n0_ + i;
          hx(k, j, i) = out_[m][0] * scale;
        }
      }
    }
    Kokkos::deep_copy(x, hx);

    Result result;
    result.cycles = 1;
    result.converged = true;
    return result;
  }

  int levels() const { return 1; }

 private:
  static constexpr double kPi = 3.14159265358979323846;
  int n0_ = 1;
  int n1_ = 1;
  int n2_ = 1;
  int plane_ = 1;
  std::array<double, 3> h_{{1.0, 1.0, 1.0}};
  fftw_complex* in_ = nullptr;
  fftw_complex* out_ = nullptr;
  fftw_plan forward_ = nullptr;
  fftw_plan inverse_ = nullptr;
  std::vector<std::complex<double>> spectrum_;
};

}  // namespace poisson
}  // namespace saturn

#endif  // SATURN_HAVE_FFTW
