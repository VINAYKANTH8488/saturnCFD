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

// Direct FFT Poisson solver for a periodic uniform box.
//
//   laplacian(p) = f,   mean(f) = 0
//
// The right-hand side is transformed with a complex-to-complex 3D FFT, divided
// by the discrete eigenvalue
//
//   lambda_k = -4 [ sin^2(pi kx/nx)/hx^2 + ... ],
//
// and transformed back. The zero mode is set to zero (the constant null
// space). This is exact for the 7-point operator up to round-off and is much
// faster than multigrid on periodic power-of-two grids; FFTW handles arbitrary
// sizes.
//
// The implementation is host-side: the field is copied to a host mirror,
// transformed, and copied back. It is intended for CPU runs and as a reference
// for the multigrid solver; a device FFT would be the next step.

template <class ExecSpace>
class FftPoisson {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using view_type = Kokkos::View<double***, Kokkos::LayoutRight, memory_space>;

  FftPoisson(const std::array<int, 3>& cells,
             const std::array<double, 3>& spacing)
      : n0_(cells[0]), n1_(cells[1]), n2_(cells[2]), h_(spacing) {
    const long long total = static_cast<long long>(n0_) * n1_ * n2_;
    in_ = fftw_alloc_complex(static_cast<std::size_t>(total));
    out_ = fftw_alloc_complex(static_cast<std::size_t>(total));
    forward_ =
        fftw_plan_dft_3d(n2_, n1_, n0_, in_, out_, FFTW_FORWARD, FFTW_ESTIMATE);
    inverse_ = fftw_plan_dft_3d(n2_, n1_, n0_, in_, out_, FFTW_BACKWARD,
                                FFTW_ESTIMATE);
  }

  ~FftPoisson() {
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

  FftPoisson(const FftPoisson&) = delete;
  FftPoisson& operator=(const FftPoisson&) = delete;

  Result solve(view_type x, view_type b, const Options& options) {
    (void)options;
    const long long total = static_cast<long long>(n0_) * n1_ * n2_;
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
    for (int k = 0; k < n2_; ++k) {
      for (int j = 0; j < n1_; ++j) {
        for (int i = 0; i < n0_; ++i) {
          const long long index =
              (static_cast<long long>(k) * n1_ + j) * n0_ + i;
          in_[index][0] = hb(k, j, i);
          in_[index][1] = 0.0;
        }
      }
    }

    fftw_execute_dft(forward_, in_, out_);
    for (int k = 0; k < n2_; ++k) {
      const double kz = 2.0 * std::sin(kPi * k / n2_) / h_[2];
      for (int j = 0; j < n1_; ++j) {
        const double ky = 2.0 * std::sin(kPi * j / n1_) / h_[1];
        for (int i = 0; i < n0_; ++i) {
          const double kx = 2.0 * std::sin(kPi * i / n0_) / h_[0];
          const long long index =
              (static_cast<long long>(k) * n1_ + j) * n0_ + i;
          const double lambda = -(kx * kx + ky * ky + kz * kz);
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
    }

    fftw_execute_dft(inverse_, in_, out_);
    auto hx = Kokkos::create_mirror_view(x);
    const double scale = 1.0 / static_cast<double>(total);
    for (int k = 0; k < n2_; ++k) {
      for (int j = 0; j < n1_; ++j) {
        for (int i = 0; i < n0_; ++i) {
          const long long index =
              (static_cast<long long>(k) * n1_ + j) * n0_ + i;
          hx(k, j, i) = out_[index][0] * scale;
        }
      }
    }
    Kokkos::deep_copy(x, hx);

    Result result;
    result.cycles = 1;
    result.initial_residual = 0.0;
    result.final_residual = 0.0;
    result.converged = true;
    return result;
  }

  int levels() const { return 1; }

 private:
  static constexpr double kPi = 3.14159265358979323846;
  int n0_ = 1;
  int n1_ = 1;
  int n2_ = 1;
  std::array<double, 3> h_{{1.0, 1.0, 1.0}};
  fftw_complex* in_ = nullptr;
  fftw_complex* out_ = nullptr;
  fftw_plan forward_ = nullptr;
  fftw_plan inverse_ = nullptr;
};

}  // namespace poisson
}  // namespace saturn

#endif  // SATURN_HAVE_FFTW
