#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "core/parallel.hpp"

namespace saturn {
namespace poisson {

// Cell-centered constant-coefficient Poisson solver on a periodic box:
//
//   laplacian(p) = f,   mean(f) = 0.
//
// The discretization is the standard 7-point finite-volume Laplacian. The
// solver is a geometric-multigrid V-cycle with red-black Gauss-Seidel
// smoothing, full-weighting restriction, and trilinear prolongation. This is
// the first building block for the Phase 3 projection method; variable
// coefficients, non-periodic boundaries, and MPI are later additions.

struct Options {
  int max_cycles = 100;
  int pre_smooth = 2;
  int post_smooth = 2;
  int coarse_smooth = 60;
  double tolerance = 1.0e-8;
  bool verbose = false;
};

struct Result {
  int cycles = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  bool converged = false;
};

template <class ExecSpace>
class GeometricMultigrid {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using view_type = Kokkos::View<double***, Kokkos::LayoutRight, memory_space>;

  GeometricMultigrid(const std::array<int, 3>& cells,
                     const std::array<double, 3>& spacing) {
    std::array<int, 3> n = cells;
    std::array<double, 3> h = spacing;
    while (true) {
      Level level;
      level.n = n;
      level.h = h;
      level.x = view_type("poisson_x", n[2], n[1], n[0]);
      level.b = view_type("poisson_b", n[2], n[1], n[0]);
      level.r = view_type("poisson_r", n[2], n[1], n[0]);
      levels_.push_back(level);

      const bool can_coarsen = n[0] % 2 == 0 && n[1] % 2 == 0 &&
                               n[2] % 2 == 0 && n[0] >= 6 && n[1] >= 6 &&
                               n[2] >= 6;
      if (!can_coarsen) {
        break;
      }
      for (int d = 0; d < 3; ++d) {
        n[d] /= 2;
        h[d] *= 2.0;
      }
    }
  }

  // Solve laplacian(x) = b with x as the initial guess. Both x and b must be
  // zero-mean on a periodic domain; they are projected internally.
  Result solve(view_type x, view_type b, const Options& options) {
    Kokkos::deep_copy(levels_[0].x, x);
    Kokkos::deep_copy(levels_[0].b, b);
    project_mean_impl(levels_[0].x);
    project_mean_impl(levels_[0].b);

    Result result;
    result.initial_residual = residual_norm(0);
    result.final_residual = result.initial_residual;
    const double target = options.tolerance * result.initial_residual;
    if (result.initial_residual == 0.0) {
      result.converged = true;
    }

    for (int cycle = 0; cycle < options.max_cycles; ++cycle) {
      if (result.final_residual <= target) {
        result.converged = true;
        break;
      }
      vcycle(0, options);
      project_mean_impl(levels_[0].x);
      result.cycles = cycle + 1;
      result.final_residual = residual_norm(0);
    }
    if (result.final_residual <= target) {
      result.converged = true;
    }
    Kokkos::deep_copy(x, levels_[0].x);
    return result;
  }

  int levels() const { return static_cast<int>(levels_.size()); }

 private:
  struct Level {
    std::array<int, 3> n{{0, 0, 0}};
    std::array<double, 3> h{{1.0, 1.0, 1.0}};
    view_type x{};
    view_type b{};
    view_type r{};
  };

  KOKKOS_INLINE_FUNCTION
  static int wrap(int i, int n) { return ((i % n) + n) % n; }

  void vcycle(int level_index, const Options& options) {
    Level& level = levels_[static_cast<std::size_t>(level_index)];
    if (level_index + 1 == static_cast<int>(levels_.size())) {
      smooth(level, options.coarse_smooth);
      return;
    }

    smooth(level, options.pre_smooth);
    compute_residual(level);

    Level& coarse = levels_[static_cast<std::size_t>(level_index + 1)];
    restrict(level, coarse);
    Kokkos::deep_copy(coarse.x, 0.0);
    vcycle(level_index + 1, options);
    prolong_add(coarse, level);

    smooth(level, options.post_smooth);
  }

  void smooth(const Level& level, int sweeps) const {
    const int n0 = level.n[0];
    const int n1 = level.n[1];
    const int n2 = level.n[2];
    const double hx2 = level.h[0] * level.h[0];
    const double hy2 = level.h[1] * level.h[1];
    const double hz2 = level.h[2] * level.h[2];
    const double diag = -2.0 / hx2 - 2.0 / hy2 - 2.0 / hz2;
    auto x = level.x;
    auto b = level.b;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      for (int color = 0; color < 2; ++color) {
        Kokkos::parallel_for(
            "poisson_smooth", policy, KOKKOS_LAMBDA(int k, int j, int i) {
              if (((i + j + k) & 1) != color) {
                return;
              }
              const int im = (i == 0) ? n0 - 1 : i - 1;
              const int ip = (i + 1 == n0) ? 0 : i + 1;
              const int jm = (j == 0) ? n1 - 1 : j - 1;
              const int jp = (j + 1 == n1) ? 0 : j + 1;
              const int km = (k == 0) ? n2 - 1 : k - 1;
              const int kp = (k + 1 == n2) ? 0 : k + 1;
              const double off = (x(k, j, im) + x(k, j, ip)) / hx2 +
                                 (x(k, jm, i) + x(k, jp, i)) / hy2 +
                                 (x(km, j, i) + x(kp, j, i)) / hz2;
              x(k, j, i) = (b(k, j, i) - off) / diag;
            });
      }
    }
  }

  void compute_residual(const Level& level) const {
    const int n0 = level.n[0];
    const int n1 = level.n[1];
    const int n2 = level.n[2];
    const double hx2 = level.h[0] * level.h[0];
    const double hy2 = level.h[1] * level.h[1];
    const double hz2 = level.h[2] * level.h[2];
    auto x = level.x;
    auto b = level.b;
    auto r = level.r;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
    Kokkos::parallel_for(
        "poisson_residual", policy, KOKKOS_LAMBDA(int k, int j, int i) {
          const int im = (i == 0) ? n0 - 1 : i - 1;
          const int ip = (i + 1 == n0) ? 0 : i + 1;
          const int jm = (j == 0) ? n1 - 1 : j - 1;
          const int jp = (j + 1 == n1) ? 0 : j + 1;
          const int km = (k == 0) ? n2 - 1 : k - 1;
          const int kp = (k + 1 == n2) ? 0 : k + 1;
          const double lap =
              (x(k, j, ip) - 2.0 * x(k, j, i) + x(k, j, im)) / hx2 +
              (x(k, jp, i) - 2.0 * x(k, j, i) + x(k, jm, i)) / hy2 +
              (x(kp, j, i) - 2.0 * x(k, j, i) + x(km, j, i)) / hz2;
          r(k, j, i) = b(k, j, i) - lap;
        });
  }

  void restrict(const Level& fine, Level& coarse) const {
    const int n0 = fine.n[0];
    const int n1 = fine.n[1];
    const int n2 = fine.n[2];
    const int c0 = coarse.n[0];
    const int c1 = coarse.n[1];
    const int c2 = coarse.n[2];
    auto r = fine.r;
    auto b = coarse.b;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{c2, c1, c0}});
    Kokkos::parallel_for(
        "poisson_restrict", policy, KOKKOS_LAMBDA(int ck, int cj, int ci) {
          double sum = 0.0;
          for (int dk = -1; dk <= 1; ++dk) {
            const double wz = (dk == 0) ? 0.5 : 0.25;
            const int k = wrap(2 * ck + dk, n2);
            for (int dj = -1; dj <= 1; ++dj) {
              const double wy = (dj == 0) ? 0.5 : 0.25;
              const int j = wrap(2 * cj + dj, n1);
              for (int di = -1; di <= 1; ++di) {
                const double wx = (di == 0) ? 0.5 : 0.25;
                const int i = wrap(2 * ci + di, n0);
                sum += wx * wy * wz * r(k, j, i);
              }
            }
          }
          b(ck, cj, ci) = sum;
        });
  }

  void prolong_add(const Level& coarse, Level& fine) const {
    const int c0 = coarse.n[0];
    const int c1 = coarse.n[1];
    const int c2 = coarse.n[2];
    const int n0 = fine.n[0];
    const int n1 = fine.n[1];
    const int n2 = fine.n[2];
    auto xc = coarse.x;
    auto xf = fine.x;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
    Kokkos::parallel_for(
        "poisson_prolong", policy, KOKKOS_LAMBDA(int k, int j, int i) {
          const int ci = i / 2;
          const int cj = j / 2;
          const int ck = k / 2;
          const int i0 = (i & 1) ? ci : wrap(ci - 1, c0);
          const int i1 = (i & 1) ? wrap(ci + 1, c0) : ci;
          const int j0 = (j & 1) ? cj : wrap(cj - 1, c1);
          const int j1 = (j & 1) ? wrap(cj + 1, c1) : cj;
          const int k0 = (k & 1) ? ck : wrap(ck - 1, c2);
          const int k1 = (k & 1) ? wrap(ck + 1, c2) : ck;
          const double wi0 = (i & 1) ? 0.75 : 0.25;
          const double wi1 = 1.0 - wi0;
          const double wj0 = (j & 1) ? 0.75 : 0.25;
          const double wj1 = 1.0 - wj0;
          const double wk0 = (k & 1) ? 0.75 : 0.25;
          const double wk1 = 1.0 - wk0;
          const double value =
              wk0 * (wj0 * (wi0 * xc(k0, j0, i0) + wi1 * xc(k0, j0, i1)) +
                     wj1 * (wi0 * xc(k0, j1, i0) + wi1 * xc(k0, j1, i1))) +
              wk1 * (wj0 * (wi0 * xc(k1, j0, i0) + wi1 * xc(k1, j0, i1)) +
                     wj1 * (wi0 * xc(k1, j1, i0) + wi1 * xc(k1, j1, i1)));
          xf(k, j, i) += value;
        });
  }

  double residual_norm(int level_index) const {
    const Level& level = levels_[static_cast<std::size_t>(level_index)];
    const long long cells =
        static_cast<long long>(level.n[0]) * level.n[1] * level.n[2];
    double sum = 0.0;
    const auto policy = make_mdrange3<ExecSpace>(
        Int3{{0, 0, 0}}, Int3{{level.n[2], level.n[1], level.n[0]}});
    auto r = level.r;
    compute_residual(level);
    Kokkos::parallel_reduce(
        "poisson_residual_norm", policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
          acc += r(k, j, i) * r(k, j, i);
        },
        Kokkos::Sum<double>{sum});
    return std::sqrt(sum / static_cast<double>(cells));
  }

  static void project_mean_impl(const view_type& v) {
    const int n0 = static_cast<int>(v.extent(2));
    const int n1 = static_cast<int>(v.extent(1));
    const int n2 = static_cast<int>(v.extent(0));
    const long long cells = static_cast<long long>(n0) * n1 * n2;
    double sum = 0.0;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
    Kokkos::parallel_reduce(
        "poisson_mean", policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& acc) { acc += v(k, j, i); },
        Kokkos::Sum<double>{sum});
    const double mean = sum / static_cast<double>(cells);
    Kokkos::parallel_for(
        "poisson_project", policy,
        KOKKOS_LAMBDA(int k, int j, int i) { v(k, j, i) -= mean; });
  }

  std::vector<Level> levels_;
};

}  // namespace poisson
}  // namespace saturn
