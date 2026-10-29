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

// Cell-centered constant-coefficient Poisson solver:
//
//   laplacian(p) = f
//
// on a uniform box with periodic, homogeneous-Neumann, or homogeneous-
// Dirichlet boundaries. The discretization is the standard 7-point
// finite-volume Laplacian with ghost cells:
//
//   Neumann:   p_ghost =  p_interior   (zero normal derivative at the face)
//   Dirichlet: p_ghost = -p_interior   (zero value at the face)
//
// Solutions with only periodic/Neumann boundaries are defined up to a
// constant; the right-hand side and initial guess are projected to zero mean.
//
// The solver is a geometric-multigrid V-cycle with red-black Gauss-Seidel
// smoothing, full-weighting restriction, and trilinear prolongation.

enum class Boundary {
  Periodic,
  Neumann,
  Dirichlet,
};

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

namespace detail {

KOKKOS_INLINE_FUNCTION
int plus_index(int i, int n, Boundary bc, int& sign) {
  sign = 1;
  if (i + 1 < n) {
    return i + 1;
  }
  if (bc == Boundary::Periodic) {
    return 0;
  }
  if (bc == Boundary::Neumann) {
    return n - 1;
  }
  sign = -1;
  return n - 1;
}

KOKKOS_INLINE_FUNCTION
int minus_index(int i, int n, Boundary bc, int& sign) {
  sign = 1;
  if (i > 0) {
    return i - 1;
  }
  if (bc == Boundary::Periodic) {
    return n - 1;
  }
  if (bc == Boundary::Neumann) {
    return 0;
  }
  sign = -1;
  return 0;
}

// Map an index that may lie one cell outside [0,n) through the boundary.
KOKKOS_INLINE_FUNCTION
int extended_index(int index, int n, Boundary low, Boundary high, int& sign) {
  sign = 1;
  if (index >= 0 && index < n) {
    return index;
  }
  if (index < 0) {
    if (low == Boundary::Periodic) {
      return ((index % n) + n) % n;
    }
    if (low == Boundary::Dirichlet) {
      sign = -1;
    }
    return 0;
  }
  if (high == Boundary::Periodic) {
    return index % n;
  }
  if (high == Boundary::Dirichlet) {
    sign = -1;
  }
  return n - 1;
}

}  // namespace detail

template <class ExecSpace>
class GeometricMultigrid {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using view_type = Kokkos::View<double***, Kokkos::LayoutRight, memory_space>;

  GeometricMultigrid(const std::array<int, 3>& cells,
                     const std::array<double, 3>& spacing)
      : GeometricMultigrid(
            cells, spacing,
            std::array<Boundary, 3>{Boundary::Periodic, Boundary::Periodic,
                                    Boundary::Periodic},
            std::array<Boundary, 3>{Boundary::Periodic, Boundary::Periodic,
                                    Boundary::Periodic}) {}

  GeometricMultigrid(const std::array<int, 3>& cells,
                     const std::array<double, 3>& spacing,
                     const std::array<Boundary, 3>& bc_low,
                     const std::array<Boundary, 3>& bc_high)
      : bc_low_(bc_low), bc_high_(bc_high) {
    has_dirichlet_ = false;
    all_periodic_ = true;
    for (int d = 0; d < 3; ++d) {
      has_dirichlet_ = has_dirichlet_ || bc_low[d] == Boundary::Dirichlet ||
                       bc_high[d] == Boundary::Dirichlet;
      all_periodic_ = all_periodic_ && bc_low[d] == Boundary::Periodic &&
                      bc_high[d] == Boundary::Periodic;
    }

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

  // Solve laplacian(x) = b with x as the initial guess. With only periodic
  // and Neumann boundaries, x and b are projected to zero mean.
  Result solve(view_type x, view_type b, const Options& options) {
    Kokkos::deep_copy(levels_[0].x, x);
    Kokkos::deep_copy(levels_[0].b, b);
    if (!has_dirichlet_) {
      project_mean_impl(levels_[0].x);
      project_mean_impl(levels_[0].b);
    }

    Result result;
    result.initial_residual = residual_norm(0);
    result.final_residual = result.initial_residual;
    const double rhs_norm = norm(levels_[0].b);
    const double target = options.tolerance * rhs_norm;
    if (result.initial_residual == 0.0 || result.initial_residual <= target) {
      result.converged = true;
    }

    for (int cycle = 0; cycle < options.max_cycles; ++cycle) {
      if (result.final_residual <= target) {
        result.converged = true;
        break;
      }
      vcycle(0, options);
      if (!has_dirichlet_) {
        project_mean_impl(levels_[0].x);
      }
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

 public:
  struct Level {
    std::array<int, 3> n{{0, 0, 0}};
    std::array<double, 3> h{{1.0, 1.0, 1.0}};
    view_type x{};
    view_type b{};
    view_type r{};
  };

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
    const Boundary bl0 = bc_low_[0];
    const Boundary bh0 = bc_high_[0];
    const Boundary bl1 = bc_low_[1];
    const Boundary bh1 = bc_high_[1];
    const Boundary bl2 = bc_low_[2];
    const Boundary bh2 = bc_high_[2];
    auto x = level.x;
    auto b = level.b;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
    if (all_periodic_) {
      for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (int color = 0; color < 2; ++color) {
          Kokkos::parallel_for(
              "poisson_smooth_periodic", policy,
              KOKKOS_LAMBDA(int k, int j, int i) {
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
      return;
    }
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      for (int color = 0; color < 2; ++color) {
        Kokkos::parallel_for(
            "poisson_smooth", policy, KOKKOS_LAMBDA(int k, int j, int i) {
              if (((i + j + k) & 1) != color) {
                return;
              }
              int sim;
              int sip;
              int sjm;
              int sjp;
              int skm;
              int skp;
              const int im = detail::minus_index(i, n0, bl0, sim);
              const int ip = detail::plus_index(i, n0, bh0, sip);
              const int jm = detail::minus_index(j, n1, bl1, sjm);
              const int jp = detail::plus_index(j, n1, bh1, sjp);
              const int km = detail::minus_index(k, n2, bl2, skm);
              const int kp = detail::plus_index(k, n2, bh2, skp);
              const double off = (sim * x(k, j, im) + sip * x(k, j, ip)) / hx2 +
                                 (sjm * x(k, jm, i) + sjp * x(k, jp, i)) / hy2 +
                                 (skm * x(km, j, i) + skp * x(kp, j, i)) / hz2;
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
    const Boundary bl0 = bc_low_[0];
    const Boundary bh0 = bc_high_[0];
    const Boundary bl1 = bc_low_[1];
    const Boundary bh1 = bc_high_[1];
    const Boundary bl2 = bc_low_[2];
    const Boundary bh2 = bc_high_[2];
    auto x = level.x;
    auto b = level.b;
    auto r = level.r;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
    if (all_periodic_) {
      Kokkos::parallel_for(
          "poisson_residual_periodic", policy,
          KOKKOS_LAMBDA(int k, int j, int i) {
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
      return;
    }
    Kokkos::parallel_for(
        "poisson_residual", policy, KOKKOS_LAMBDA(int k, int j, int i) {
          int sim;
          int sip;
          int sjm;
          int sjp;
          int skm;
          int skp;
          const int im = detail::minus_index(i, n0, bl0, sim);
          const int ip = detail::plus_index(i, n0, bh0, sip);
          const int jm = detail::minus_index(j, n1, bl1, sjm);
          const int jp = detail::plus_index(j, n1, bh1, sjp);
          const int km = detail::minus_index(k, n2, bl2, skm);
          const int kp = detail::plus_index(k, n2, bh2, skp);
          const double lap =
              (sip * x(k, j, ip) - 2.0 * x(k, j, i) + sim * x(k, j, im)) / hx2 +
              (sjp * x(k, jp, i) - 2.0 * x(k, j, i) + sjm * x(k, jm, i)) / hy2 +
              (skp * x(kp, j, i) - 2.0 * x(k, j, i) + skm * x(km, j, i)) / hz2;
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
    const Boundary bl0 = bc_low_[0];
    const Boundary bh0 = bc_high_[0];
    const Boundary bl1 = bc_low_[1];
    const Boundary bh1 = bc_high_[1];
    const Boundary bl2 = bc_low_[2];
    const Boundary bh2 = bc_high_[2];
    auto r = fine.r;
    auto b = coarse.b;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{c2, c1, c0}});
    Kokkos::parallel_for(
        "poisson_restrict", policy, KOKKOS_LAMBDA(int ck, int cj, int ci) {
          double sum = 0.0;
          for (int dk = -1; dk <= 1; ++dk) {
            const double wz = (dk == 0) ? 0.5 : 0.25;
            int sz;
            const int k = detail::extended_index(2 * ck + dk, n2, bl2, bh2, sz);
            for (int dj = -1; dj <= 1; ++dj) {
              const double wy = (dj == 0) ? 0.5 : 0.25;
              int sy;
              const int j =
                  detail::extended_index(2 * cj + dj, n1, bl1, bh1, sy);
              for (int di = -1; di <= 1; ++di) {
                const double wx = (di == 0) ? 0.5 : 0.25;
                int sx;
                const int i =
                    detail::extended_index(2 * ci + di, n0, bl0, bh0, sx);
                sum += wx * wy * wz * sx * sy * sz * r(k, j, i);
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
    const Boundary bl0 = bc_low_[0];
    const Boundary bh0 = bc_high_[0];
    const Boundary bl1 = bc_low_[1];
    const Boundary bh1 = bc_high_[1];
    const Boundary bl2 = bc_low_[2];
    const Boundary bh2 = bc_high_[2];
    auto xc = coarse.x;
    auto xf = fine.x;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
    Kokkos::parallel_for(
        "poisson_prolong", policy, KOKKOS_LAMBDA(int k, int j, int i) {
          const int ci = i / 2;
          const int cj = j / 2;
          const int ck = k / 2;
          int s0;
          int s1;
          const int i0 =
              detail::extended_index((i & 1) ? ci : ci - 1, c0, bl0, bh0, s0);
          const int i1 =
              detail::extended_index((i & 1) ? ci + 1 : ci, c0, bl0, bh0, s1);
          const double wi0 = (i & 1) ? 0.75 : 0.25;
          const double wi1 = 1.0 - wi0;
          const int j0 =
              detail::extended_index((j & 1) ? cj : cj - 1, c1, bl1, bh1, s0);
          const int j1 =
              detail::extended_index((j & 1) ? cj + 1 : cj, c1, bl1, bh1, s1);
          const double wj0 = (j & 1) ? 0.75 : 0.25;
          const double wj1 = 1.0 - wj0;
          const int k0 =
              detail::extended_index((k & 1) ? ck : ck - 1, c2, bl2, bh2, s0);
          const int k1 =
              detail::extended_index((k & 1) ? ck + 1 : ck, c2, bl2, bh2, s1);
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

  double norm(const view_type& v) const {
    const int n0 = static_cast<int>(v.extent(2));
    const int n1 = static_cast<int>(v.extent(1));
    const int n2 = static_cast<int>(v.extent(0));
    const long long cells = static_cast<long long>(n0) * n1 * n2;
    double sum = 0.0;
    const auto policy =
        make_mdrange3<ExecSpace>(Int3{{0, 0, 0}}, Int3{{n2, n1, n0}});
    Kokkos::parallel_reduce(
        "poisson_norm", policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
          acc += v(k, j, i) * v(k, j, i);
        },
        Kokkos::Sum<double>{sum});
    return std::sqrt(sum / static_cast<double>(cells));
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

 private:
  std::vector<Level> levels_;
  std::array<Boundary, 3> bc_low_{
      {Boundary::Periodic, Boundary::Periodic, Boundary::Periodic}};
  std::array<Boundary, 3> bc_high_{
      {Boundary::Periodic, Boundary::Periodic, Boundary::Periodic}};
  bool has_dirichlet_ = false;
  bool all_periodic_ = true;
};

}  // namespace poisson
}  // namespace saturn
