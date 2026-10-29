// Lid-driven cavity on a staggered MAC grid.
//
// Square cavity [0,1]^2, periodic in z, no-slip x/y walls, moving top lid
// u_lid=1 (Re = u_lid L / nu). The steady centreline profiles are compared
// with the Ghia, Ghia & Shin (1982) Re=100 data.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"
#include "numerics/cavity.hpp"
#include "numerics/poisson.hpp"
#include "numerics/projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SATURN_DEFAULT_SPACE
#define SATURN_DEFAULT_SPACE "default"
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Params {
  std::vector<int> n_list{32, 48};
  int nz = 4;
  double re = 100.0;
  double u_lid = 1.0;
  double tend = 15.0;
  double cfl = 0.3;
  double poisson_tol = 1.0e-10;
  double rms_tol = 0.03;
  std::string space = "default";
  bool check = false;
  bool verbose = false;
};

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [options]\n"
      "  --n N              single cavity resolution\n"
      "  --n-list N1 N2 ... resolutions for a convergence sweep\n"
      "  --nz N             periodic z layers (default 4)\n"
      "  --re VALUE         Reynolds number (default 100)\n"
      "  --u-lid VALUE      lid speed (default 1)\n"
      "  --tend VALUE       final time (default 15)\n"
      "  --cfl VALUE        CFL (default 0.3)\n"
      "  --rms-tol VALUE    Ghia centreline RMS tolerance (default 0.03)\n"
      "  --space NAME       serial|openmp|cuda|hip|sycl|default\n"
      "  --check            fail on divergence or Ghia RMS error\n"
      "  --verbose          print the Kokkos configuration\n"
      "  --help             show this help\n",
      argv0);
}

double parse_double(const std::string& flag, const std::string& value) {
  try {
    std::size_t pos = 0;
    const double result = std::stod(value, &pos);
    if (pos != value.size()) {
      throw std::invalid_argument{"trailing characters"};
    }
    return result;
  } catch (const std::exception&) {
    std::fprintf(stderr, "invalid number for %s: '%s'\n", flag.c_str(),
                 value.c_str());
    std::exit(2);
  }
}

int parse_int(const std::string& flag, const std::string& value) {
  try {
    std::size_t pos = 0;
    const int result = std::stoi(value, &pos);
    if (pos != value.size()) {
      throw std::invalid_argument{"trailing characters"};
    }
    return result;
  } catch (const std::exception&) {
    std::fprintf(stderr, "invalid integer for %s: '%s'\n", flag.c_str(),
                 value.c_str());
    std::exit(2);
  }
}

std::string require_value(int& i, int argc, char** argv,
                          const std::string& flag) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "missing value for %s\n", flag.c_str());
    std::exit(2);
  }
  return argv[++i];
}

Params parse(int argc, char** argv) {
  Params p;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--n") {
      p.n_list = {parse_int(a, require_value(i, argc, argv, a))};
    } else if (a == "--n-list") {
      p.n_list.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        p.n_list.push_back(parse_int(a, argv[++i]));
      }
      if (p.n_list.empty()) {
        std::fprintf(stderr, "--n-list needs at least one value\n");
        std::exit(2);
      }
    } else if (a == "--nz") {
      p.nz = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--re") {
      p.re = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--u-lid") {
      p.u_lid = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--tend") {
      p.tend = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--cfl") {
      p.cfl = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--poisson-tol") {
      p.poisson_tol = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--rms-tol") {
      p.rms_tol = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--space") {
      p.space = require_value(i, argc, argv, a);
    } else if (a == "--check") {
      p.check = true;
    } else if (a == "--verbose") {
      p.verbose = true;
    } else if (a.rfind("--kokkos-", 0) == 0) {
      // Handled by Kokkos::initialize().
    } else {
      std::fprintf(stderr, "warning: ignoring unknown argument '%s'\n",
                   a.c_str());
    }
  }
  if (p.nz < 1) {
    std::fprintf(stderr, "--nz must be positive\n");
    std::exit(2);
  }
  return p;
}

// Ghia et al. (1982), Re=100, along the vertical centreline x=0.5.
constexpr int kGhiaCount = 17;
constexpr double kGhiaY[kGhiaCount] = {
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
    0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000};
constexpr double kGhiaU[kGhiaCount] = {
    1.00000,  0.84123,  0.78871,  0.73722,  0.68717,  0.23151,
    0.00332,  -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,
    -0.06434, -0.04775, -0.04192, -0.03717, 0.00000};
constexpr double kGhiaX[kGhiaCount] = {
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
    0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000};
constexpr double kGhiaV[kGhiaCount] = {
    0.00000,  -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
    -0.22445, -0.24533, 0.05454,  0.17527,  0.17507,  0.16077,
    0.12317,  0.10890,  0.10090,  0.09233,  0.00000};

// Ghia et al. (1982), Re=400, same station coordinates.
constexpr double kGhia400U[kGhiaCount] = {
    1.00000,  0.75837,  0.68439,  0.61756,  0.55892,  0.29093,
    0.16256,  0.02135,  -0.11477, -0.17119, -0.32726, -0.24299,
    -0.14612, -0.10338, -0.09266, -0.08186, 0.00000};
constexpr double kGhia400V[kGhiaCount] = {
    0.00000,  -0.12146, -0.15663, -0.19254, -0.22847, -0.23827,
    -0.44993, -0.38598, 0.05186,  0.30174,  0.30203,  0.28124,
    0.22965,  0.20920,  0.19713,  0.18360,  0.00000};

struct CaseResult {
  int n = 0;
  int steps = 0;
  double dt = 0.0;
  double rms_u = 0.0;
  double rms_v = 0.0;
  double divergence = 0.0;
  double seconds = 0.0;
};

// Linear interpolation of a cell-centered profile given at points
// (j+0.5)*h, with wall values supplied at the two ends.
double interpolate(const std::vector<double>& values, double h, double wall_lo,
                   double wall_hi, double x) {
  const int n = static_cast<int>(values.size());
  const double x0 = 0.5 * h;
  const double x1 = (n - 0.5) * h;
  if (x <= x0) {
    const double t = (x - 0.0) / (x0 - 0.0);
    return (1.0 - t) * wall_lo + t * values[0];
  }
  if (x >= x1) {
    const double t = (x - x1) / (1.0 - x1);
    return (1.0 - t) * values[n - 1] + t * wall_hi;
  }
  const double pos = x / h - 0.5;
  const int j = static_cast<int>(std::floor(pos));
  const double t = pos - static_cast<double>(j);
  return (1.0 - t) * values[j] + t * values[j + 1];
}

template <class ExecSpace>
CaseResult run_case(int n, const Params& params) {
  using view_type = saturn::projection::ScalarView<ExecSpace>;

  saturn::cavity::Geometry g;
  g.n0 = n;
  g.n1 = n;
  g.n2 = params.nz;
  g.hx = 1.0 / static_cast<double>(n);
  g.hy = 1.0 / static_cast<double>(n);
  g.hz = 1.0 / static_cast<double>(params.nz);
  g.u_lid = params.u_lid;
  const double nu = params.u_lid / params.re;

  view_type u("u", g.n2, g.n1, g.n0);
  view_type v("v", g.n2, g.n1, g.n0);
  view_type w("w", g.n2, g.n1, g.n0);
  view_type u1("u1", g.n2, g.n1, g.n0);
  view_type v1("v1", g.n2, g.n1, g.n0);
  view_type w1("w1", g.n2, g.n1, g.n0);
  view_type u2("u2", g.n2, g.n1, g.n0);
  view_type v2("v2", g.n2, g.n1, g.n0);
  view_type w2("w2", g.n2, g.n1, g.n0);
  view_type ku("ku", g.n2, g.n1, g.n0);
  view_type kv("kv", g.n2, g.n1, g.n0);
  view_type kw("kw", g.n2, g.n1, g.n0);
  view_type phi("phi", g.n2, g.n1, g.n0);
  Kokkos::deep_copy(phi, 0.0);
  view_type scratch("scratch", g.n2, g.n1, g.n0);

  Kokkos::deep_copy(u, 0.0);
  Kokkos::deep_copy(v, 0.0);
  Kokkos::deep_copy(w, 0.0);

  std::array<saturn::poisson::Boundary, 3> bc_low{
      {saturn::poisson::Boundary::Neumann, saturn::poisson::Boundary::Neumann,
       saturn::poisson::Boundary::Periodic}};
  std::array<saturn::poisson::Boundary, 3> bc_high = bc_low;
  saturn::poisson::GeometricMultigrid<ExecSpace> solver(
      std::array<int, 3>{{g.n0, g.n1, g.n2}},
      std::array<double, 3>{{g.hx, g.hy, g.hz}}, bc_low, bc_high);
  saturn::poisson::Options poptions;
  poptions.tolerance = params.poisson_tol;
  poptions.max_cycles = 100;

  auto do_project = [&](view_type& pu, view_type& pv, view_type& pw) {
    saturn::cavity::project<ExecSpace>(pu, pv, pw, phi, scratch, g, solver,
                                       poptions);
  };

  do_project(u, v, w);

  const auto policy = saturn::make_mdrange3<ExecSpace>(
      saturn::Int3{{0, 0, 0}}, saturn::Int3{{g.n2, g.n1, g.n0}});
  const double hmin = std::min({g.hx, g.hy, g.hz});
  Kokkos::Timer timer;
  double time = 0.0;
  int steps = 0;
  double last_dt = 0.0;
  bool valid = true;
  while (time < params.tend - 1.0e-14) {
    double max_speed = 0.0;
    double non_finite = 0.0;
    Kokkos::parallel_reduce(
        "cavity_max_speed", policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& acc, double& bad) {
          const double speed =
              Kokkos::sqrt(u(k, j, i) * u(k, j, i) + v(k, j, i) * v(k, j, i) +
                           w(k, j, i) * w(k, j, i));
          if (!(speed == speed) || Kokkos::abs(speed) > 1.0e100) {
            bad += 1.0;
            return;
          }
          acc = speed > acc ? speed : acc;
        },
        Kokkos::Max<double>{max_speed}, Kokkos::Sum<double>{non_finite});
    non_finite = saturn::mpi_sum(non_finite);
    if (non_finite > 0.0) {
      valid = false;
      break;
    }
    double dt = params.cfl * hmin / std::max(max_speed, 1.0e-12);
    dt = std::min(dt, 0.2 * hmin * hmin / nu);
    if (time + dt > params.tend) {
      dt = params.tend - time;
    }
    last_dt = dt;
    saturn::cavity::advance_ssprk3<ExecSpace>(
        u, v, w, u1, v1, w1, u2, v2, w2, ku, kv, kw, g, nu, dt, do_project);
    time += dt;
    ++steps;
  }
  Kokkos::fence();
  const double seconds = saturn::mpi_max(timer.seconds());

  CaseResult result;
  result.n = n;
  result.steps = steps;
  result.dt = last_dt;
  result.seconds = seconds;
  if (!valid) {
    result.rms_u = std::numeric_limits<double>::infinity();
    result.rms_v = std::numeric_limits<double>::infinity();
    result.divergence = std::numeric_limits<double>::infinity();
    return result;
  }

  double max_div = 0.0;
  Kokkos::parallel_reduce(
      "cavity_divergence", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& div) {
        const int ip = (i + 1 == g.n0) ? -1 : i + 1;
        const int jp = (j + 1 == g.n1) ? -1 : j + 1;
        const int kp = (k + 1 == g.n2) ? 0 : k + 1;
        const double up = (ip < 0) ? 0.0 : u(k, j, ip);
        const double vp = (jp < 0) ? 0.0 : v(k, jp, i);
        const double d =
            Kokkos::abs((up - u(k, j, i)) / g.hx + (vp - v(k, j, i)) / g.hy +
                        (w(kp, j, i) - w(k, j, i)) / g.hz);
        div = d > div ? d : div;
      },
      Kokkos::Max<double>{max_div});
  result.divergence = saturn::mpi_max(max_div);

  auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  const int i_center = n / 2;
  const int j_center = n / 2;
  const int k_center = params.nz / 2;
  std::vector<double> u_center(n);
  std::vector<double> v_center(n);
  for (int j = 0; j < n; ++j) {
    const int ip = (i_center + 1 == n) ? i_center : i_center + 1;
    u_center[j] = 0.5 * (hu(k_center, j, i_center) + hu(k_center, j, ip));
  }
  for (int i = 0; i < n; ++i) {
    const int jp = (j_center + 1 == n) ? j_center : j_center + 1;
    v_center[i] = 0.5 * (hv(k_center, j_center, i) + hv(k_center, jp, i));
  }

  double sum_u = 0.0;
  double sum_v = 0.0;
  const bool has_ghia = params.re == 100.0 || params.re == 400.0;
  const double* ref_u = (params.re == 400.0) ? kGhia400U : kGhiaU;
  const double* ref_v = (params.re == 400.0) ? kGhia400V : kGhiaV;
  if (has_ghia) {
    for (int q = 0; q < kGhiaCount; ++q) {
      const double uq =
          interpolate(u_center, g.hy, 0.0, params.u_lid, kGhiaY[q]);
      const double vq = interpolate(v_center, g.hx, 0.0, 0.0, kGhiaX[q]);
      const double du = uq - ref_u[q];
      const double dv = vq - ref_v[q];
      sum_u += du * du;
      sum_v += dv * dv;
    }
    result.rms_u = std::sqrt(sum_u / kGhiaCount);
    result.rms_v = std::sqrt(sum_v / kGhiaCount);
  }
  return result;
}

template <class ExecSpace>
int run(const Params& params) {
  const bool root = saturn::mpi_rank() == 0;
  if (root) {
    std::printf("\n%s %s lid-driven cavity Re=%.0f\n", saturn::kProjectName,
                saturn::kVersion, params.re);
    std::printf("Kokkos: %s (concurrency %d)\n", ExecSpace::name(),
                ExecSpace{}.concurrency());
    if (params.verbose) {
      Kokkos::print_configuration(std::cout, true);
    }
    std::printf("%6s %8s %12s %14s %14s %14s %10s\n", "N", "steps", "dt",
                "RMS u", "RMS v", "max |div|", "time [s]");
  }

  int status = 0;
  for (const int n : params.n_list) {
    const CaseResult result = run_case<ExecSpace>(n, params);
    if (root) {
      std::printf("%6d %8d %12.4e %14.4e %14.4e %14.4e %10.3f\n", result.n,
                  result.steps, result.dt, result.rms_u, result.rms_v,
                  result.divergence, result.seconds);
    }
    if (params.check) {
      if (result.divergence > 1.0e-7) {
        if (root) {
          std::fprintf(stderr, "cavity check failed: |div| = %.3e at n=%d\n",
                       result.divergence, n);
        }
        status = 1;
      }
      const bool has_ghia = params.re == 100.0 || params.re == 400.0;
      if (has_ghia &&
          (result.rms_u > params.rms_tol || result.rms_v > params.rms_tol)) {
        if (root) {
          std::fprintf(stderr,
                       "cavity check failed: Ghia RMS (u %.4f, v %.4f) > "
                       "%.4f at n=%d\n",
                       result.rms_u, result.rms_v, params.rms_tol, n);
        }
        status = 1;
      }
    }
  }
  if (root && params.check && status == 0) {
    std::printf("cavity check passed\n");
  }
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  const Params params = parse(argc, argv);

#ifdef SATURN_ENABLE_MPI
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif

  Kokkos::initialize(argc, argv);
  const int status = saturn::dispatch_execution_space(
      params.space, SATURN_DEFAULT_SPACE, [&](auto tag) {
        using space_type = decltype(tag);
        return run<space_type>(params);
      });
  Kokkos::finalize();

#ifdef SATURN_ENABLE_MPI
  MPI_Finalize();
#endif
  return status;
}
