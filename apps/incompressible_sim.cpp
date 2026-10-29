// Incompressible Navier-Stokes prototype: MAC projection solver.
//
// Case: two-dimensional Taylor-Green vortex on [0,2pi]^2 x [0,2pi] with
// periodic boundaries. The exact solution
//
//   u =  sin(x) cos(y) exp(-2 nu t)
//   v = -cos(x) sin(y) exp(-2 nu t)
//   w = 0
//
// decays without changing shape and is used for a second-order convergence
// check of the advection, diffusion, and projection steps.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"
#include "numerics/fft_poisson.hpp"
#include "numerics/incompressible.hpp"
#include "numerics/poisson.hpp"
#include "numerics/projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
  std::string case_name = "tgv2d";
  std::vector<int> n_list{32, 64};
  int nz = 8;
  double nu = 0.05;
  double tend = 1.0;
  double cfl = 0.3;
  double min_order = 1.7;
  double poisson_tol = 1.0e-10;
  double filter_alpha = 0.0;
  std::string poisson = "mg";
  std::string space = "default";
  std::string history;
  int diag_every = 0;
  bool check = false;
  bool verbose = false;
};

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [options]\n"
      "  --case NAME        tgv2d|tgv3d|abc (default tgv2d)\n"
      "  --n N              single resolution\n"
      "  --n-list N1 N2 ... resolutions for a convergence sweep\n"
      "  --nz N             periodic z layers (default 8)\n"
      "  --nu VALUE         kinematic viscosity (default 0.05)\n"
      "  --tend VALUE       final time (default 1.0)\n"
      "  --cfl VALUE        CFL (default 0.3)\n"
      "  --min-order VALUE  check threshold (default 1.7)\n"
      "  --filter ALPHA     4th-order periodic filter strength per step\n"
      "  --poisson NAME     mg|fft periodic pressure solver (default mg)\n"
      "  --history FILE     append t,kinetic,max_div diagnostics as CSV\n"
      "  --diag-every N     history interval in steps (default 0 = endpoints)\n"
      "  --space NAME       serial|openmp|cuda|hip|sycl|default\n"
      "  --check            fail if convergence or divergence is bad\n"
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
    } else if (a == "--case") {
      p.case_name = require_value(i, argc, argv, a);
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
    } else if (a == "--nu") {
      p.nu = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--tend") {
      p.tend = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--cfl") {
      p.cfl = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--min-order") {
      p.min_order = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--poisson-tol") {
      p.poisson_tol = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--filter") {
      p.filter_alpha = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--poisson") {
      p.poisson = require_value(i, argc, argv, a);
    } else if (a == "--history") {
      p.history = require_value(i, argc, argv, a);
    } else if (a == "--diag-every") {
      p.diag_every = parse_int(a, require_value(i, argc, argv, a));
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
  if (p.case_name != "tgv2d" && p.case_name != "tgv3d" &&
      p.case_name != "abc") {
    std::fprintf(stderr, "unknown case '%s' (tgv2d|tgv3d|abc)\n",
                 p.case_name.c_str());
    std::exit(2);
  }
  if (p.nz < 1 || p.nz % 2 != 0) {
    std::fprintf(stderr, "--nz must be a positive even number\n");
    std::exit(2);
  }
  return p;
}

struct CaseResult {
  int n = 0;
  int steps = 0;
  double dt = 0.0;
  double l2 = 0.0;
  double kinetic = 0.0;
  double divergence = 0.0;
  double order = 0.0;
  double seconds = 0.0;
};

template <class ExecSpace, class Solver>
CaseResult run_case(int n, const Params& params) {
  using view_type = saturn::projection::ScalarView<ExecSpace>;

  const double lx = 2.0 * kPi;
  const double ly = 2.0 * kPi;
  const double lz = 2.0 * kPi;
  const int nz = (params.case_name == "abc" || params.case_name == "tgv3d")
                     ? n
                     : params.nz;
  const double hx = lx / static_cast<double>(n);
  const double hy = ly / static_cast<double>(n);
  const double hz = lz / static_cast<double>(nz);
  const std::array<double, 3> h{{hx, hy, hz}};
  const int case_id =
      (params.case_name == "abc") ? 1 : ((params.case_name == "tgv3d") ? 2 : 0);

  view_type u("u", nz, n, n);
  view_type v("v", nz, n, n);
  view_type w("w", nz, n, n);
  view_type u1("u1", nz, n, n);
  view_type v1("v1", nz, n, n);
  view_type w1("w1", nz, n, n);
  view_type u2("u2", nz, n, n);
  view_type v2("v2", nz, n, n);
  view_type w2("w2", nz, n, n);
  view_type ku("ku", nz, n, n);
  view_type kv("kv", nz, n, n);
  view_type kw("kw", nz, n, n);
  view_type phi("phi", nz, n, n);
  Kokkos::deep_copy(phi, 0.0);
  view_type scratch("scratch", nz, n, n);

  const auto policy = saturn::make_mdrange3<ExecSpace>(
      saturn::Int3{{0, 0, 0}}, saturn::Int3{{nz, n, n}});
  Kokkos::parallel_for(
      "incompressible_initial", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const double xf = i * hx;
        const double yf = j * hy;
        const double xc = (i + 0.5) * hx;
        const double yc = (j + 0.5) * hy;
        const double zc = (k + 0.5) * hz;
        if (case_id == 1) {
          // ABC (Beltrami) flow: curl(u) = u, div(u) = 0.
          u(k, j, i) = Kokkos::sin(zc) + Kokkos::cos(yc);
          v(k, j, i) = Kokkos::sin(xc) + Kokkos::cos(zc);
          w(k, j, i) = Kokkos::sin(yc) + Kokkos::cos(xc);
        } else if (case_id == 2) {
          u(k, j, i) = Kokkos::sin(xf) * Kokkos::cos(yc) * Kokkos::cos(zc);
          v(k, j, i) = -Kokkos::cos(xc) * Kokkos::sin(yf) * Kokkos::cos(zc);
          w(k, j, i) = 0.0;
        } else {
          u(k, j, i) = Kokkos::sin(xf) * Kokkos::cos(yc);
          v(k, j, i) = -Kokkos::cos(xc) * Kokkos::sin(yf);
          w(k, j, i) = 0.0;
        }
      });

  Solver solver(std::array<int, 3>{{n, n, nz}}, h);
  saturn::poisson::Options poptions;
  poptions.tolerance = params.poisson_tol;
  poptions.max_cycles = 100;

  // Start from a discretely divergence-free field.
  saturn::projection::project<ExecSpace>(u, v, w, phi, scratch, h, 1.0, solver,
                                         poptions);

  auto do_project = [&](view_type& pu, view_type& pv, view_type& pw) {
    saturn::projection::project<ExecSpace>(pu, pv, pw, phi, scratch, h, 1.0,
                                           solver, poptions);
  };

  const bool root = saturn::mpi_rank() == 0;
  std::ofstream history;
  if (root && !params.history.empty()) {
    history.open(params.history);
    if (!history) {
      std::fprintf(stderr, "could not open history file %s\n",
                   params.history.c_str());
      std::exit(2);
    }
    history << "t,kinetic,max_div\n";
  }
  auto record = [&](double current_time) {
    double sum_kinetic = 0.0;
    double max_div = 0.0;
    Kokkos::parallel_reduce(
        "history_diagnostics", policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& kin, double& div) {
          const int ip = (i + 1 == n) ? 0 : i + 1;
          const int jp = (j + 1 == n) ? 0 : j + 1;
          const int kp = (k + 1 == nz) ? 0 : k + 1;
          const double uc = 0.5 * (u(k, j, i) + u(k, j, ip));
          const double vc = 0.5 * (v(k, j, i) + v(k, jp, i));
          const double wc = 0.5 * (w(k, j, i) + w(kp, j, i));
          kin += uc * uc + vc * vc + wc * wc;
          const double d = Kokkos::abs((u(k, j, ip) - u(k, j, i)) / hx +
                                       (v(k, jp, i) - v(k, j, i)) / hy +
                                       (w(kp, j, i) - w(k, j, i)) / hz);
          div = d > div ? d : div;
        },
        Kokkos::Sum<double>{sum_kinetic}, Kokkos::Max<double>{max_div});
    const double kinetic = 0.5 * saturn::mpi_sum(sum_kinetic) * hx * hy * hz;
    const double divergence = saturn::mpi_max(max_div);
    if (root && history.is_open()) {
      history << current_time << "," << kinetic << "," << divergence << "\n";
      history.flush();
    }
  };
  record(0.0);
  int last_record_step = 0;

  const double hmin = std::min({hx, hy, hz});
  Kokkos::Timer timer;
  double time = 0.0;
  int steps = 0;
  double last_dt = 0.0;
  bool valid = true;
  while (time < params.tend - 1.0e-14) {
    double max_speed = 0.0;
    double non_finite = 0.0;
    Kokkos::parallel_reduce(
        "max_speed", policy,
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
    if (params.nu > 0.0) {
      dt = std::min(dt, 0.2 * hmin * hmin / params.nu);
    }
    if (max_speed > 1.0e6 || dt < 1.0e-10 * hmin) {
      valid = false;
      break;
    }
    if (time + dt > params.tend) {
      dt = params.tend - time;
    }
    last_dt = dt;

    saturn::incompressible::advance_ssprk3<ExecSpace>(
        u, v, w, u1, v1, w1, u2, v2, w2, ku, kv, kw, h, params.nu, dt,
        do_project);
    if (params.filter_alpha > 0.0) {
      saturn::incompressible::filter4<ExecSpace>(u, v, w, u1, v1, w1,
                                                 params.filter_alpha);
      do_project(u, v, w);
    }
    time += dt;
    ++steps;
    if (params.diag_every > 0 && steps % params.diag_every == 0) {
      record(time);
      last_record_step = steps;
    }
  }
  Kokkos::fence();
  if (steps != last_record_step) {
    record(time);
  }
  const double seconds = saturn::mpi_max(timer.seconds());

  // Cell-centered velocities and the exact solution at t_end.
  double sum_error = 0.0;
  double sum_kinetic = 0.0;
  double max_div = 0.0;
  const double decay =
      std::exp((case_id == 1 ? -1.0 : -2.0) * params.nu * time);
  Kokkos::parallel_reduce(
      "incompressible_diagnostics", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& err, double& kin,
                    double& div) {
        const int ip = (i + 1 == n) ? 0 : i + 1;
        const int jp = (j + 1 == n) ? 0 : j + 1;
        const int kp = (k + 1 == nz) ? 0 : k + 1;
        const double uc = 0.5 * (u(k, j, i) + u(k, j, ip));
        const double vc = 0.5 * (v(k, j, i) + v(k, jp, i));
        const double wc = 0.5 * (w(k, j, i) + w(kp, j, i));
        kin += uc * uc + vc * vc + wc * wc;
        if (case_id != 2) {
          const double x = (i + 0.5) * hx;
          const double y = (j + 0.5) * hy;
          const double z = (k + 0.5) * hz;
          double ue;
          double ve;
          double we;
          if (case_id == 1) {
            ue = (Kokkos::sin(z) + Kokkos::cos(y)) * decay;
            ve = (Kokkos::sin(x) + Kokkos::cos(z)) * decay;
            we = (Kokkos::sin(y) + Kokkos::cos(x)) * decay;
          } else {
            ue = Kokkos::sin(x) * Kokkos::cos(y) * decay;
            ve = -Kokkos::cos(x) * Kokkos::sin(y) * decay;
            we = 0.0;
          }
          const double eu = uc - ue;
          const double ev = vc - ve;
          const double ew = wc - we;
          err += eu * eu + ev * ev + ew * ew;
        }
        const double d = Kokkos::abs((u(k, j, ip) - u(k, j, i)) / hx +
                                     (v(k, jp, i) - v(k, j, i)) / hy +
                                     (w(kp, j, i) - w(k, j, i)) / hz);
        div = d > div ? d : div;
      },
      Kokkos::Sum<double>{sum_error}, Kokkos::Sum<double>{sum_kinetic},
      Kokkos::Max<double>{max_div});

  const double cells = static_cast<double>(n) * n * nz;
  const double cell_volume = hx * hy * hz;
  CaseResult result;
  result.n = n;
  result.steps = steps;
  result.dt = last_dt;
  if (!valid) {
    result.l2 = std::numeric_limits<double>::infinity();
    result.kinetic = std::numeric_limits<double>::quiet_NaN();
    result.divergence = std::numeric_limits<double>::infinity();
    result.seconds = saturn::mpi_max(timer.seconds());
    return result;
  }
  result.l2 = (case_id == 2)
                  ? 0.0
                  : std::sqrt(saturn::mpi_sum(sum_error) / (3.0 * cells));
  result.kinetic = 0.5 * saturn::mpi_sum(sum_kinetic) * cell_volume;
  result.divergence = saturn::mpi_max(max_div);
  result.seconds = seconds;
  return result;
}

template <class ExecSpace>
int run(const Params& params) {
  const bool root = saturn::mpi_rank() == 0;
#ifndef SATURN_HAVE_FFTW
  if (params.poisson == "fft") {
    if (root) {
      std::fprintf(stderr, "--poisson fft requires an FFTW3 build\n");
    }
    return 2;
  }
#endif
  if (root) {
    std::printf("\n%s %s incompressible %s\n", saturn::kProjectName,
                saturn::kVersion, params.case_name.c_str());
    std::printf("Kokkos: %s (concurrency %d)\n", ExecSpace::name(),
                ExecSpace{}.concurrency());
    if (params.verbose) {
      Kokkos::print_configuration(std::cout, true);
    }
    std::printf("%6s %8s %12s %14s %14s %14s %8s %10s\n", "N", "steps", "dt",
                "L2 velocity", "kinetic", "max |div|", "order", "time [s]");
  }

  std::vector<CaseResult> results;
  double previous_error = 0.0;
  double previous_h = 0.0;
  for (const int n : params.n_list) {
    CaseResult result;
#ifdef SATURN_HAVE_FFTW
    if (params.poisson == "fft") {
      result = run_case<ExecSpace, saturn::poisson::FftPoisson<ExecSpace>>(
          n, params);
    } else {
      result =
          run_case<ExecSpace, saturn::poisson::GeometricMultigrid<ExecSpace>>(
              n, params);
    }
#else
    result =
        run_case<ExecSpace, saturn::poisson::GeometricMultigrid<ExecSpace>>(
            n, params);
#endif
    if (params.case_name != "tgv3d" && previous_error > 0.0) {
      const double h = 2.0 * kPi / static_cast<double>(n);
      result.order =
          std::log(previous_error / result.l2) / std::log(previous_h / h);
    }
    previous_error = result.l2;
    previous_h = 2.0 * kPi / static_cast<double>(n);
    results.push_back(result);
    if (root) {
      std::printf("%6d %8d %12.4e %14.4e %14.4e %14.4e %8.2f %10.3f\n",
                  result.n, result.steps, result.dt, result.l2, result.kinetic,
                  result.divergence, result.order, result.seconds);
    }
  }

  int status = 0;
  if (params.check) {
    bool order_ok = true;
    if (params.case_name != "tgv3d" && results.size() >= 2) {
      for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i].order < params.min_order) {
          order_ok = false;
        }
      }
    }
    bool divergence_ok = true;
    for (const auto& result : results) {
      divergence_ok = divergence_ok && result.divergence < 1.0e-7;
    }
    if (!order_ok) {
      if (root) {
        std::fprintf(stderr,
                     "incompressible %s check failed: order below %.2f\n",
                     params.case_name.c_str(), params.min_order);
      }
      status = 1;
    }
    if (!divergence_ok) {
      if (root) {
        std::fprintf(stderr,
                     "incompressible %s check failed: |div| too large\n",
                     params.case_name.c_str());
      }
      status = 1;
    }
    if (status == 0 && root) {
      std::printf("incompressible %s check passed\n", params.case_name.c_str());
    }
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
