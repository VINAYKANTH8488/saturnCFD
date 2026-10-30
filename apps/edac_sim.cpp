// Entropically Damped Artificial Compressibility (EDAC) prototype.
//
// Exact 2D Taylor-Green vortex on [0,2pi]^2 x [0,2pi], periodic:
//
//   u =  sin(x) cos(y) exp(-2 nu t)
//   v = -cos(x) sin(y) exp(-2 nu t)
//   p = (cos(2x)+cos(2y))/4 exp(-4 nu t)
//
// EDAC evolves a pseudo-pressure instead of solving a Poisson equation:
//
//   du/dt = -(u.grad)u + nu laplacian(u) - grad(p)
//   dp/dt = -c^2 div(u) + nu_p laplacian(p),  c = 1/M_c, nu_p = c^2 tau
//
// with tau = tau_factor*h/c. This app is a verification prototype for the
// Phase 3 EDAC experiment; projection + GMG remains the production path.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"
#include "numerics/incompressible/edac.hpp"
#include "numerics/incompressible/projection.hpp"

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
  std::vector<int> n_list{32, 64};
  int nz = 8;
  double nu = 0.01;
  double mc = 0.05;
  double tau_factor = 0.01;
  double tend = 2.0;
  double cfl = 0.5;
  bool p0_exact = true;
  bool check = false;
  bool verbose = false;
  std::string space = "default";
};

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [options]\n"
      "  --n N              single resolution\n"
      "  --n-list N1 N2 ... resolutions\n"
      "  --nz N             periodic z layers (default 8)\n"
      "  --nu VALUE         kinematic viscosity (default 0.01)\n"
      "  --mc VALUE         pseudo-Mach number M_c (default 0.05)\n"
      "  --tau VALUE        tau_factor, tau = tau_factor*h/c (default 0.01)\n"
      "  --tend VALUE       final time (default 2)\n"
      "  --cfl VALUE        CFL (default 0.5)\n"
      "  --zero-p           start from p=0 instead of the exact pressure\n"
      "  --space NAME       serial|openmp|cuda|hip|sycl|default\n"
      "  --check            fail on large error or divergence\n"
      "  --verbose          print the Kokkos configuration\n"
      "  --help             show this help\n",
      argv0);
}

double parseDouble(const std::string& flag, const std::string& value) {
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

int parseInt(const std::string& flag, const std::string& value) {
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

std::string requireValue(int& i, int argc, char** argv,
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
      p.n_list = {parseInt(a, requireValue(i, argc, argv, a))};
    } else if (a == "--n-list") {
      p.n_list.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        p.n_list.push_back(parseInt(a, argv[++i]));
      }
      if (p.n_list.empty()) {
        std::fprintf(stderr, "--n-list needs at least one value\n");
        std::exit(2);
      }
    } else if (a == "--nz") {
      p.nz = parseInt(a, requireValue(i, argc, argv, a));
    } else if (a == "--nu") {
      p.nu = parseDouble(a, requireValue(i, argc, argv, a));
    } else if (a == "--mc") {
      p.mc = parseDouble(a, requireValue(i, argc, argv, a));
    } else if (a == "--tau") {
      p.tau_factor = parseDouble(a, requireValue(i, argc, argv, a));
    } else if (a == "--tend") {
      p.tend = parseDouble(a, requireValue(i, argc, argv, a));
    } else if (a == "--cfl") {
      p.cfl = parseDouble(a, requireValue(i, argc, argv, a));
    } else if (a == "--zero-p") {
      p.p0_exact = false;
    } else if (a == "--check") {
      p.check = true;
    } else if (a == "--verbose") {
      p.verbose = true;
    } else if (a == "--space") {
      p.space = requireValue(i, argc, argv, a);
    } else if (a.rfind("--kokkos-", 0) == 0) {
      // Handled by Kokkos::initialize().
    } else {
      std::fprintf(stderr, "warning: ignoring unknown argument '%s'\n",
                   a.c_str());
    }
  }
  return p;
}

struct CaseResult {
  int n = 0;
  int steps = 0;
  double dt = 0.0;
  double error = 0.0;
  double divergence = 0.0;
  double seconds = 0.0;
};

template <class ExecSpace>
CaseResult runCase(int n, const Params& params) {
  using view_type = saturn::projection::ScalarView<ExecSpace>;

  const double lx = 2.0 * kPi;
  const double ly = 2.0 * kPi;
  const double lz = 2.0 * kPi;
  const double hx = lx / static_cast<double>(n);
  const double hy = ly / static_cast<double>(n);
  const double hz = lz / static_cast<double>(params.nz);
  const std::array<double, 3> h{{hx, hy, hz}};
  const int nz = params.nz;
  const double c = 1.0 / params.mc;
  saturn::edac::Parameters ep;
  ep.nu = params.nu;
  ep.c = c;
  ep.nu_p = c * params.tau_factor * std::min({hx, hy, hz});

  view_type u("u", nz, n, n);
  view_type v("v", nz, n, n);
  view_type w("w", nz, n, n);
  view_type p("p", nz, n, n);
  view_type u1("u1", nz, n, n);
  view_type v1("v1", nz, n, n);
  view_type w1("w1", nz, n, n);
  view_type p1("p1", nz, n, n);
  view_type u2("u2", nz, n, n);
  view_type v2("v2", nz, n, n);
  view_type w2("w2", nz, n, n);
  view_type p2("p2", nz, n, n);
  view_type ku("ku", nz, n, n);
  view_type kv("kv", nz, n, n);
  view_type kw("kw", nz, n, n);
  view_type kp("kp", nz, n, n);
  view_type gx("gx", nz, n, n);
  view_type gy("gy", nz, n, n);
  view_type gz("gz", nz, n, n);
  view_type div("div", nz, n, n);
  view_type lap("lap", nz, n, n);

  const auto policy = saturn::makeMdrange3<ExecSpace>(saturn::Int3{{0, 0, 0}},
                                                      saturn::Int3{{nz, n, n}});
  Kokkos::parallel_for(
      "edac_initial", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const double xf = i * hx;
        const double yc = (j + 0.5) * hy;
        const double xc = (i + 0.5) * hx;
        const double yf = j * hy;
        const double zc = (k + 0.5) * hz;
        u(k, j, i) = Kokkos::sin(xf) * Kokkos::cos(yc);
        v(k, j, i) = -Kokkos::cos(xc) * Kokkos::sin(yf);
        w(k, j, i) = 0.0;
        p(k, j, i) = 0.25 * (Kokkos::cos(2.0 * xc) + Kokkos::cos(2.0 * yc));
        (void)zc;
      });
  if (!params.p0_exact) {
    Kokkos::deep_copy(p, 0.0);
  }

  const double hmin = std::min({hx, hy, hz});
  Kokkos::Timer timer;
  double time = 0.0;
  int steps = 0;
  double last_dt = 0.0;
  while (time < params.tend - 1.0e-14) {
    double maxSpeed = 0.0;
    Kokkos::parallel_reduce(
        "edac_max_speed", policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
          const double speed =
              Kokkos::sqrt(u(k, j, i) * u(k, j, i) + v(k, j, i) * v(k, j, i) +
                           w(k, j, i) * w(k, j, i));
          acc = speed > acc ? speed : acc;
        },
        Kokkos::Max<double>{maxSpeed});
    double dt = params.cfl * hmin / std::max(maxSpeed + c, 1.0e-12);
    if (params.nu > 0.0) {
      dt = std::min(dt, 0.2 * hmin * hmin / params.nu);
    }
    if (ep.nu_p > 0.0) {
      dt = std::min(dt, 0.2 * hmin * hmin / ep.nu_p);
    }
    if (time + dt > params.tend) {
      dt = params.tend - time;
    }
    last_dt = dt;

    saturn::edac::advanceSsprk3<ExecSpace>(u, v, w, p, u1, v1, w1, p1, u2, v2,
                                           w2, p2, ku, kv, kw, kp, gx, gy, gz,
                                           div, lap, h, ep, dt);
    time += dt;
    ++steps;
  }
  Kokkos::fence();
  const double seconds = saturn::mpiMax(timer.seconds());

  double sum_error = 0.0;
  double sum_div = 0.0;
  double max_div = 0.0;
  const double decay = std::exp(-2.0 * params.nu * time);
  Kokkos::parallel_reduce(
      "edac_error", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& err, double& d2,
                    double& dmax) {
        const int ip = (i + 1 == n) ? 0 : i + 1;
        const int jp = (j + 1 == n) ? 0 : j + 1;
        const int kpp = (k + 1 == nz) ? 0 : k + 1;
        const double uc = 0.5 * (u(k, j, i) + u(k, j, ip));
        const double vc = 0.5 * (v(k, j, i) + v(k, jp, i));
        const double wc = 0.5 * (w(k, j, i) + w(kpp, j, i));
        const double x = (i + 0.5) * hx;
        const double y = (j + 0.5) * hy;
        const double ue = Kokkos::sin(x) * Kokkos::cos(y) * decay;
        const double ve = -Kokkos::cos(x) * Kokkos::sin(y) * decay;
        const double eu = uc - ue;
        const double ev = vc - ve;
        err += eu * eu + ev * ev + wc * wc;
        const double d = (u(k, j, ip) - u(k, j, i)) / hx +
                         (v(k, jp, i) - v(k, j, i)) / hy +
                         (w(kpp, j, i) - w(k, j, i)) / hz;
        d2 += d * d;
        dmax = Kokkos::abs(d) > dmax ? Kokkos::abs(d) : dmax;
      },
      Kokkos::Sum<double>{sum_error}, Kokkos::Sum<double>{sum_div},
      Kokkos::Max<double>{max_div});

  const double cells = static_cast<double>(n) * n * nz;
  CaseResult result;
  result.n = n;
  result.steps = steps;
  result.dt = last_dt;
  result.error = std::sqrt(saturn::mpiSum(sum_error) / (3.0 * cells));
  result.divergence = saturn::mpiSum(std::sqrt(sum_div / cells));
  result.seconds = seconds;
  return result;
}

template <class ExecSpace>
int run(const Params& params) {
  const bool root = saturn::mpiRank() == 0;
  if (root) {
    std::printf("\n%s %s EDAC TGV\n", saturn::kProjectName, saturn::kVersion);
    std::printf("Kokkos: %s (concurrency %d)\n", ExecSpace::name(),
                ExecSpace{}.concurrency());
    if (params.verbose) {
      Kokkos::print_configuration(std::cout, true);
    }
    std::printf("%6s %8s %12s %14s %14s %10s\n", "N", "steps", "dt",
                "L2 velocity", "L2 div", "time [s]");
  }

  int status = 0;
  for (const int n : params.n_list) {
    const CaseResult result = runCase<ExecSpace>(n, params);
    if (root) {
      std::printf("%6d %8d %12.4e %14.4e %14.4e %10.3f\n", result.n,
                  result.steps, result.dt, result.error, result.divergence,
                  result.seconds);
    }
    if (params.check) {
      if (result.error > 1.0e-2 || result.divergence > 1.0e-2) {
        if (root) {
          std::fprintf(stderr,
                       "EDAC check failed at n=%d: error %.3e div %.3e\n", n,
                       result.error, result.divergence);
        }
        status = 1;
      }
    }
  }
  if (root && params.check && status == 0) {
    std::printf("EDAC TGV check passed\n");
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
  const int status = saturn::dispatchExecutionSpace(
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
