// Wall-bounded channel flows on a staggered MAC grid.
//
//   couette     u(y) = u_top y/Ly, no body force (exact steady solution)
//   poiseuille  u(y) = gx y (Ly - y) / (2 nu), driven by a body force
//
// Periodic in x and z, no-slip walls at y=0 and y=Ly, homogeneous Neumann
// pressure at both walls. The Couette case is an exact discrete steady state
// of the linear momentum operator; Poiseuille checks the body-force/viscous
// balance and the wall closure.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"
#include "numerics/channel.hpp"
#include "numerics/channel_poisson.hpp"
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
#include <type_traits>
#include <vector>

#ifndef SATURN_DEFAULT_SPACE
#define SATURN_DEFAULT_SPACE "default"
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Params {
  std::string case_name = "couette";
  std::vector<int> ny_list{16, 32, 64};
  int nx = 32;
  int nz = 8;
  double lx = 2.0 * kPi;
  double lz = 2.0 * kPi;
  double ly = 1.0;
  double nu = 0.1;
  double u_top = 1.0;
  double gx = 0.0;
  double perturb = 0.0;
  double filter_alpha = 0.0;
  double tend = 0.1;
  double cfl = 0.3;
  double poisson_tol = 1.0e-10;
  std::string stats;
  std::string poisson = "mg";
  std::string space = "default";
  bool check = false;
  bool verbose = false;
};

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [options]\n"
      "  --case NAME        couette|poiseuille (default couette)\n"
      "  --n-list N1 ...    wall-normal resolutions (default 16 32 64)\n"
      "  --nx N             streamwise cells (default 32)\n"
      "  --nz N             spanwise cells (default 8)\n"
      "  --lx VALUE         streamwise length (default 2 pi)\n"
      "  --lz VALUE         spanwise length (default 2 pi)\n"
      "  --ly VALUE         channel height (default 1)\n"
      "  --nu VALUE         kinematic viscosity (default 0.1)\n"
      "  --u-top VALUE      top-wall speed for Couette (default 1)\n"
      "  --gx VALUE         streamwise body force for Poiseuille (default 1)\n"
      "  --perturb VALUE    initial perturbation amplitude (default 0)\n"
      "  --filter ALPHA     wall-aware 4th-order filter strength (default 0)\n"
      "  --stats FILE       write plane-averaged channel statistics as CSV\n"
      "  --poisson NAME     mg|fft pressure solver (default mg)\n"
      "  --tend VALUE       final time (default 0.1)\n"
      "  --cfl VALUE        CFL (default 0.3)\n"
      "  --space NAME       serial|openmp|cuda|hip|sycl|default\n"
      "  --check            fail on bad divergence or solution error\n"
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
  bool gx_set = false;
  bool u_top_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--case") {
      p.case_name = require_value(i, argc, argv, a);
    } else if (a == "--n-list") {
      p.ny_list.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        p.ny_list.push_back(parse_int(a, argv[++i]));
      }
      if (p.ny_list.empty()) {
        std::fprintf(stderr, "--n-list needs at least one value\n");
        std::exit(2);
      }
    } else if (a == "--nx") {
      p.nx = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--nz") {
      p.nz = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--lx") {
      p.lx = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--lz") {
      p.lz = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--ly") {
      p.ly = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--perturb") {
      p.perturb = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--stats") {
      p.stats = require_value(i, argc, argv, a);
    } else if (a == "--filter") {
      p.filter_alpha = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--poisson") {
      p.poisson = require_value(i, argc, argv, a);
    } else if (a == "--nu") {
      p.nu = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--u-top") {
      p.u_top = parse_double(a, require_value(i, argc, argv, a));
      u_top_set = true;
    } else if (a == "--gx") {
      p.gx = parse_double(a, require_value(i, argc, argv, a));
      gx_set = true;
    } else if (a == "--tend") {
      p.tend = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--cfl") {
      p.cfl = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--poisson-tol") {
      p.poisson_tol = parse_double(a, require_value(i, argc, argv, a));
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
  if (p.case_name != "couette" && p.case_name != "poiseuille") {
    std::fprintf(stderr, "unknown case '%s' (couette|poiseuille)\n",
                 p.case_name.c_str());
    std::exit(2);
  }
  if (p.case_name == "couette") {
    if (!u_top_set) {
      p.u_top = 1.0;
    }
    if (!gx_set) {
      p.gx = 0.0;
    }
  } else {
    if (!u_top_set) {
      p.u_top = 0.0;
    }
    if (!gx_set) {
      p.gx = 1.0;
    }
  }
  return p;
}

struct CaseResult {
  int ny = 0;
  int steps = 0;
  double dt = 0.0;
  double l2 = 0.0;
  double divergence = 0.0;
  double centerline = 0.0;
  double seconds = 0.0;
};

template <class Solver, class ExecSpace>
Solver make_channel_solver(const saturn::channel::Geometry& g) {
#ifdef SATURN_HAVE_FFTW
  if constexpr (std::is_same_v<Solver,
                               saturn::poisson::ChannelFftPoisson<ExecSpace>>) {
    return Solver(std::array<int, 3>{{g.n0, g.n1, g.n2}},
                  std::array<double, 3>{{g.hx, g.hy, g.hz}});
  } else
#endif
  {
    std::array<saturn::poisson::Boundary, 3> bc{
        {saturn::poisson::Boundary::Periodic,
         saturn::poisson::Boundary::Neumann,
         saturn::poisson::Boundary::Periodic}};
    return Solver(std::array<int, 3>{{g.n0, g.n1, g.n2}},
                  std::array<double, 3>{{g.hx, g.hy, g.hz}}, bc, bc);
  }
}

template <class ExecSpace, class Solver>
CaseResult run_case(int ny, const Params& params) {
  using view_type = saturn::projection::ScalarView<ExecSpace>;

  saturn::channel::Geometry g;
  g.n0 = params.nx;
  g.n1 = ny;
  g.n2 = params.nz;
  const double lx = params.lx;
  const double lz = params.lz;
  g.hx = lx / static_cast<double>(g.n0);
  g.hy = params.ly / static_cast<double>(ny);
  g.hz = lz / static_cast<double>(g.n2);
  g.ly = params.ly;
  g.u_top = (params.case_name == "couette") ? params.u_top : 0.0;
  g.gx = (params.case_name == "poiseuille") ? params.gx : 0.0;
  g.nu = params.nu;

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

  const auto policy = saturn::make_mdrange3<ExecSpace>(
      saturn::Int3{{0, 0, 0}}, saturn::Int3{{g.n2, g.n1, g.n0}});
  const double perturb = params.perturb;
  Kokkos::parallel_for(
      "channel_initial", policy, KOKKOS_LAMBDA(int k, int j, int i) {
        const double y = (j + 0.5) * g.hy;
        const double xc = (i + 0.5) * g.hx;
        const double yc = (j + 0.5) * g.hy;
        const double zc = (k + 0.5) * g.hz;
        const double xf = i * g.hx;
        const double yf = j * g.hy;
        const double zf = k * g.hz;
        const double profile = (g.u_top > 0.0)
                                   ? g.u_top * y / g.ly
                                   : g.gx * y * (g.ly - y) / (2.0 * g.nu);
        u(k, j, i) = profile + perturb * Kokkos::sin(4.0 * kPi * xf / lx) *
                                   Kokkos::cos(kPi * yc / g.ly) *
                                   Kokkos::sin(2.0 * kPi * zc / lz);
        v(k, j, i) = perturb * Kokkos::cos(2.0 * kPi * xc / lx) *
                     Kokkos::sin(kPi * yf / g.ly) *
                     Kokkos::sin(2.0 * kPi * zc / lz);
        w(k, j, i) = perturb * Kokkos::sin(2.0 * kPi * xc / lx) *
                     Kokkos::cos(kPi * yc / g.ly) *
                     Kokkos::cos(4.0 * kPi * zf / lz);
      });

  Solver solver = make_channel_solver<Solver, ExecSpace>(g);
  saturn::poisson::Options poptions;
  poptions.tolerance = params.poisson_tol;
  poptions.max_cycles = 100;

  auto do_project = [&](view_type& pu, view_type& pv, view_type& pw) {
    saturn::channel::project<ExecSpace>(pu, pv, pw, phi, scratch, g, solver,
                                        poptions);
  };

  do_project(u, v, w);

  const double hmin = std::min({g.hx, g.hy, g.hz});
  bool valid = true;
  Kokkos::Timer timer;
  double time = 0.0;
  int steps = 0;
  double last_dt = 0.0;
  while (time < params.tend - 1.0e-14) {
    double max_speed = 0.0;
    Kokkos::parallel_reduce(
        "channel_max_speed", policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
          const double speed =
              Kokkos::sqrt(u(k, j, i) * u(k, j, i) + v(k, j, i) * v(k, j, i) +
                           w(k, j, i) * w(k, j, i));
          acc = speed > acc ? speed : acc;
        },
        Kokkos::Max<double>{max_speed});
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

    saturn::channel::advance_ssprk3<ExecSpace>(u, v, w, u1, v1, w1, u2, v2, w2,
                                               ku, kv, kw, g, params.nu, dt,
                                               do_project);
    time += dt;
    ++steps;
  }
  Kokkos::fence();
  const double seconds = saturn::mpi_max(timer.seconds());

  double sum_error = 0.0;
  double max_div = 0.0;
  double centerline = 0.0;
  Kokkos::parallel_reduce(
      "channel_error", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& err, double& div,
                    double& center) {
        const int ip = (i + 1 == g.n0) ? 0 : i + 1;
        const int jp = (j + 1 == g.n1) ? -1 : j + 1;
        const int kp = (k + 1 == g.n2) ? 0 : k + 1;
        const double y = (j + 0.5) * g.hy;
        const double ue = (g.u_top > 0.0)
                              ? g.u_top * y / g.ly
                              : g.gx * y * (g.ly - y) / (2.0 * g.nu);
        const double uc = 0.5 * (u(k, j, i) + u(k, j, ip));
        const double vc = 0.5 * (v(k, j, i) + ((jp < 0) ? 0.0 : v(k, jp, i)));
        const double wc = 0.5 * (w(k, j, i) + w(kp, j, i));
        const double eu = uc - ue;
        err += eu * eu + vc * vc + wc * wc;
        const double vp = (jp < 0) ? 0.0 : v(k, jp, i);
        const double d = Kokkos::abs((u(k, j, ip) - u(k, j, i)) / g.hx +
                                     (vp - v(k, j, i)) / g.hy +
                                     (w(kp, j, i) - w(k, j, i)) / g.hz);
        div = d > div ? d : div;
        if (j == g.n1 / 2) {
          center = Kokkos::abs(uc) > center ? Kokkos::abs(uc) : center;
        }
      },
      Kokkos::Sum<double>{sum_error}, Kokkos::Max<double>{max_div},
      Kokkos::Max<double>{centerline});

  const double cells = static_cast<double>(g.n0) * g.n1 * g.n2;
  CaseResult result;
  result.ny = ny;
  result.steps = steps;
  result.dt = last_dt;
  if (!valid) {
    result.l2 = std::numeric_limits<double>::infinity();
    result.divergence = std::numeric_limits<double>::infinity();
    result.seconds = saturn::mpi_max(timer.seconds());
    return result;
  }
  result.l2 = std::sqrt(saturn::mpi_sum(sum_error) / (3.0 * cells));
  result.divergence = saturn::mpi_max(max_div);
  result.centerline = saturn::mpi_max(centerline);
  result.seconds = seconds;

  if (!params.stats.empty()) {
    auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
    auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
    const double count = static_cast<double>(g.n0) * g.n2;
    std::vector<double> mean_u(g.n1), mean_v(g.n1), mean_w(g.n1);
    std::vector<double> uu(g.n1), vv(g.n1), ww(g.n1), uv(g.n1);
    for (int j = 0; j < g.n1; ++j) {
      double su = 0.0;
      double sv = 0.0;
      double sw = 0.0;
      double suu = 0.0;
      double svv = 0.0;
      double sww = 0.0;
      double suv = 0.0;
      for (int k = 0; k < g.n2; ++k) {
        for (int i = 0; i < g.n0; ++i) {
          const int ip = (i + 1 == g.n0) ? 0 : i + 1;
          const int jp = (j + 1 == g.n1) ? -1 : j + 1;
          const int kp = (k + 1 == g.n2) ? 0 : k + 1;
          const double uc = 0.5 * (hu(k, j, i) + hu(k, j, ip));
          const double vc =
              0.5 * (hv(k, j, i) + ((jp < 0) ? 0.0 : hv(k, jp, i)));
          const double wc = 0.5 * (hw(k, j, i) + hw(kp, j, i));
          su += uc;
          sv += vc;
          sw += wc;
          suu += uc * uc;
          svv += vc * vc;
          sww += wc * wc;
          suv += uc * vc;
        }
      }
      mean_u[j] = su / count;
      mean_v[j] = sv / count;
      mean_w[j] = sw / count;
      uu[j] = suu / count - mean_u[j] * mean_u[j];
      vv[j] = svv / count - mean_v[j] * mean_v[j];
      ww[j] = sww / count - mean_w[j] * mean_w[j];
      uv[j] = suv / count - mean_u[j] * mean_v[j];
    }

    auto wall_u = [&](int j) {
      double sum = 0.0;
      for (int k = 0; k < g.n2; ++k) {
        for (int i = 0; i < g.n0; ++i) {
          const int ip = (i + 1 == g.n0) ? 0 : i + 1;
          sum += 0.5 * (hu(k, j, i) + hu(k, j, ip));
        }
      }
      return sum / count;
    };
    const double u_bottom = wall_u(0);
    const double u_last = wall_u(g.n1 - 1);
    const double u_wall_top =
        (params.case_name == "couette") ? params.u_top : 0.0;
    const double shear_bottom = 2.0 * u_bottom / g.hy;
    const double shear_top = 2.0 * (u_wall_top - u_last) / g.hy;
    const double tau_w =
        params.nu * 0.5 * (std::abs(shear_bottom) + std::abs(shear_top));
    const double u_tau = std::sqrt(std::max(tau_w, 0.0));
    const double delta = 0.5 * g.ly;
    const double re_tau = (u_tau > 0.0) ? u_tau * delta / params.nu : 0.0;

    if (saturn::mpi_rank() == 0) {
      std::printf("  u_tau=%.6e  Re_tau=%.1f  (nu=%.4e, delta=%.4f)\n", u_tau,
                  re_tau, params.nu, delta);
      std::ofstream out(params.stats);
      if (out) {
        out << "y,yplus,uplus,urms,vrms,wrms,uv\n";
        for (int j = 0; j < g.n1; ++j) {
          const double y = (j + 0.5) * g.hy;
          const double yplus = (u_tau > 0.0) ? y * u_tau / params.nu : 0.0;
          const double uplus = (u_tau > 0.0) ? mean_u[j] / u_tau : 0.0;
          out << y << "," << yplus << "," << uplus << ","
              << std::sqrt(std::max(uu[j], 0.0)) << ","
              << std::sqrt(std::max(vv[j], 0.0)) << ","
              << std::sqrt(std::max(ww[j], 0.0)) << "," << uv[j] << "\n";
        }
        std::printf("  statistics: %s\n", params.stats.c_str());
      }
    }
  }
  return result;
}

template <class ExecSpace>
int run(const Params& params) {
  const bool root = saturn::mpi_rank() == 0;
  if (root) {
    std::printf("\n%s %s channel %s\n", saturn::kProjectName, saturn::kVersion,
                params.case_name.c_str());
    std::printf("Kokkos: %s (concurrency %d)\n", ExecSpace::name(),
                ExecSpace{}.concurrency());
    if (params.verbose) {
      Kokkos::print_configuration(std::cout, true);
    }
    std::printf("%6s %8s %12s %14s %14s %14s %10s\n", "NY", "steps", "dt",
                "L2 error", "max |div|", "center |u|", "time [s]");
  }

  int status = 0;
  for (const int ny : params.ny_list) {
    CaseResult result;
#ifdef SATURN_HAVE_FFTW
    if (params.poisson == "fft") {
      result =
          run_case<ExecSpace, saturn::poisson::ChannelFftPoisson<ExecSpace>>(
              ny, params);
    } else {
      result =
          run_case<ExecSpace, saturn::poisson::GeometricMultigrid<ExecSpace>>(
              ny, params);
    }
#else
    result =
        run_case<ExecSpace, saturn::poisson::GeometricMultigrid<ExecSpace>>(
            ny, params);
#endif
    if (root) {
      std::printf("%6d %8d %12.4e %14.4e %14.4e %14.4e %10.3f\n", result.ny,
                  result.steps, result.dt, result.l2, result.divergence,
                  result.centerline, result.seconds);
    }
    if (params.check) {
      if (result.divergence > 1.0e-7) {
        if (root) {
          std::fprintf(stderr, "channel check failed: |div| = %.3e at ny=%d\n",
                       result.divergence, ny);
        }
        status = 1;
      }
      const double tolerance =
          (params.case_name == "couette") ? 1.0e-8 : 5.0e-2;
      if (result.l2 > tolerance) {
        if (root) {
          std::fprintf(stderr,
                       "channel check failed: L2 error %.3e > %.3e at ny=%d\n",
                       result.l2, tolerance, ny);
        }
        status = 1;
      }
    }
  }
  if (root && params.check && status == 0) {
    std::printf("channel %s check passed\n", params.case_name.c_str());
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
