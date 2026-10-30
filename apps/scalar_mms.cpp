// Second-order scalar advection-diffusion verification with a manufactured
// traveling-wave solution on a periodic 3D box.
//
//   c_t + u . grad(c) = nu * laplacian(c)
//   c(x,t) = 1 + 0.5 exp(-nu |k|^2 t) sin(k.x - omega t),  omega = k.u
//
// This exercises the mesh/field/ghost-fill/RK3 path and measures the observed
// spatial order. It is the first quantitative gate for Phase 1.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"
#include "fields/patch_field.hpp"
#include "halo/face_exchange.hpp"
#include "io/vtk_writer.hpp"
#include "mesh/decomposition.hpp"
#include "mesh/uniform_grid.hpp"
#include "numerics/scalar_advection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SATURN_DEFAULT_SPACE
#define SATURN_DEFAULT_SPACE "default"
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kKx = 1.0;
constexpr double kKy = 2.0;
constexpr double kKz = 1.0;
constexpr double kVx = 1.0;
constexpr double kVy = 0.5;
constexpr double kVz = 0.25;
constexpr double kOmega = kKx * kVx + kKy * kVy + kKz * kVz;
constexpr double kK2 = kKx * kKx + kKy * kKy + kKz * kKz;

struct Params {
  std::vector<int> n_list{16, 32, 64, 128};
  double nu = 1.0e-2;
  double tend = 0.2;
  double cfl = 0.2;
  std::string space = "default";
  std::string json;
  std::string output;
  bool verbose = false;
  bool check = false;
  bool overlap = false;
};

struct CaseResult {
  int n = 0;
  double h = 0.0;
  double dt = 0.0;
  int steps = 0;
  double l2 = 0.0;
  double linf = 0.0;
  double mass_drift = 0.0;
  double seconds = 0.0;
  double order = 0.0;
};

KOKKOS_INLINE_FUNCTION
double exactScalar(double x, double y, double z, double t, double nu) {
  const double phase = kKx * x + kKy * y + kKz * z - kOmega * t;
  return 1.0 + 0.5 * Kokkos::exp(-nu * kK2 * t) * Kokkos::sin(phase);
}

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --n N                  single resolution\n"
            << "  --n-list N1 N2 ...     resolutions for a convergence sweep\n"
            << "  --nu VALUE             diffusivity (default 0.01)\n"
            << "  --tend VALUE           final time (default 0.2)\n"
            << "  --cfl VALUE            advective CFL (default 0.2)\n"
            << "  --space NAME           serial|openmp|cuda|hip|sycl|default\n"
            << "  --json PATH            write results as JSON\n"
            << "  --output BASE          write the final field as VTK "
               "(BASE.vti/.pvti)\n"
            << "  --overlap              overlap halo exchange with interior "
               "RHS (opt-in)\n"
            << "  --check                fail if observed order < 1.7 or mass "
               "drift > 1e-10\n"
            << "  --verbose              print the full Kokkos configuration\n"
            << "  --help                 show this help\n";
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
    } else if (a == "--nu") {
      p.nu = parseDouble(a, requireValue(i, argc, argv, a));
    } else if (a == "--tend") {
      p.tend = parseDouble(a, requireValue(i, argc, argv, a));
    } else if (a == "--cfl") {
      p.cfl = parseDouble(a, requireValue(i, argc, argv, a));
    } else if (a == "--space") {
      p.space = requireValue(i, argc, argv, a);
    } else if (a == "--json") {
      p.json = requireValue(i, argc, argv, a);
    } else if (a == "--output") {
      p.output = requireValue(i, argc, argv, a);
    } else if (a == "--overlap") {
      p.overlap = true;
    } else if (a == "--verbose") {
      p.verbose = true;
    } else if (a == "--check") {
      p.check = true;
    } else if (a.rfind("--kokkos-", 0) == 0) {
      // Handled by Kokkos::initialize().
    } else {
      std::fprintf(stderr, "warning: ignoring unknown argument '%s'\n",
                   a.c_str());
    }
  }

  for (int n : p.n_list) {
    if (n < 8) {
      std::fprintf(stderr, "resolution must be >= 8 (got %d)\n", n);
      std::exit(2);
    }
  }
  if (p.nu < 0.0 || p.tend <= 0.0 || p.cfl <= 0.0) {
    std::fprintf(stderr, "nu must be >= 0, tend/cfl must be > 0\n");
    std::exit(2);
  }
  return p;
}

template <class ExecSpace>
CaseResult runCase(int n, const Params& p) {
  using field_type = saturn::PatchField<ExecSpace>;

  const int ng = 1;
  const double nu = p.nu;
  const double tend = p.tend;
  const bool periodic[3] = {true, true, true};
  const auto decomposition = saturn::Decomposition::create(
      saturn::Box::fromExtents(n, n, n), periodic);
  const auto grid = saturn::UniformGrid::fromCellsLocal(
      n, n, n, 2.0 * kPi, 2.0 * kPi, 2.0 * kPi, ng, decomposition.local);
  const saturn::Box box = grid.localBox();

  field_type q0(box, 1, ng, "q0");
  field_type q1(box, 1, ng, "q1");
  field_type q2(box, 1, ng, "q2");
  field_type rhs(box, 1, ng, "rhs");
  saturn::FaceExchange<ExecSpace> exchange(decomposition, 1, ng);

  // Initial condition on the host mirror.
  auto hq = Kokkos::create_mirror_view(q0.data());
  for (int k = box.lo[2]; k <= box.hi[2]; ++k) {
    for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        const double x = (i + 0.5) * grid.h[0];
        const double y = (j + 0.5) * grid.h[1];
        const double z = (k + 0.5) * grid.h[2];
        hq(0, k - box.lo[2] + ng, j - box.lo[1] + ng, i - box.lo[0] + ng) =
            exactScalar(x, y, z, 0.0, p.nu);
      }
    }
  }
  Kokkos::deep_copy(q0.data(), hq);

  const std::array<double, 3> velocity{{kVx, kVy, kVz}};
  const double hmin = std::min({grid.h[0], grid.h[1], grid.h[2]});
  const double umax = std::abs(kVx) + std::abs(kVy) + std::abs(kVz);
  const double dt_adv = p.cfl * hmin / umax;
  const double dt_visc = 0.1 * hmin * hmin / std::max(p.nu, 1.0e-30);
  double dt = std::min(dt_adv, dt_visc);
  const int steps = std::max(1, static_cast<int>(std::ceil(p.tend / dt)));
  dt = p.tend / steps;

  const double cellVolume = grid.h[0] * grid.h[1] * grid.h[2];
  const auto full_policy = saturn::makeMdrange3<ExecSpace>(
      saturn::Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      saturn::Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  const saturn::Box interior_box = box.shrink(1);

  auto mass = [&](const field_type& f) {
    double sum = 0.0;
    Kokkos::parallel_reduce(
        "mass", full_policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
          acc += f(0, i, j, k);
        },
        sum);
    return sum * cellVolume;
  };

  const double mass0 = saturn::mpiSum(mass(q0));
  Kokkos::Timer timer;

  auto computeRhs = [&](field_type& q) {
    if (p.overlap) {
      exchange.begin(q);
      saturn::scalarAdvectionDiffusionRhs(q, rhs, velocity, p.nu, grid.h,
                                          interior_box);
      exchange.end(q);
      saturn::scalarAdvectionDiffusionRhsShell(q, rhs, velocity, p.nu, grid.h,
                                               box);
    } else {
      exchange.exchange(q);
      saturn::scalarAdvectionDiffusionRhs(q, rhs, velocity, p.nu, grid.h);
    }
  };

  for (int step = 0; step < steps; ++step) {
    computeRhs(q0);

    Kokkos::parallel_for(
        "rk3_stage1", full_policy, KOKKOS_LAMBDA(int k, int j, int i) {
          q1(0, i, j, k) = q0(0, i, j, k) + dt * rhs(0, i, j, k);
        });

    computeRhs(q1);

    Kokkos::parallel_for(
        "rk3_stage2", full_policy, KOKKOS_LAMBDA(int k, int j, int i) {
          q2(0, i, j, k) = 0.75 * q0(0, i, j, k) +
                           0.25 * (q1(0, i, j, k) + dt * rhs(0, i, j, k));
        });

    computeRhs(q2);

    Kokkos::parallel_for(
        "rk3_stage3", full_policy, KOKKOS_LAMBDA(int k, int j, int i) {
          q1(0, i, j, k) =
              (1.0 / 3.0) * q0(0, i, j, k) +
              (2.0 / 3.0) * (q2(0, i, j, k) + dt * rhs(0, i, j, k));
        });

    std::swap(q0, q1);
  }

  exchange.exchange(q0);
  ExecSpace().fence();
  const double seconds = saturn::mpiMax(timer.seconds());
  const double mass1 = saturn::mpiSum(mass(q0));

  if (!p.output.empty()) {
    const std::array<double, 3> origin{{0.0, 0.0, 0.0}};
    const bool ok =
        saturn::writeVti(p.output, q0, origin, grid.h, grid.global, tend);
    if (!ok && saturn::mpiRank() == 0) {
      std::fprintf(stderr, "VTK output failed for base '%s'\n",
                   p.output.c_str());
    }
  }

  double l2 = 0.0;
  Kokkos::parallel_reduce(
      "l2", full_policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
        const double x = (i + 0.5) * (2.0 * kPi / static_cast<double>(n));
        const double y = (j + 0.5) * (2.0 * kPi / static_cast<double>(n));
        const double z = (k + 0.5) * (2.0 * kPi / static_cast<double>(n));
        const double e = q0(0, i, j, k) - exactScalar(x, y, z, tend, nu);
        acc += e * e;
      },
      l2);
  l2 = std::sqrt(saturn::mpiSum(l2) / (static_cast<double>(n) * n * n));

  double linf = 0.0;
  Kokkos::parallel_reduce(
      "linf", full_policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& acc) {
        const double x = (i + 0.5) * (2.0 * kPi / static_cast<double>(n));
        const double y = (j + 0.5) * (2.0 * kPi / static_cast<double>(n));
        const double z = (k + 0.5) * (2.0 * kPi / static_cast<double>(n));
        const double e = q0(0, i, j, k) - exactScalar(x, y, z, tend, nu);
        const double a = Kokkos::abs(e);
        acc = (a > acc) ? a : acc;
      },
      Kokkos::Max<double>{linf});
  linf = saturn::mpiMax(linf);

  CaseResult result;
  result.n = n;
  result.h = hmin;
  result.dt = dt;
  result.steps = steps;
  result.l2 = l2;
  result.linf = linf;
  result.mass_drift = (mass1 - mass0) / std::max(std::abs(mass0), 1.0e-300);
  result.seconds = seconds;
  return result;
}

template <class ExecSpace>
int runSweep(const Params& p) {
  std::vector<CaseResult> results;
  double previous_error = 0.0;
  double previous_h = 0.0;
  const bool root = saturn::mpiRank() == 0;

  if (root) {
    std::cout << "\n"
              << saturn::kProjectName << " " << saturn::kVersion
              << " scalar MMS\n"
              << "Kokkos: " << ExecSpace::name() << " (concurrency "
              << ExecSpace{}.concurrency() << ")\n";
    if (p.verbose) {
      Kokkos::print_configuration(std::cout, true);
    }
    std::cout << std::left << std::setw(6) << "N" << std::setw(12) << "h"
              << std::setw(12) << "dt" << std::setw(8) << "steps"
              << std::setw(14) << "L2" << std::setw(8) << "order"
              << std::setw(14) << "Linf" << std::setw(14) << "mass drift"
              << "time [s]\n";
  }

  for (int n : p.n_list) {
    CaseResult r = runCase<ExecSpace>(n, p);
    if (previous_error > 0.0 && previous_h > 0.0) {
      r.order = std::log(previous_error / r.l2) / std::log(previous_h / r.h);
    }
    previous_error = r.l2;
    previous_h = r.h;
    results.push_back(r);

    if (root) {
      std::cout << std::left << std::setw(6) << r.n << std::setw(12)
                << std::scientific << std::setprecision(3) << r.h
                << std::setw(12) << r.dt << std::setw(8) << r.steps
                << std::setw(14) << r.l2 << std::setw(8) << std::fixed
                << std::setprecision(2) << r.order << std::setw(14)
                << std::scientific << r.linf << std::setw(14) << r.mass_drift
                << std::fixed << std::setprecision(3) << r.seconds << "\n";
    }
  }

  if (root && !p.json.empty()) {
    const std::filesystem::path path(p.json);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(p.json);
    if (!out) {
      std::fprintf(stderr, "could not open %s for writing\n", p.json.c_str());
      return 2;
    }
    out << std::setprecision(10) << "[\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
      const auto& r = results[i];
      out << "  {\n"
          << "    \"n\": " << r.n << ",\n"
          << "    \"h\": " << r.h << ",\n"
          << "    \"dt\": " << r.dt << ",\n"
          << "    \"steps\": " << r.steps << ",\n"
          << "    \"l2\": " << r.l2 << ",\n"
          << "    \"linf\": " << r.linf << ",\n"
          << "    \"order\": " << r.order << ",\n"
          << "    \"mass_drift\": " << r.mass_drift << ",\n"
          << "    \"seconds\": " << r.seconds << "\n"
          << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "]\n";
    std::cout << "json: " << p.json << "\n";
  }

  if (p.check) {
    bool ok = results.size() >= 2;
    for (std::size_t i = 1; i < results.size(); ++i) {
      if (results[i].order < 1.7) {
        ok = false;
      }
    }
    for (const auto& r : results) {
      if (std::abs(r.mass_drift) > 1.0e-10) {
        ok = false;
      }
    }
    ok = saturn::mpiAll(ok);
    if (!ok) {
      if (root) {
        std::fprintf(stderr,
                     "scalar MMS check failed: order or mass conservation "
                     "out of tolerance\n");
      }
      return 1;
    }
    if (root) {
      std::printf("scalar MMS check passed\n");
    }
  }

  return 0;
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
        return runSweep<space_type>(params);
      });
  Kokkos::finalize();

#ifdef SATURN_ENABLE_MPI
  MPI_Finalize();
#endif
  return status;
}
