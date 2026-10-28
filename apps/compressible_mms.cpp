// 3D compressible Navier-Stokes verification.
//
// Two manufactured/exact cases exercise the full path (reconstruction +
// Riemann flux + viscous fluxes + RK):
//
//   entropy : rho = 1 + A sin(k.(x) - omega t), u = const, p = const.
//             Exact for the inviscid Euler equations; mu must be 0.
//
//   shear   : u = A sin(ky y) sin(kz z) exp(-nu (ky^2+kz^2) t),
//             v = w = 0, rho = 1, p = 1.
//             Exact for compressible Navier-Stokes with constant mu when the
//             viscous heating is removed by the isothermal energy sink. The
//             sink is a verification device, not a production model.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"
#include "fields/patch_field.hpp"
#include "halo/box_exchange.hpp"
#include "mesh/decomposition.hpp"
#include "mesh/uniform_grid.hpp"
#include "numerics/compressible_ns.hpp"
#include "numerics/reconstruction.hpp"
#include "numerics/riemann.hpp"

#include <algorithm>
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

struct Params {
  std::vector<int> n_list{16, 32, 64};
  std::string case_name = "entropy";
  double gamma = 1.4;
  double gas_constant = 1.0;
  double mu = 0.0;
  double prandtl = 0.72;
  double tend = 0.2;
  double cfl = 0.4;
  double min_order = 3.7;
  double energy_tol = 1.0e-10;
  std::string scheme = "central4";
  std::string solver = "hllc";
  std::string time = "rk4";
  std::string space = "default";
  std::string json;
  bool sink = true;
  bool verbose = false;
  bool check = false;
};

struct CaseResult {
  int n = 0;
  double h = 0.0;
  double dt = 0.0;
  int steps = 0;
  double l2_rho = 0.0;
  double l2_u = 0.0;
  double l2_p = 0.0;
  double order = 0.0;
  double mass_drift = 0.0;
  double energy_drift = 0.0;
  double seconds = 0.0;
};

struct ExactSolution {
  Params params;

  saturn::PrimitiveState state(double x, double y, double z,
                               double time) const {
    if (params.case_name == "shear") {
      const double k = 2.0 * kPi;
      const double nu = params.mu;  // rho = 1
      const double decay = std::exp(-nu * 2.0 * k * k * time);
      const double u = 0.1 * std::sin(k * y) * std::sin(k * z) * decay;
      return saturn::PrimitiveState{1.0, u, 0.0, 0.0, 1.0};
    }
    const double kx = 2.0 * kPi;
    const double ky = 2.0 * kPi;
    const double kz = 2.0 * kPi;
    const double ux = 1.0;
    const double uy = 0.5;
    const double uz = 0.25;
    const double omega = kx * ux + ky * uy + kz * uz;
    const double phase = kx * x + ky * y + kz * z - omega * time;
    const double rho = 1.0 + 0.2 * std::sin(phase);
    return saturn::PrimitiveState{rho, ux, uy, uz, 1.0};
  }
};

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --case NAME            entropy|shear (default entropy)\n"
      << "  --n N                  single resolution\n"
      << "  --n-list N1 N2 ...     resolutions for a convergence sweep\n"
      << "  --tend VALUE           final time (default 0.2)\n"
      << "  --cfl VALUE            CFL (default 0.4)\n"
      << "  --mu VALUE             dynamic viscosity (shear default 0.05)\n"
      << "  --prandtl VALUE        Prandtl number (default 0.72)\n"
      << "  --scheme NAME          central4|central6|weno5z|teno6\n"
      << "  --solver NAME          llf|hllc|hllc_lm|ausm_up\n"
      << "  --time NAME            rk3|rk4 (default rk4)\n"
      << "  --no-sink              disable the isothermal energy sink (shear)\n"
      << "  --min-order VALUE      check threshold (default 3.7)\n"
      << "  --energy-tol VALUE     |dE/E| limit for the inviscid entropy case\n"
      << "                         (default 1e-10)\n"
      << "  --space NAME           serial|openmp|cuda|hip|sycl|default\n"
      << "  --json PATH            write results as JSON\n"
      << "  --check                fail if the observed order is too low\n"
      << "  --verbose              print the Kokkos configuration\n"
      << "  --help                 show this help\n";
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
  bool mu_set = false;
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
    } else if (a == "--tend") {
      p.tend = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--cfl") {
      p.cfl = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--mu") {
      p.mu = parse_double(a, require_value(i, argc, argv, a));
      mu_set = true;
    } else if (a == "--prandtl") {
      p.prandtl = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--scheme") {
      p.scheme = require_value(i, argc, argv, a);
    } else if (a == "--solver") {
      p.solver = require_value(i, argc, argv, a);
    } else if (a == "--time") {
      p.time = require_value(i, argc, argv, a);
    } else if (a == "--no-sink") {
      p.sink = false;
    } else if (a == "--min-order") {
      p.min_order = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--energy-tol") {
      p.energy_tol = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--space") {
      p.space = require_value(i, argc, argv, a);
    } else if (a == "--json") {
      p.json = require_value(i, argc, argv, a);
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

  if (p.case_name != "entropy" && p.case_name != "shear") {
    std::fprintf(stderr, "unknown case '%s'\n", p.case_name.c_str());
    std::exit(2);
  }
  if (p.case_name == "shear" && !mu_set) {
    p.mu = 0.05;
  }
  if (p.case_name == "entropy" && p.mu != 0.0) {
    std::fprintf(stderr,
                 "entropy case is inviscid; set --mu 0 or omit --mu\n");
    std::exit(2);
  }
  if (p.time != "rk3" && p.time != "rk4") {
    std::fprintf(stderr, "unknown time integrator '%s'\n", p.time.c_str());
    std::exit(2);
  }
  for (const int n : p.n_list) {
    if (n < 8) {
      std::fprintf(stderr, "resolution must be >= 8 (got %d)\n", n);
      std::exit(2);
    }
  }
  if (p.tend <= 0.0 || p.cfl <= 0.0 || p.mu < 0.0 || p.prandtl <= 0.0) {
    std::fprintf(stderr, "invalid tend/cfl/mu/prandtl\n");
    std::exit(2);
  }
  return p;
}

saturn::Reconstruction parse_scheme(const std::string& name) {
  if (name == "central4") {
    return saturn::Reconstruction::Central4;
  }
  if (name == "central6") {
    return saturn::Reconstruction::Central6;
  }
  if (name == "teno6") {
    return saturn::Reconstruction::Teno6;
  }
  return saturn::Reconstruction::Weno5Z;
}

const char* scheme_name(saturn::Reconstruction scheme) {
  switch (scheme) {
    case saturn::Reconstruction::Central2:
      return "central2";
    case saturn::Reconstruction::Central4:
      return "central4";
    case saturn::Reconstruction::Central6:
      return "central6";
    case saturn::Reconstruction::Weno5Z:
      return "weno5z";
    case saturn::Reconstruction::Teno6:
      return "teno6";
  }
  return "central4";
}

saturn::RiemannSolver parse_solver(const std::string& name) {
  if (name == "llf") {
    return saturn::RiemannSolver::Llf;
  }
  if (name == "hllc_lm") {
    return saturn::RiemannSolver::HllcLowMach;
  }
  if (name == "ausm_up") {
    return saturn::RiemannSolver::AusmPlusUp;
  }
  return saturn::RiemannSolver::Hllc;
}

const char* solver_name(saturn::RiemannSolver solver) {
  switch (solver) {
    case saturn::RiemannSolver::Llf:
      return "llf";
    case saturn::RiemannSolver::Hllc:
      return "hllc";
    case saturn::RiemannSolver::HllcLowMach:
      return "hllc_lm";
    case saturn::RiemannSolver::AusmPlusUp:
      return "ausm_up";
  }
  return "hllc";
}

template <class ExecSpace>
void copy_field(const saturn::PatchField<ExecSpace>& source,
                saturn::PatchField<ExecSpace>& destination) {
  const saturn::Box box = source.box();
  const auto policy = saturn::make_mdrange3<ExecSpace>(
      saturn::Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      saturn::Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  Kokkos::parallel_for(
      "copy_field", policy,
      KOKKOS_LAMBDA(int k, int j, int i) {
        for (int v = 0; v < 5; ++v) {
          destination(v, i, j, k) = source(v, i, j, k);
        }
      });
}

// destination = a * x + b * y
template <class ExecSpace>
void combine_fields(saturn::PatchField<ExecSpace>& destination, double a,
                    const saturn::PatchField<ExecSpace>& x, double b,
                    const saturn::PatchField<ExecSpace>& y) {
  const saturn::Box box = destination.box();
  const auto policy = saturn::make_mdrange3<ExecSpace>(
      saturn::Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      saturn::Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  Kokkos::parallel_for(
      "combine_fields", policy,
      KOKKOS_LAMBDA(int k, int j, int i) {
        for (int v = 0; v < 5; ++v) {
          destination(v, i, j, k) =
              a * x(v, i, j, k) + b * y(v, i, j, k);
        }
      });
}

// destination += scale * x
template <class ExecSpace>
void add_scaled_field(saturn::PatchField<ExecSpace>& destination,
                      double scale,
                      const saturn::PatchField<ExecSpace>& x) {
  const saturn::Box box = destination.box();
  const auto policy = saturn::make_mdrange3<ExecSpace>(
      saturn::Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      saturn::Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});
  Kokkos::parallel_for(
      "add_scaled_field", policy,
      KOKKOS_LAMBDA(int k, int j, int i) {
        for (int v = 0; v < 5; ++v) {
          destination(v, i, j, k) += scale * x(v, i, j, k);
        }
      });
}

template <class ExecSpace>
CaseResult run_case(int n, const Params& params,
                    const ExactSolution& exact) {
  using field_type = saturn::PatchField<ExecSpace>;
  using workspace_type = saturn::compressible::CompressibleWorkspace<ExecSpace>;

  constexpr int kGhost = 4;
  const bool periodic[3] = {true, true, true};
  const auto decomposition = saturn::Decomposition::create(
      saturn::Box::from_extents(n, n, n), periodic);
  const auto grid = saturn::UniformGrid::from_cells_local(
      n, n, n, 1.0, 1.0, 1.0, kGhost, decomposition.local);
  const saturn::Box box = grid.local_box();

  saturn::compressible::GasProperties gas;
  gas.gamma = params.gamma;
  gas.gas_constant = params.gas_constant;
  gas.mu = params.mu;
  gas.prandtl = params.prandtl;

  const saturn::Reconstruction scheme = parse_scheme(params.scheme);
  const saturn::RiemannSolver solver = parse_solver(params.solver);

  field_type q(box, 5, kGhost, "q");
  field_type stage(box, 5, kGhost, "stage");
  field_type next(box, 5, kGhost, "next");
  field_type rhs(box, 5, 0, "rhs");
  field_type k1(box, 5, 0, "k1");
  field_type k2(box, 5, 0, "k2");
  field_type k3(box, 5, 0, "k3");
  workspace_type workspace(box);
  saturn::BoxExchange<ExecSpace> exchange(decomposition, 5, kGhost);

  auto host_q = Kokkos::create_mirror_view(q.data());
  for (int k = box.lo[2]; k <= box.hi[2]; ++k) {
    for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        const double x = (i + 0.5) * grid.h[0];
        const double y = (j + 0.5) * grid.h[1];
        const double z = (k + 0.5) * grid.h[2];
        const saturn::PrimitiveState state = exact.state(x, y, z, 0.0);
        const saturn::ConservativeState conservative =
            saturn::riemann::to_conservative(state, gas.gamma);
        const int ip = i - box.lo[0] + kGhost;
        const int jp = j - box.lo[1] + kGhost;
        const int kp = k - box.lo[2] + kGhost;
        host_q(0, kp, jp, ip) = conservative.rho;
        host_q(1, kp, jp, ip) = conservative.mom_n;
        host_q(2, kp, jp, ip) = conservative.mom_t1;
        host_q(3, kp, jp, ip) = conservative.mom_t2;
        host_q(4, kp, jp, ip) = conservative.energy;
      }
    }
  }
  Kokkos::deep_copy(q.data(), host_q);

  const auto compute_rhs = [&](field_type& state, field_type& out) {
    exchange.exchange(state);
    saturn::compressible::compressible_rhs(
        state, workspace, out, grid, gas, scheme, solver, params.sink);
  };

  const double hmin = std::min({grid.h[0], grid.h[1], grid.h[2]});
  const double cell_volume = grid.h[0] * grid.h[1] * grid.h[2];
  const auto policy = saturn::make_mdrange3<ExecSpace>(
      saturn::Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      saturn::Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});

  // Conservative totals before the time loop. For periodic inviscid flow the
  // fluxes telescope and these must not change (up to round-off).
  double initial_mass = 0.0;
  double initial_energy = 0.0;
  Kokkos::parallel_reduce(
      "conservation_initial", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& m, double& e) {
        m += q(0, i, j, k);
        e += q(4, i, j, k);
      },
      Kokkos::Sum<double>{initial_mass}, Kokkos::Sum<double>{initial_energy});
  initial_mass = saturn::mpi_sum(initial_mass) * cell_volume;
  initial_energy = saturn::mpi_sum(initial_energy) * cell_volume;

  Kokkos::Timer timer;
  double time = 0.0;
  int steps = 0;
  double last_dt = 0.0;

  while (time < params.tend - 1.0e-14) {
    const double max_speed =
        saturn::compressible::max_wave_speed(q, gas);
    const double max_nu =
        saturn::compressible::max_kinematic_diffusivity(q, gas);
    double dt = params.cfl * hmin / std::max(max_speed, 1.0e-300);
    if (max_nu > 0.0) {
      dt = std::min(dt, 0.2 * hmin * hmin / max_nu);
    }
    if (time + dt > params.tend) {
      dt = params.tend - time;
    }
    last_dt = dt;

    if (params.time == "rk4") {
      compute_rhs(q, rhs);
      copy_field(rhs, k1);
      combine_fields(stage, 1.0, q, 0.5 * dt, k1);
      compute_rhs(stage, rhs);
      copy_field(rhs, k2);
      combine_fields(stage, 1.0, q, 0.5 * dt, k2);
      compute_rhs(stage, rhs);
      copy_field(rhs, k3);
      combine_fields(stage, 1.0, q, dt, k3);
      compute_rhs(stage, rhs);
      add_scaled_field(q, dt / 6.0, k1);
      add_scaled_field(q, dt / 3.0, k2);
      add_scaled_field(q, dt / 3.0, k3);
      add_scaled_field(q, dt / 6.0, rhs);
    } else {
      compute_rhs(q, rhs);
      combine_fields(stage, 1.0, q, dt, rhs);
      compute_rhs(stage, rhs);
      combine_fields(next, 0.75, q, 0.25, stage);
      add_scaled_field(next, 0.25 * dt, rhs);
      compute_rhs(next, rhs);
      combine_fields(q, 1.0 / 3.0, q, 2.0 / 3.0, next);
      add_scaled_field(q, (2.0 / 3.0) * dt, rhs);
    }

    time += dt;
    ++steps;
  }
  Kokkos::fence();
  exchange.exchange(q);

  const double seconds = saturn::mpi_max(timer.seconds());

  double sum_rho = 0.0;
  double sum_u = 0.0;
  double sum_p = 0.0;
  Kokkos::parallel_reduce(
      "mms_errors", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& rho_acc, double& u_acc,
                    double& p_acc) {
        const double x = (i + 0.5) * grid.h[0];
        const double y = (j + 0.5) * grid.h[1];
        const double z = (k + 0.5) * grid.h[2];
        const saturn::PrimitiveState numerical =
            saturn::compressible::load_primitive(q, i, j, k, gas);
        const saturn::PrimitiveState reference =
            exact.state(x, y, z, params.tend);
        const double drho = numerical.rho - reference.rho;
        const double du = numerical.u - reference.u;
        const double dp = numerical.p - reference.p;
        rho_acc += drho * drho;
        u_acc += du * du;
        p_acc += dp * dp;
      },
      Kokkos::Sum<double>{sum_rho}, Kokkos::Sum<double>{sum_u},
      Kokkos::Sum<double>{sum_p});

  const double cells = static_cast<double>(n) * n * n;
  const double l2_rho = std::sqrt(saturn::mpi_sum(sum_rho) / cells);
  const double l2_u = std::sqrt(saturn::mpi_sum(sum_u) / cells);
  const double l2_p = std::sqrt(saturn::mpi_sum(sum_p) / cells);

  double final_mass = 0.0;
  double final_energy = 0.0;
  Kokkos::parallel_reduce(
      "conservation_final", policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& m, double& e) {
        m += q(0, i, j, k);
        e += q(4, i, j, k);
      },
      Kokkos::Sum<double>{final_mass}, Kokkos::Sum<double>{final_energy});
  final_mass = saturn::mpi_sum(final_mass) * cell_volume;
  final_energy = saturn::mpi_sum(final_energy) * cell_volume;

  CaseResult result;
  result.n = n;
  result.h = hmin;
  result.dt = last_dt;
  result.steps = steps;
  result.l2_rho = l2_rho;
  result.l2_u = l2_u;
  result.l2_p = l2_p;
  result.mass_drift =
      std::abs(final_mass - initial_mass) / std::abs(initial_mass);
  result.energy_drift =
      std::abs(final_energy - initial_energy) / std::abs(initial_energy);
  result.seconds = seconds;
  return result;
}

template <class ExecSpace>
int run_sweep(const Params& params, const ExactSolution& exact) {
  std::vector<CaseResult> results;
  double previous_error = 0.0;
  double previous_h = 0.0;
  const bool root = saturn::mpi_rank() == 0;

  if (root) {
    std::cout << "\n" << saturn::kProjectName << " " << saturn::kVersion
              << " compressible " << params.case_name << " MMS\n"
              << "scheme: " << scheme_name(parse_scheme(params.scheme))
              << ", solver: " << solver_name(parse_solver(params.solver))
              << ", time: " << params.time << ", mu: " << params.mu << "\n"
              << "Kokkos: " << ExecSpace::name() << " (concurrency "
              << ExecSpace{}.concurrency() << ")\n";
    if (params.verbose) {
      Kokkos::print_configuration(std::cout, true);
    }
    std::cout << std::left << std::setw(6) << "N" << std::setw(12) << "h"
              << std::setw(12) << "dt" << std::setw(8) << "steps"
              << std::setw(14) << "L2 rho" << std::setw(14) << "L2 u"
              << std::setw(14) << "L2 p" << std::setw(8) << "order"
              << std::setw(12) << "dM/M" << std::setw(12) << "dE/E"
              << "time [s]\n";
  }

  for (const int n : params.n_list) {
    CaseResult result = run_case<ExecSpace>(n, params, exact);
    const double error = params.case_name == "shear" ? result.l2_u
                                                     : result.l2_rho;
    if (previous_error > 0.0 && previous_h > 0.0) {
      result.order =
          std::log(previous_error / error) / std::log(previous_h / result.h);
    }
    previous_error = error;
    previous_h = result.h;
    results.push_back(result);

    if (root) {
      std::cout << std::left << std::setw(6) << result.n << std::setw(12)
                << std::scientific << std::setprecision(3) << result.h
                << std::setw(12) << result.dt << std::setw(8) << result.steps
                << std::setw(14) << result.l2_rho << std::setw(14)
                << result.l2_u << std::setw(14) << result.l2_p
                << std::setw(8) << std::fixed << std::setprecision(2)
                << result.order << std::scientific << std::setprecision(2)
                << std::setw(12) << result.mass_drift
                << std::setw(12) << result.energy_drift
                << std::fixed << std::setprecision(3)
                << result.seconds << "\n";
    }
  }

  if (root && !params.json.empty()) {
    const std::filesystem::path path(params.json);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(params.json);
    if (!out) {
      std::fprintf(stderr, "could not open %s for writing\n",
                   params.json.c_str());
      return 2;
    }
    out << std::setprecision(10) << "[\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
      const auto& result = results[i];
      out << "  {\n"
          << "    \"n\": " << result.n << ",\n"
          << "    \"h\": " << result.h << ",\n"
          << "    \"dt\": " << result.dt << ",\n"
          << "    \"steps\": " << result.steps << ",\n"
          << "    \"l2_rho\": " << result.l2_rho << ",\n"
          << "    \"l2_u\": " << result.l2_u << ",\n"
          << "    \"l2_p\": " << result.l2_p << ",\n"
          << "    \"order\": " << result.order << ",\n"
          << "    \"mass_drift\": " << result.mass_drift << ",\n"
          << "    \"energy_drift\": " << result.energy_drift << ",\n"
          << "    \"seconds\": " << result.seconds << "\n"
          << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "]\n";
    std::cout << "json: " << params.json << "\n";
  }

  if (params.check) {
    bool order_ok = true;
    if (results.size() >= 2) {
      for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i].order < params.min_order) {
          order_ok = false;
        }
      }
    }

    // Periodic inviscid flow must conserve mass and total energy to
    // round-off; the conservative face fluxes telescope exactly.
    bool conservation_ok = true;
    double max_mass_drift = 0.0;
    double max_energy_drift = 0.0;
    if (params.case_name == "entropy") {
      for (const auto& result : results) {
        max_mass_drift = std::max(max_mass_drift, result.mass_drift);
        max_energy_drift = std::max(max_energy_drift, result.energy_drift);
        if (result.mass_drift > params.energy_tol ||
            result.energy_drift > params.energy_tol) {
          conservation_ok = false;
        }
      }
    }

    const bool ok = saturn::mpi_all(order_ok && conservation_ok);
    if (!ok) {
      if (root) {
        if (!order_ok) {
          std::fprintf(stderr,
                       "compressible MMS check failed: observed order below "
                       "%.2f\n",
                       params.min_order);
        }
        if (!conservation_ok) {
          std::fprintf(stderr,
                       "compressible MMS check failed: conservation drift "
                       "above %.2e (mass %.2e, energy %.2e)\n",
                       params.energy_tol, max_mass_drift, max_energy_drift);
        }
      }
      return 1;
    }
    if (root) {
      if (params.case_name == "entropy") {
        std::printf(
            "compressible entropy MMS check passed "
            "(max |dM/M| = %.2e, max |dE/E| = %.2e)\n",
            max_mass_drift, max_energy_drift);
      } else {
        std::printf("compressible %s MMS check passed\n",
                    params.case_name.c_str());
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Params params = parse(argc, argv);
  ExactSolution exact;
  exact.params = params;

#ifdef SATURN_ENABLE_MPI
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif

  Kokkos::initialize(argc, argv);
  const int status = saturn::dispatch_execution_space(
      params.space, SATURN_DEFAULT_SPACE, [&](auto tag) {
        using space_type = decltype(tag);
        return run_sweep<space_type>(params, exact);
      });
  Kokkos::finalize();

#ifdef SATURN_ENABLE_MPI
  MPI_Finalize();
#endif
  return status;
}
