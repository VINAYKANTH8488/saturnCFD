// Production-style 3D compressible Navier-Stokes driver.
//
// Cases:
//   tgv        3D Taylor-Green vortex on a periodic box
//   acoustic   small pressure pulse on a periodic box
//   vortex     low-Mach isentropic vortex on a periodic box
//   dm         double Mach reflection setup (single-rank; approximate BCs)
//
// The driver supports VTK snapshots of primitive fields and binary
// checkpoints. Multi-rank runs currently require all-periodic boundaries;
// single-rank runs may use transmissive/slip boundaries as well.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"
#include "fields/boundary_conditions.hpp"
#include "fields/patch_field.hpp"
#include "halo/box_exchange.hpp"
#include "halo/global_exchange.hpp"
#include "io/checkpoint.hpp"
#include "io/vtk_writer.hpp"
#include "mesh/decomposition.hpp"
#include "mesh/uniform_grid.hpp"
#include "numerics/compressible_ns.hpp"
#include "numerics/reconstruction.hpp"
#include "numerics/riemann.hpp"
#include "numerics/time_integration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef SATURN_DEFAULT_SPACE
#define SATURN_DEFAULT_SPACE "default"
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Params {
  std::string case_name = "tgv";
  int nx = 32;
  int ny = 32;
  int nz = 32;
  double lx = 2.0 * kPi;
  double ly = 2.0 * kPi;
  double lz = 2.0 * kPi;
  double tend = 1.0;
  double cfl = 0.4;
  double gamma = 1.4;
  double gas_constant = 1.0;
  double mu = 1.0 / 1600.0;
  double prandtl = 0.72;
  double energy_tol = 1.0e-10;
  std::string scheme = "central4";
  std::string solver = "hllc";
  std::string time = "rk3";
  std::string space = "default";
  std::string output;
  std::string checkpoint;
  std::string restart;
  std::string history;
  int plot_every = 100;
  int checkpoint_every = 0;
  int diag_every = 0;
  bool verbose = false;
  bool check = false;
  saturn::BoundaryType bc[3][2] = {
      {saturn::BoundaryType::Periodic, saturn::BoundaryType::Periodic},
      {saturn::BoundaryType::Periodic, saturn::BoundaryType::Periodic},
      {saturn::BoundaryType::Periodic, saturn::BoundaryType::Periodic}};
};

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --case NAME          tgv|acoustic|vortex|dm (default tgv)\n"
      << "  --n N                cubic resolution (sets nx=ny=nz)\n"
      << "  --nx/--ny/--nz N     per-dimension resolution\n"
      << "  --tend T             final time\n"
      << "  --cfl C              CFL (default 0.4)\n"
      << "  --gamma G            ratio of specific heats (1.4)\n"
      << "  --mu VALUE           dynamic viscosity (case default)\n"
      << "  --prandtl VALUE      Prandtl number (0.72)\n"
      << "  --scheme NAME        central4|central6|weno5z|teno6\n"
      << "  --solver NAME        llf|hllc|hllc_lm|ausm_up\n"
      << "  --time NAME          rk3|rk4 (default rk3)\n"
      << "  --output BASE        write primitive VTK snapshots\n"
      << "  --plot-every N       snapshot interval (default 100 steps)\n"
      << "  --checkpoint BASE    write binary checkpoints\n"
      << "  --checkpoint-every N checkpoint interval (default 0 = off)\n"
      << "  --history FILE       append global diagnostics as CSV\n"
      << "  --diag-every N       diagnostics interval (default 0 = snapshots)\n"
      << "  --restart BASE       restart from a checkpoint\n"
      << "  --space NAME         serial|openmp|cuda|hip|sycl|default\n"
      << "  --check              fail if the final state is invalid or a\n"
      << "                       periodic inviscid run drifts in energy\n"
      << "  --energy-tol VALUE   |dE/E| limit for the check (default 1e-10)\n"
      << "  --verbose            print the Kokkos configuration\n"
      << "  --help               show this help\n";
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

void configure_case(Params& p, bool mu_set, bool scheme_set) {
  if (p.case_name == "tgv") {
    p.lx = p.ly = p.lz = 2.0 * kPi;
    p.nx = p.ny = p.nz = 32;
    p.tend = 1.0;
    p.mu = mu_set ? p.mu : 1.0 / 1600.0;
    p.scheme = scheme_set ? p.scheme : "central4";
    for (int d = 0; d < 3; ++d) {
      p.bc[d][0] = p.bc[d][1] = saturn::BoundaryType::Periodic;
    }
  } else if (p.case_name == "acoustic") {
    p.lx = p.ly = p.lz = 1.0;
    p.nx = p.ny = p.nz = 32;
    p.tend = 0.5;
    p.mu = mu_set ? p.mu : 0.0;
    p.scheme = scheme_set ? p.scheme : "central4";
    for (int d = 0; d < 3; ++d) {
      p.bc[d][0] = p.bc[d][1] = saturn::BoundaryType::Periodic;
    }
  } else if (p.case_name == "vortex") {
    p.lx = 10.0;
    p.ly = 10.0;
    p.lz = 1.0;
    p.nx = 64;
    p.ny = 64;
    p.nz = 4;
    p.tend = 1.0;
    p.mu = mu_set ? p.mu : 0.0;
    p.scheme = scheme_set ? p.scheme : "central4";
    for (int d = 0; d < 3; ++d) {
      p.bc[d][0] = p.bc[d][1] = saturn::BoundaryType::Periodic;
    }
  } else if (p.case_name == "dm") {
    p.lx = 4.0;
    p.ly = 1.0;
    p.lz = 0.25;
    p.nx = 128;
    p.ny = 32;
    p.nz = 4;
    p.tend = 0.1;
    p.mu = mu_set ? p.mu : 0.0;
    p.scheme = scheme_set ? p.scheme : "weno5z";
    p.bc[0][0] = saturn::BoundaryType::Transmissive;
    p.bc[0][1] = saturn::BoundaryType::Transmissive;
    p.bc[1][0] = saturn::BoundaryType::SlipWall;
    p.bc[1][1] = saturn::BoundaryType::Transmissive;
    p.bc[2][0] = saturn::BoundaryType::Periodic;
    p.bc[2][1] = saturn::BoundaryType::Periodic;
  } else {
    std::fprintf(stderr, "unknown case '%s'\n", p.case_name.c_str());
    std::exit(2);
  }
}

Params parse(int argc, char** argv) {
  Params p;
  bool n_set = false;
  bool nx_set = false;
  bool ny_set = false;
  bool nz_set = false;
  bool tend_set = false;
  bool mu_set = false;
  bool scheme_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--case") {
      p.case_name = require_value(i, argc, argv, a);
    } else if (a == "--n") {
      p.nx = p.ny = p.nz = parse_int(a, require_value(i, argc, argv, a));
      n_set = true;
    } else if (a == "--nx") {
      p.nx = parse_int(a, require_value(i, argc, argv, a));
      nx_set = true;
    } else if (a == "--ny") {
      p.ny = parse_int(a, require_value(i, argc, argv, a));
      ny_set = true;
    } else if (a == "--nz") {
      p.nz = parse_int(a, require_value(i, argc, argv, a));
      nz_set = true;
    } else if (a == "--tend") {
      p.tend = parse_double(a, require_value(i, argc, argv, a));
      tend_set = true;
    } else if (a == "--cfl") {
      p.cfl = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--gamma") {
      p.gamma = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--mu") {
      p.mu = parse_double(a, require_value(i, argc, argv, a));
      mu_set = true;
    } else if (a == "--prandtl") {
      p.prandtl = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--scheme") {
      p.scheme = require_value(i, argc, argv, a);
      scheme_set = true;
    } else if (a == "--solver") {
      p.solver = require_value(i, argc, argv, a);
    } else if (a == "--time") {
      p.time = require_value(i, argc, argv, a);
    } else if (a == "--output") {
      p.output = require_value(i, argc, argv, a);
    } else if (a == "--plot-every") {
      p.plot_every = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--checkpoint") {
      p.checkpoint = require_value(i, argc, argv, a);
    } else if (a == "--checkpoint-every") {
      p.checkpoint_every = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--history") {
      p.history = require_value(i, argc, argv, a);
    } else if (a == "--diag-every") {
      p.diag_every = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--energy-tol") {
      p.energy_tol = parse_double(a, require_value(i, argc, argv, a));
    } else if (a == "--restart") {
      p.restart = require_value(i, argc, argv, a);
    } else if (a == "--space") {
      p.space = require_value(i, argc, argv, a);
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

  if (n_set && (nx_set || ny_set || nz_set)) {
    std::fprintf(stderr, "do not combine --n with --nx/--ny/--nz\n");
    std::exit(2);
  }
  const int user_nx = p.nx;
  const int user_ny = p.ny;
  const int user_nz = p.nz;
  const double user_tend = p.tend;
  configure_case(p, mu_set, scheme_set);
  if (n_set) {
    p.nx = p.ny = p.nz = user_nx;
  } else {
    if (nx_set) {
      p.nx = user_nx;
    }
    if (ny_set) {
      p.ny = user_ny;
    }
    if (nz_set) {
      p.nz = user_nz;
    }
  }
  if (tend_set) {
    p.tend = user_tend;
  }
  if (p.nx < 8 || p.ny < 8 || p.nz < 4) {
    std::fprintf(stderr, "resolution too small (need nx,ny >= 8, nz >= 4)\n");
    std::exit(2);
  }
  if (p.tend <= 0.0 || p.cfl <= 0.0 || p.mu < 0.0 || p.prandtl <= 0.0 ||
      p.gamma <= 1.0) {
    std::fprintf(stderr, "invalid time/CFL/mu/Prandtl/gamma\n");
    std::exit(2);
  }
  if (p.time != "rk3" && p.time != "rk4") {
    std::fprintf(stderr, "unknown time integrator '%s'\n", p.time.c_str());
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

saturn::PrimitiveState initial_state(const Params& p, double x, double y,
                                     double z) {
  if (p.case_name == "tgv") {
    const double rho = 1.0;
    const double u = std::sin(x) * std::cos(y) * std::cos(z);
    const double v = -std::cos(x) * std::sin(y) * std::cos(z);
    const double w = 0.0;
    const double pressure =
        1.0 / p.gamma +
        (std::cos(2.0 * x) + std::cos(2.0 * y)) *
            (std::cos(2.0 * z) + 2.0) / 16.0;
    return saturn::PrimitiveState{rho, u, v, w, pressure};
  }
  if (p.case_name == "acoustic") {
    const double dx = x - 0.5 * p.lx;
    const double dy = y - 0.5 * p.ly;
    const double dz = z - 0.5 * p.lz;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double amplitude = 1.0e-3 * std::exp(-r2 / (0.1 * 0.1));
    return saturn::PrimitiveState{1.0 + amplitude, 0.0, 0.0, 0.0,
                                  1.0 + p.gamma * amplitude};
  }
  if (p.case_name == "vortex") {
    const double x0 = 0.5 * p.lx;
    const double y0 = 0.5 * p.ly;
    const double beta = 0.1;
    const double dx = x - x0;
    const double dy = y - y0;
    const double r2 = dx * dx + dy * dy;
    const double g = std::exp(0.5 * (1.0 - r2));
    const double u = 0.1 - beta / (2.0 * kPi) * dy * g;
    const double v = beta / (2.0 * kPi) * dx * g;
    const double temperature =
        1.0 - (p.gamma - 1.0) * beta * beta / (8.0 * p.gamma * kPi * kPi) *
                  std::exp(1.0 - r2);
    const double rho = std::pow(temperature, 1.0 / (p.gamma - 1.0));
    const double pressure = rho * temperature;
    return saturn::PrimitiveState{rho, u, v, 0.0, pressure};
  }
  // Double Mach reflection: post-shock state for x < 1/6, pre-shock after.
  const bool post_shock = x < 1.0 / 6.0;
  if (post_shock) {
    return saturn::PrimitiveState{8.0, 8.25, 0.0, 0.0, 116.5};
  }
  return saturn::PrimitiveState{1.4, 0.0, 0.0, 0.0, 1.0};
}

template <class ExecSpace>
int run(const Params& params) {
  using field_type = saturn::PatchField<ExecSpace>;
  using workspace_type = saturn::compressible::CompressibleWorkspace<ExecSpace>;

  bool all_periodic = true;
  bool periodic_flags[3];
  for (int d = 0; d < 3; ++d) {
    periodic_flags[d] = params.bc[d][0] == saturn::BoundaryType::Periodic &&
                        params.bc[d][1] == saturn::BoundaryType::Periodic;
    all_periodic = all_periodic && periodic_flags[d];
  }
  const int mpi_size = saturn::mpi_size();

  constexpr int kGhost = 4;
  const auto decomposition = saturn::Decomposition::create(
      saturn::Box::from_extents(params.nx, params.ny, params.nz),
      periodic_flags);
  const auto grid = saturn::UniformGrid::from_cells_local(
      params.nx, params.ny, params.nz, params.lx, params.ly, params.lz, kGhost,
      decomposition.local);
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
  saturn::GlobalExchange<ExecSpace> global_exchange(decomposition, 5, kGhost);

  int step = 0;
  double time = 0.0;
  if (!params.restart.empty()) {
    if (!saturn::read_checkpoint(params.restart, q, step, time,
                                 decomposition)) {
      if (saturn::mpi_rank() == 0) {
        std::fprintf(stderr, "could not restart from %s\n",
                     params.restart.c_str());
      }
      return 2;
    }
  } else {
    auto host_q = Kokkos::create_mirror_view(q.data());
    for (int k = box.lo[2]; k <= box.hi[2]; ++k) {
      for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          const double x = (i + 0.5) * grid.h[0];
          const double y = (j + 0.5) * grid.h[1];
          const double z = (k + 0.5) * grid.h[2];
          const saturn::PrimitiveState state = initial_state(params, x, y, z);
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
  }

  const bool use_box_exchange = mpi_size > 1 && all_periodic;
  const bool use_global_exchange = mpi_size > 1 && !all_periodic;
  auto compute_rhs = [&](field_type& state, field_type& out) {
    if (use_box_exchange) {
      exchange.exchange(state);
    } else if (use_global_exchange) {
      global_exchange.exchange(state, params.bc);
    } else {
      saturn::apply_boundary_conditions(state, params.bc);
    }
    saturn::compressible::compressible_rhs(
        state, workspace, out, grid, gas, scheme, solver, false);
  };

  const double hmin = std::min({grid.h[0], grid.h[1], grid.h[2]});
  const double cell_volume = grid.h[0] * grid.h[1] * grid.h[2];
  const auto interior_policy = saturn::make_mdrange3<ExecSpace>(
      saturn::Int3{{box.lo[2], box.lo[1], box.lo[0]}},
      saturn::Int3{{box.hi[2] + 1, box.hi[1] + 1, box.hi[0] + 1}});

  const bool root = saturn::mpi_rank() == 0;
  std::ofstream history;
  if (!params.history.empty() && root) {
    const std::filesystem::path path(params.history);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    history.open(params.history);
    if (!history) {
      std::fprintf(stderr, "could not open history file %s\n",
                   params.history.c_str());
      return 2;
    }
    history << "t,step,mass,momentum,kinetic,internal,total_energy,max_mach\n";
  }

  // Returns {mass, momentum, kinetic, internal, total, max Mach}; the total
  // energy is kinetic + internal and is conserved for periodic inviscid flow.
  auto diagnostics = [&](double current_time) {
    double mass = 0.0;
    double momentum = 0.0;
    double kinetic = 0.0;
    double internal = 0.0;
    double max_mach = 0.0;
    Kokkos::parallel_reduce(
        "diagnostics", interior_policy,
        KOKKOS_LAMBDA(int k, int j, int i, double& m, double& mom,
                      double& kin, double& internal_energy, double& mach) {
          const saturn::PrimitiveState state =
              saturn::compressible::load_primitive(q, i, j, k, gas);
          const double speed2 = state.u * state.u + state.v * state.v +
                                state.w * state.w;
          const double speed = Kokkos::sqrt(speed2);
          const double c =
              Kokkos::sqrt(gas.gamma * state.p / state.rho);
          m += state.rho;
          mom += state.rho * speed;
          kin += 0.5 * state.rho * speed2;
          internal_energy += state.p / (gas.gamma - 1.0);
          const double local_mach = speed / c;
          mach = local_mach > mach ? local_mach : mach;
        },
        Kokkos::Sum<double>{mass}, Kokkos::Sum<double>{momentum},
        Kokkos::Sum<double>{kinetic}, Kokkos::Sum<double>{internal},
        Kokkos::Max<double>{max_mach});
    mass = saturn::mpi_sum(mass) * cell_volume;
    momentum = saturn::mpi_sum(momentum) * cell_volume;
    kinetic = saturn::mpi_sum(kinetic) * cell_volume;
    internal = saturn::mpi_sum(internal) * cell_volume;
    max_mach = saturn::mpi_max(max_mach);
    const double total = kinetic + internal;
    if (root) {
      std::printf(
          "t=%.4f step=%d mass=%.8e |p|=%.6e K=%.8e e=%.8e E=%.10e "
          "Mach_max=%.3e\n",
          current_time, step, mass, momentum, kinetic, internal, total,
          max_mach);
      if (history.is_open()) {
        history << std::setprecision(12) << current_time << "," << step << ","
                << mass << "," << momentum << "," << kinetic << ","
                << internal << "," << total << "," << max_mach << "\n";
        history.flush();
      }
    }
    return std::array<double, 6>{mass,     momentum, kinetic,
                                 internal, total,    max_mach};
  };

  auto write_snapshot = [&](const std::string& base, int snapshot_step) {
    field_type primitive(box, 5, 0, "primitive");
    Kokkos::parallel_for(
        "primitive_output", interior_policy,
        KOKKOS_LAMBDA(int k, int j, int i) {
          const saturn::PrimitiveState state =
              saturn::compressible::load_primitive(q, i, j, k, gas);
          primitive(0, i, j, k) = state.rho;
          primitive(1, i, j, k) = state.u;
          primitive(2, i, j, k) = state.v;
          primitive(3, i, j, k) = state.w;
          primitive(4, i, j, k) = state.p;
        });
    char suffix[64];
    std::snprintf(suffix, sizeof(suffix), "_%06d", snapshot_step);
    const std::array<double, 3> origin{{0.0, 0.0, 0.0}};
    const std::array<double, 3> spacing{
        {grid.h[0], grid.h[1], grid.h[2]}};
    saturn::write_vti_fields(
        base + suffix, primitive, {"rho", "u", "v", "w", "p"}, origin,
        spacing, decomposition.global, time);
  };

  if (!params.output.empty()) {
    write_snapshot(params.output, step);
  }
  if (params.checkpoint_every > 0 && !params.checkpoint.empty()) {
    saturn::write_checkpoint(params.checkpoint, q, step, time, decomposition);
  }
  int last_diagnostics_step = step;
  auto last_diagnostics = diagnostics(time);
  const double initial_total_energy = last_diagnostics[4];
  const bool expect_energy_conservation = all_periodic && params.mu == 0.0;

  Kokkos::Timer timer;
  double last_dt = 0.0;
  while (time < params.tend - 1.0e-14) {
    const double max_speed = saturn::compressible::max_wave_speed(q, gas);
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
      saturn::advance_rk4(q, stage, rhs, k1, k2, k3, dt, compute_rhs);
    } else {
      saturn::advance_ssprk3(q, stage, next, rhs, dt, compute_rhs);
    }

    time += dt;
    ++step;

    if (!params.output.empty() && params.plot_every > 0 &&
        step % params.plot_every == 0) {
      write_snapshot(params.output, step);
      if (step != last_diagnostics_step) {
        last_diagnostics = diagnostics(time);
        last_diagnostics_step = step;
      }
    }
    if (params.diag_every > 0 && step % params.diag_every == 0 &&
        step != last_diagnostics_step) {
      last_diagnostics = diagnostics(time);
      last_diagnostics_step = step;
    }
    if (!params.checkpoint.empty() && params.checkpoint_every > 0 &&
        step % params.checkpoint_every == 0) {
      saturn::write_checkpoint(params.checkpoint, q, step, time,
                               decomposition);
    }
  }
  Kokkos::fence();

  if (!params.output.empty()) {
    write_snapshot(params.output, step);
  }
  if (!params.checkpoint.empty() && params.checkpoint_every > 0) {
    saturn::write_checkpoint(params.checkpoint, q, step, time, decomposition);
  }
  if (step != last_diagnostics_step) {
    last_diagnostics = diagnostics(time);
  }
  const auto final_diagnostics = last_diagnostics;
  const double seconds = saturn::mpi_max(timer.seconds());

  bool valid = true;
  double local_invalid = 0.0;
  Kokkos::parallel_reduce(
      "validity", interior_policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& bad) {
        const saturn::PrimitiveState state =
            saturn::compressible::load_primitive(q, i, j, k, gas);
        const bool ok = state.rho > 0.0 && state.p > 0.0 &&
                        state.rho == state.rho && state.p == state.p &&
                        state.rho < 1.0e200 && state.p < 1.0e200;
        if (!ok) {
          bad += 1.0;
        }
      },
      Kokkos::Sum<double>{local_invalid});
  local_invalid = saturn::mpi_sum(local_invalid);
  valid = local_invalid == 0.0;
  const double energy_drift =
      initial_total_energy > 0.0
          ? std::abs(final_diagnostics[4] - initial_total_energy) /
                initial_total_energy
          : 0.0;
  const bool conservation_ok =
      !expect_energy_conservation || energy_drift <= params.energy_tol;

  if (root) {
    std::printf("finished: steps=%d time=%.6f dt_last=%.3e wall=%.3fs\n",
                step, time, last_dt, seconds);
    std::printf(
        "valid=%s K=%.8e e=%.8e E=%.10e Mach_max=%.3e dE/E=%.2e\n",
        valid ? "yes" : "no", final_diagnostics[2], final_diagnostics[3],
        final_diagnostics[4], final_diagnostics[5], energy_drift);
  }
  if (params.check && (!valid || !conservation_ok)) {
    if (root && !conservation_ok) {
      std::fprintf(stderr,
                   "energy conservation check failed: |dE/E| = %.2e > %.2e\n",
                   energy_drift, params.energy_tol);
    }
    return 1;
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
