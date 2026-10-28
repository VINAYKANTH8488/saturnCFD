// saturnCFD 7-point 3D stencil microbenchmark.
//
// Purpose: establish the memory-bandwidth roofline and the per-cell cost of the
// stencil kernel that dominates explicit FVM stages. This is a calibration
// tool, not a solver.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"

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

#ifndef SATURN_DEFAULT_SPACE
#define SATURN_DEFAULT_SPACE "default"
#endif

namespace {

struct Params {
  int nx = 256;
  int ny = 256;
  int nz = 256;
  int iters = 50;
  int warmup = 3;
  int tile = 0;
  std::string space = "default";
  std::string json;
  bool verify = false;
  bool verbose = false;
};

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --nx N --ny N --nz N   grid size (default 256^3)\n"
            << "  --iters N              timed iterations (default 50)\n"
            << "  --warmup N             warmup iterations (default 3)\n"
            << "  --tile N               inner MDRange tile size (0 = backend "
               "default)\n"
            << "  --space NAME           serial|openmp|cuda|hip|sycl|default\n"
            << "  --verify               compare against a host reference "
               "(small grids)\n"
            << "  --json PATH            write results as JSON\n"
            << "  --verbose              print the full Kokkos configuration\n"
            << "  --help                 show this help\n"
            << "\n"
            << "Kokkos options (for example --kokkos-num-threads=4) are passed "
               "through.\n";
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
    } else if (a == "--nx") {
      p.nx = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--ny") {
      p.ny = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--nz") {
      p.nz = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--iters") {
      p.iters = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--warmup") {
      p.warmup = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--tile") {
      p.tile = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--space") {
      p.space = require_value(i, argc, argv, a);
    } else if (a == "--json") {
      p.json = require_value(i, argc, argv, a);
    } else if (a == "--verify") {
      p.verify = true;
    } else if (a == "--verbose") {
      p.verbose = true;
    } else if (a.rfind("--kokkos-", 0) == 0) {
      // Handled by Kokkos::initialize().
    } else {
      std::fprintf(stderr, "warning: ignoring unknown argument '%s'\n",
                   a.c_str());
    }
  }

  if (p.nx < 3 || p.ny < 3 || p.nz < 3) {
    std::fprintf(stderr, "grid dimensions must be >= 3\n");
    std::exit(2);
  }
  if (p.iters < 1) {
    std::fprintf(stderr, "--iters must be >= 1\n");
    std::exit(2);
  }
  return p;
}

template <class ExecSpace>
int run_impl(const Params& p) {
  using view_type = Kokkos::View<double***, Kokkos::LayoutRight,
                                 typename ExecSpace::memory_space>;
  using policy_type = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;

  const double interior =
      static_cast<double>(p.nx - 2) * (p.ny - 2) * (p.nz - 2);

  view_type in("in", p.nz, p.ny, p.nx);
  view_type out("out", p.nz, p.ny, p.nx);

  Kokkos::parallel_for(
      "init", policy_type({0, 0, 0}, {p.nz, p.ny, p.nx}),
      KOKKOS_LAMBDA(int k, int j, int i) {
        in(k, j, i) = 1.0 + 1.0e-6 * static_cast<double>(i + 2 * j + 3 * k);
      });
  ExecSpace().fence();

  constexpr double c0 = 1.0;
  constexpr double c1 = 0.1;
  const policy_type interior_policy = [&]() {
    if (p.tile > 0) {
      return policy_type({1, 1, 1}, {p.nz - 1, p.ny - 1, p.nx - 1},
                         {1, 1, p.tile});
    }
    if constexpr (saturn::is_host_execution_space<ExecSpace>) {
      return policy_type({1, 1, 1}, {p.nz - 1, p.ny - 1, p.nx - 1});
    } else {
      return policy_type({1, 1, 1}, {p.nz - 1, p.ny - 1, p.nx - 1},
                         {1, 1, saturn::kInnerTile});
    }
  }();

  auto launch = [&]() {
    Kokkos::parallel_for(
        "stencil7", interior_policy, KOKKOS_LAMBDA(int k, int j, int i) {
          out(k, j, i) =
              c0 * in(k, j, i) +
              c1 * (in(k - 1, j, i) + in(k + 1, j, i) + in(k, j - 1, i) +
                    in(k, j + 1, i) + in(k, j, i - 1) + in(k, j, i + 1));
        });
  };

  for (int w = 0; w < p.warmup; ++w) {
    launch();
  }
  ExecSpace().fence();

  Kokkos::Timer timer;
  for (int it = 0; it < p.iters; ++it) {
    launch();
  }
  ExecSpace().fence();
  const double seconds = timer.seconds();

  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "checksum", interior_policy,
      KOKKOS_LAMBDA(int k, int j, int i, double& acc) { acc += out(k, j, i); },
      checksum);
  ExecSpace().fence();

  const double updates = interior * static_cast<double>(p.iters);
  const double gcells_per_s = updates / seconds / 1.0e9;
  const double ns_per_cell_iter = seconds / updates * 1.0e9;

  // Effective traffic: one read of the input array and one write of the output
  // array per cell. The naive count assumes every neighbor load misses cache
  // (8 accesses per cell) and is reported for comparison only.
  const double bytes_effective = 2.0 * 8.0 * updates;
  const double bytes_naive = 8.0 * 8.0 * updates;
  const double gbps_effective = bytes_effective / seconds / 1.0e9;
  const double gbps_naive = bytes_naive / seconds / 1.0e9;

  double max_error = 0.0;
  if (p.verify &&
      static_cast<long long>(p.nx) * p.ny * p.nz <= 64LL * 64LL * 64LL) {
    auto h_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in);
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    for (int k = 1; k < p.nz - 1; ++k) {
      for (int j = 1; j < p.ny - 1; ++j) {
        for (int i = 1; i < p.nx - 1; ++i) {
          const double ref =
              c0 * h_in(k, j, i) +
              c1 * (h_in(k - 1, j, i) + h_in(k + 1, j, i) + h_in(k, j - 1, i) +
                    h_in(k, j + 1, i) + h_in(k, j, i - 1) + h_in(k, j, i + 1));
          max_error = std::max(max_error, std::abs(h_out(k, j, i) - ref));
        }
      }
    }
  }

  std::cout << "\n"
            << saturn::kProjectName << " " << saturn::kVersion << "\n"
            << "Kokkos:      " << ExecSpace::name() << " (concurrency "
            << ExecSpace{}.concurrency() << ")\n";
  if (p.verbose) {
    Kokkos::print_configuration(std::cout, true);
  }
  std::cout << "grid:        " << p.nx << " x " << p.ny << " x " << p.nz << " ("
            << (static_cast<double>(p.nx) * p.ny * p.nz / 1.0e6)
            << " M cells)\n"
            << "backend:     " << ExecSpace::name() << "\n"
            << "iterations:  " << p.iters << " (warmup " << p.warmup << ")\n"
            << "time:        " << std::fixed << std::setprecision(6) << seconds
            << " s  (" << (seconds / p.iters * 1.0e3) << " ms/iter)\n"
            << "throughput:  " << std::setprecision(3) << gcells_per_s
            << " Gcell-updates/s, " << ns_per_cell_iter << " ns/cell/iter\n"
            << "bandwidth:   " << gbps_effective
            << " GB/s effective (read+write), " << gbps_naive
            << " GB/s naive (8 accesses)\n"
            << "checksum:    " << std::setprecision(10) << checksum << "\n";
  if (p.verify) {
    if (static_cast<long long>(p.nx) * p.ny * p.nz <= 64LL * 64LL * 64LL) {
      std::cout << "verify:      max_abs_error = " << std::scientific
                << max_error << std::defaultfloat << "\n";
    } else {
      std::cout << "verify:      skipped (grid larger than 64^3)\n";
    }
  }

  if (!p.json.empty()) {
    const std::filesystem::path path(p.json);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out_file(p.json);
    if (!out_file) {
      std::fprintf(stderr, "could not open %s for writing\n", p.json.c_str());
      return 2;
    }
    out_file << std::setprecision(10) << "{\n"
             << "  \"code\": \"" << saturn::kProjectName << "\",\n"
             << "  \"version\": \"" << saturn::kVersion << "\",\n"
             << "  \"backend\": \"" << ExecSpace::name() << "\",\n"
             << "  \"concurrency\": " << ExecSpace{}.concurrency() << ",\n"
             << "  \"space\": \"" << p.space << "\",\n"
             << "  \"nx\": " << p.nx << ",\n"
             << "  \"ny\": " << p.ny << ",\n"
             << "  \"nz\": " << p.nz << ",\n"
             << "  \"iters\": " << p.iters << ",\n"
             << "  \"warmup\": " << p.warmup << ",\n"
             << "  \"tile\": " << p.tile << ",\n"
             << "  \"seconds\": " << seconds << ",\n"
             << "  \"ms_per_iter\": " << seconds / p.iters * 1.0e3 << ",\n"
             << "  \"ns_per_cell_iter\": " << ns_per_cell_iter << ",\n"
             << "  \"gcell_updates_per_s\": " << gcells_per_s << ",\n"
             << "  \"gbps_effective\": " << gbps_effective << ",\n"
             << "  \"gbps_naive\": " << gbps_naive << ",\n"
             << "  \"checksum\": " << checksum << ",\n"
             << "  \"max_abs_error\": " << max_error << "\n"
             << "}\n";
    std::cout << "json:        " << p.json << "\n";
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Params params = parse(argc, argv);

  Kokkos::initialize(argc, argv);
  const int status = saturn::dispatch_execution_space(
      params.space, SATURN_DEFAULT_SPACE, [&](auto tag) {
        using space_type = decltype(tag);
        return run_impl<space_type>(params);
      });
  Kokkos::finalize();
  return status;
}
