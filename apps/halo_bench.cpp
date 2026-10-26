// Halo-exchange bandwidth benchmark.
//
// Times FaceExchange on a periodic cube. Reports per-exchange latency,
// aggregate message bandwidth, and cost per cell. Run with 1, 2, 4, 8, ...
// ranks to measure strong/weak scaling.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/mpi.hpp"
#include "core/parallel.hpp"
#include "core/version.hpp"
#include "fields/patch_field.hpp"
#include "halo/face_exchange.hpp"
#include "mesh/decomposition.hpp"
#include "mesh/uniform_grid.hpp"

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
  int n = 128;
  int ng = 2;
  int iters = 20;
  int warmup = 3;
  std::string space = "default";
  std::string json;
  bool verbose = false;
};

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --n N                  global cells per direction (default 128)\n"
      << "  --ng N                 ghost layers (default 2)\n"
      << "  --iters N              timed exchanges (default 20)\n"
      << "  --warmup N             warmup exchanges (default 3)\n"
      << "  --space NAME           serial|openmp|cuda|hip|sycl|default\n"
      << "  --json PATH            write results as JSON\n"
      << "  --verbose              print full Kokkos configuration\n"
      << "  --help                 show this help\n";
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
      p.n = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--ng") {
      p.ng = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--iters") {
      p.iters = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--warmup") {
      p.warmup = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--space") {
      p.space = require_value(i, argc, argv, a);
    } else if (a == "--json") {
      p.json = require_value(i, argc, argv, a);
    } else if (a == "--verbose") {
      p.verbose = true;
    } else if (a.rfind("--kokkos-", 0) == 0) {
      // Handled by Kokkos::initialize().
    } else {
      std::fprintf(stderr, "warning: ignoring unknown argument '%s'\n",
                   a.c_str());
    }
  }
  if (p.n < 4 || p.ng < 1 || p.iters < 1) {
    std::fprintf(stderr, "n >= 4, ng >= 1, iters >= 1 are required\n");
    std::exit(2);
  }
  return p;
}

template <class ExecSpace>
int run_impl(const Params& p) {
  using field_type = saturn::PatchField<ExecSpace>;

  const bool periodic[3] = {true, true, true};
  const auto decomposition = saturn::Decomposition::create(
      saturn::Box::from_extents(p.n, p.n, p.n), periodic);
  const auto grid = saturn::UniformGrid::from_cells_local(
      p.n, p.n, p.n, 1.0, 1.0, 1.0, p.ng, decomposition.local);
  const saturn::Box box = grid.local_box();

  field_type q(box, 1, p.ng, "q");
  auto hq = Kokkos::create_mirror_view(q.data());
  for (int k = box.lo[2]; k <= box.hi[2]; ++k) {
    for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        hq(0, k - box.lo[2] + p.ng, j - box.lo[1] + p.ng,
           i - box.lo[0] + p.ng) =
            static_cast<double>(i + 10 * j + 100 * k);
      }
    }
  }
  Kokkos::deep_copy(q.data(), hq);

  saturn::FaceExchange<ExecSpace> exchange(decomposition, 1, p.ng);
  for (int w = 0; w < p.warmup; ++w) {
    exchange.exchange(q);
  }
  ExecSpace().fence();

  Kokkos::Timer timer;
  for (int it = 0; it < p.iters; ++it) {
    exchange.exchange(q);
  }
  ExecSpace().fence();
  const double seconds = saturn::mpi_max(timer.seconds());

  // Aggregate bytes across all ranks: two messages (send+recv) per
  // non-self, non-null face.
  long long local_bytes = 0;
  for (int d = 0; d < 3; ++d) {
    const int n1 = (d == 0) ? box.length(1) : box.length(0);
    const int n2 = (d == 0) ? box.length(2)
                            : ((d == 1) ? box.length(2) : box.length(1));
    const long long face_bytes =
        static_cast<long long>(p.ng) * n1 * n2 * 8;
    for (int side_sign = -1; side_sign <= 1; side_sign += 2) {
      const int partner = decomposition.partner(d, side_sign);
#ifdef SATURN_ENABLE_MPI
      if (partner == MPI_PROC_NULL || partner == decomposition.rank) {
        continue;
      }
      local_bytes += 2 * face_bytes;
#else
      (void)partner;
#endif
    }
  }

  long long total_bytes = local_bytes;
#ifdef SATURN_ENABLE_MPI
  MPI_Allreduce(&local_bytes, &total_bytes, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  const double cells = static_cast<double>(p.n) * p.n * p.n;
  const double us_per_exchange = seconds / p.iters * 1.0e6;
  const double ns_per_cell = seconds / p.iters / cells * 1.0e9;
  const double gbps = total_bytes * p.iters / seconds / 1.0e9;

  if (decomposition.rank == 0) {
    std::cout << "\n" << saturn::kProjectName << " " << saturn::kVersion
              << " halo benchmark\n"
              << "Kokkos:      " << ExecSpace::name() << " (concurrency "
              << ExecSpace{}.concurrency() << ")\n";
    if (p.verbose) {
      Kokkos::print_configuration(std::cout, true);
    }
    std::cout << "ranks:       " << decomposition.size << "  dims="
              << decomposition.dims[0] << "x" << decomposition.dims[1] << "x"
              << decomposition.dims[2] << "\n"
              << "grid:        " << p.n << "^3, local " << box.length(0)
              << "x" << box.length(1) << "x" << box.length(2) << "\n"
              << "ghosts:      " << p.ng << "\n"
              << "time:        " << std::fixed << std::setprecision(3)
              << us_per_exchange << " us/exchange\n"
              << "bandwidth:   " << std::setprecision(3) << gbps
              << " GB/s aggregate\n"
              << "cost:        " << ns_per_cell << " ns/cell/exchange\n";
  }

  if (decomposition.rank == 0 && !p.json.empty()) {
    const std::filesystem::path path(p.json);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(p.json);
    if (!out) {
      std::fprintf(stderr, "could not open %s for writing\n", p.json.c_str());
      return 2;
    }
    out << std::setprecision(10)
        << "{\n"
        << "  \"code\": \"" << saturn::kProjectName << "\",\n"
        << "  \"version\": \"" << saturn::kVersion << "\",\n"
        << "  \"backend\": \"" << ExecSpace::name() << "\",\n"
        << "  \"space\": \"" << p.space << "\",\n"
        << "  \"ranks\": " << decomposition.size << ",\n"
        << "  \"dims\": [" << decomposition.dims[0] << ", "
        << decomposition.dims[1] << ", " << decomposition.dims[2] << "],\n"
        << "  \"n\": " << p.n << ",\n"
        << "  \"ng\": " << p.ng << ",\n"
        << "  \"iters\": " << p.iters << ",\n"
        << "  \"local_cells\": " << box.num_cells() << ",\n"
        << "  \"seconds\": " << seconds << ",\n"
        << "  \"us_per_exchange\": " << us_per_exchange << ",\n"
        << "  \"gbps_aggregate\": " << gbps << ",\n"
        << "  \"ns_per_cell_exchange\": " << ns_per_cell << "\n"
        << "}\n";
    std::cout << "json:        " << p.json << "\n";
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
        return run_impl<space_type>(params);
      });
  Kokkos::finalize();

#ifdef SATURN_ENABLE_MPI
  MPI_Finalize();
#endif
  return status;
}
