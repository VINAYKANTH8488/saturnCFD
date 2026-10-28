// STREAM-style memory-bandwidth microbenchmark (copy and triad).
//
// This separates "the machine cannot stream faster" from "the stencil kernel is
// inefficient". Compare the stencil's effective GB/s against these numbers.

#include <Kokkos_Core.hpp>

#include "core/execution_space.hpp"
#include "core/version.hpp"

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

enum class Kernel { Copy, Triad };

struct Params {
  int n = 32 * 1024 * 1024;
  int iters = 50;
  int warmup = 3;
  Kernel kernel = Kernel::Copy;
  std::string space = "default";
  std::string json;
  bool verbose = false;
};

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --n N                  number of doubles (default 32M)\n"
            << "  --kernel NAME          copy | triad (default copy)\n"
            << "  --iters N              timed iterations (default 50)\n"
            << "  --warmup N             warmup iterations (default 3)\n"
            << "  --space NAME           serial|openmp|cuda|hip|sycl|default\n"
            << "  --json PATH            write results as JSON\n"
            << "  --verbose              print the full Kokkos configuration\n"
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
    } else if (a == "--iters") {
      p.iters = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--warmup") {
      p.warmup = parse_int(a, require_value(i, argc, argv, a));
    } else if (a == "--kernel") {
      const std::string value = require_value(i, argc, argv, a);
      if (value == "copy") {
        p.kernel = Kernel::Copy;
      } else if (value == "triad") {
        p.kernel = Kernel::Triad;
      } else {
        std::fprintf(stderr, "unknown kernel '%s' (use copy|triad)\n",
                     value.c_str());
        std::exit(2);
      }
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

  if (p.n < 1) {
    std::fprintf(stderr, "--n must be >= 1\n");
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
  using view_type = Kokkos::View<double*, Kokkos::LayoutRight,
                                 typename ExecSpace::memory_space>;

  const bool triad = (p.kernel == Kernel::Triad);
  view_type a("a", p.n);
  view_type b("b", p.n);
  view_type c("c", p.n);

  Kokkos::parallel_for(
      "init", p.n, KOKKOS_LAMBDA(int i) {
        a(i) = 1.0 + 1.0e-6 * static_cast<double>(i);
        b(i) = 2.0 - 1.0e-6 * static_cast<double>(i);
        c(i) = 0.0;
      });
  ExecSpace().fence();

  auto launch = [&]() {
    if (triad) {
      Kokkos::parallel_for(
          "triad", p.n, KOKKOS_LAMBDA(int i) { c(i) = a(i) + 0.5 * b(i); });
    } else {
      Kokkos::parallel_for("copy", p.n, KOKKOS_LAMBDA(int i) { c(i) = a(i); });
    }
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
      "checksum", p.n, KOKKOS_LAMBDA(int i, double& acc) { acc += c(i); },
      checksum);
  ExecSpace().fence();

  const double elements = static_cast<double>(p.n) * p.iters;
  const double bytes_per_element = triad ? 24.0 : 16.0;  // 3 or 2 arrays
  const double gbps = bytes_per_element * elements / seconds / 1.0e9;
  const double ns_per_element_iter = seconds / elements * 1.0e9;

  std::cout << "\n"
            << saturn::kProjectName << " " << saturn::kVersion << "\n"
            << "Kokkos:      " << ExecSpace::name() << " (concurrency "
            << ExecSpace{}.concurrency() << ")\n";
  if (p.verbose) {
    Kokkos::print_configuration(std::cout, true);
  }
  std::cout << "kernel:      " << (triad ? "triad" : "copy") << "\n"
            << "elements:    " << p.n << " doubles x " << p.iters
            << " iterations\n"
            << "time:        " << std::fixed << std::setprecision(6) << seconds
            << " s  (" << (seconds / p.iters * 1.0e3) << " ms/iter)\n"
            << "bandwidth:   " << std::setprecision(3) << gbps << " GB/s\n"
            << "throughput:  " << ns_per_element_iter << " ns/element/iter\n"
            << "checksum:    " << std::setprecision(10) << checksum << "\n";

  if (!p.json.empty()) {
    const std::filesystem::path path(p.json);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(p.json);
    if (!out) {
      std::fprintf(stderr, "could not open %s for writing\n", p.json.c_str());
      return 2;
    }
    out << std::setprecision(10) << "{\n"
        << "  \"code\": \"" << saturn::kProjectName << "\",\n"
        << "  \"version\": \"" << saturn::kVersion << "\",\n"
        << "  \"backend\": \"" << ExecSpace::name() << "\",\n"
        << "  \"concurrency\": " << ExecSpace{}.concurrency() << ",\n"
        << "  \"space\": \"" << p.space << "\",\n"
        << "  \"kernel\": \"" << (triad ? "triad" : "copy") << "\",\n"
        << "  \"n\": " << p.n << ",\n"
        << "  \"iters\": " << p.iters << ",\n"
        << "  \"warmup\": " << p.warmup << ",\n"
        << "  \"bytes_per_element\": " << bytes_per_element << ",\n"
        << "  \"seconds\": " << seconds << ",\n"
        << "  \"ms_per_iter\": " << seconds / p.iters * 1.0e3 << ",\n"
        << "  \"gbps\": " << gbps << ",\n"
        << "  \"ns_per_element_iter\": " << ns_per_element_iter << ",\n"
        << "  \"checksum\": " << checksum << "\n"
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
