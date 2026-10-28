#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    Kokkos::View<double*> values("values", 10);
    Kokkos::parallel_for(
        "fill", 10,
        KOKKOS_LAMBDA(int i) { values(i) = static_cast<double>(i); });

    double sum = 0.0;
    Kokkos::parallel_reduce(
        "sum", 10, KOKKOS_LAMBDA(int i, double& acc) { acc += values(i); },
        sum);

    if (std::abs(sum - 45.0) > 1.0e-12) {
      std::fprintf(stderr, "smoke test failed: sum = %.17g\n", sum);
      status = 1;
    } else {
      std::printf("saturnCFD smoke test passed (sum = %.1f)\n", sum);
    }
  }
  Kokkos::finalize();
  return status;
}
