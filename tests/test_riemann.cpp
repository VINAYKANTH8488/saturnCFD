// Unit tests for the ideal-gas Riemann solvers:
// physical-flux consistency, contact preservation, symmetry, shock handling,
// and low-Mach behaviour.

#include <Kokkos_Core.hpp>

#include "numerics/riemann.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

using saturn::Flux;
using saturn::PrimitiveState;
using saturn::RiemannSolver;
using saturn::riemann::flux;

constexpr double kGamma = 1.4;
constexpr RiemannSolver kAllSolvers[] = {
    RiemannSolver::Llf, RiemannSolver::Hllc, RiemannSolver::HllcLowMach,
    RiemannSolver::AusmPlusUp};

int g_failures = 0;

const char* solver_name(RiemannSolver solver) {
  switch (solver) {
    case RiemannSolver::Llf:
      return "llf";
    case RiemannSolver::Hllc:
      return "hllc";
    case RiemannSolver::HllcLowMach:
      return "hllc_lm";
    case RiemannSolver::AusmPlusUp:
      return "ausm_plus_up";
  }
  return "unknown";
}

bool finite(double value) {
  return value == value && std::fabs(value) < 1.0e300;
}

double flux_scale(const Flux& value) {
  return std::max(
      1.0, std::max(std::fabs(value.rho),
                    std::max(std::fabs(value.mom_n), std::fabs(value.energy))));
}

void check_flux(const char* what, const char* solver, const Flux& actual,
                const Flux& expected, double tolerance) {
  const double scale = std::max(flux_scale(expected), flux_scale(actual));
  const double error = std::max(
      std::fabs(actual.rho - expected.rho),
      std::max(std::fabs(actual.mom_n - expected.mom_n),
               std::max(std::fabs(actual.mom_t1 - expected.mom_t1),
                        std::max(std::fabs(actual.mom_t2 - expected.mom_t2),
                                 std::fabs(actual.energy - expected.energy)))));
  if (!(error <= tolerance * scale)) {
    std::printf(
        "FAIL %s [%s]: actual [%.12g %.12g %.12g %.12g %.12g] "
        "expected [%.12g %.12g %.12g %.12g %.12g] error %.3e\n",
        what, solver, actual.rho, actual.mom_n, actual.mom_t1, actual.mom_t2,
        actual.energy, expected.rho, expected.mom_n, expected.mom_t1,
        expected.mom_t2, expected.energy, error);
    ++g_failures;
  }
}

void check_value(const char* what, const char* solver, double actual,
                 double expected, double tolerance) {
  const double scale = std::max(1.0, std::fabs(expected));
  if (!(std::fabs(actual - expected) <= tolerance * scale)) {
    std::printf("FAIL %s [%s]: actual %.17g expected %.17g\n", what, solver,
                actual, expected);
    ++g_failures;
  }
}

PrimitiveState mirror(const PrimitiveState& state) {
  return PrimitiveState{state.rho, -state.u, state.v, state.w, state.p};
}

double distance(const Flux& left, const Flux& right) {
  return std::fabs(left.rho - right.rho) + std::fabs(left.mom_n - right.mom_n) +
         std::fabs(left.mom_t1 - right.mom_t1) +
         std::fabs(left.mom_t2 - right.mom_t2) +
         std::fabs(left.energy - right.energy);
}

void test_uniform_consistency() {
  const PrimitiveState states[] = {
      {1.0, 0.2, 0.1, -0.3, 1.0},
      {0.7, -0.4, 0.2, 0.0, 0.5},
      {2.0, 0.0, 0.0, 0.0, 2.0},
      {0.2, 1.3, -0.2, 0.4, 0.1},
  };
  for (const auto& state : states) {
    const Flux expected = saturn::riemann::physical_flux(state, kGamma);
    for (const auto solver : kAllSolvers) {
      check_flux("uniform consistency", solver_name(solver),
                 flux(solver, state, state, kGamma), expected, 1.0e-12);
    }
  }
}

void test_stationary_contact() {
  const PrimitiveState left{1.0, 0.0, 0.0, 0.0, 1.0};
  const PrimitiveState right{3.0, 0.0, 0.0, 0.0, 1.0};
  const Flux expected{0.0, 1.0, 0.0, 0.0, 0.0};
  for (const auto solver : kAllSolvers) {
    if (solver == RiemannSolver::Llf) {
      continue;  // LLF diffuses a stationary contact by construction.
    }
    check_flux("stationary contact", solver_name(solver),
               flux(solver, left, right, kGamma), expected, 1.0e-12);
  }
}

void test_moving_contact() {
  const PrimitiveState left{1.0, 0.5, 0.0, 0.0, 1.0};
  const PrimitiveState right{3.0, 0.5, 0.0, 0.0, 1.0};
  // Upwind density is 1. Energy per volume on the left:
  // p/(gamma-1) + 0.5 rho u^2 = 2.5 + 0.125 = 2.625,
  // enthalpy (E+p)/rho = 3.625, energy flux = 0.5 * 3.625.
  const Flux expected{0.5, 1.25, 0.0, 0.0, 1.8125};
  for (const auto solver : kAllSolvers) {
    if (solver == RiemannSolver::Llf) {
      continue;  // LLF adds diffusion even for a moving contact.
    }
    check_flux("moving contact", solver_name(solver),
               flux(solver, left, right, kGamma), expected, 1.0e-12);
  }
}

void test_symmetry() {
  const PrimitiveState left{1.0, 0.3, 0.2, -0.1, 0.9};
  const PrimitiveState right{0.4, -0.2, 0.5, 0.3, 0.2};
  for (const auto solver : kAllSolvers) {
    const Flux value = flux(solver, left, right, kGamma);
    const Flux reflected = flux(solver, mirror(right), mirror(left), kGamma);
    const Flux expected{-value.rho, value.mom_n, -value.mom_t1, -value.mom_t2,
                        -value.energy};
    check_flux("mirror symmetry", solver_name(solver), reflected, expected,
               1.0e-12);
  }

  // The low-Mach blend must also be mirror-symmetric.
  const PrimitiveState slow_left{1.0, 1.0e-3, 0.2, -0.1, 1.0};
  const PrimitiveState slow_right{0.8, -5.0e-4, -0.2, 0.1, 0.9999};
  const Flux value =
      flux(RiemannSolver::HllcLowMach, slow_left, slow_right, kGamma);
  const Flux reflected = flux(RiemannSolver::HllcLowMach, mirror(slow_right),
                              mirror(slow_left), kGamma);
  const Flux expected{-value.rho, value.mom_n, -value.mom_t1, -value.mom_t2,
                      -value.energy};
  check_flux("low-Mach mirror symmetry", "hllc_lm", reflected, expected,
             1.0e-12);
}

void test_sod_fluxes() {
  const PrimitiveState left{1.0, 0.0, 0.0, 0.0, 1.0};
  const PrimitiveState right{0.125, 0.0, 0.0, 0.0, 0.1};
  for (const auto solver : kAllSolvers) {
    const Flux value = flux(solver, left, right, kGamma);
    if (!(finite(value.rho) && finite(value.mom_n) && finite(value.mom_t1) &&
          finite(value.mom_t2) && finite(value.energy))) {
      std::printf("FAIL Sod flux [%s]: non-finite flux\n", solver_name(solver));
      ++g_failures;
    }
    if (!(value.rho > 0.0 && value.mom_n > 0.0)) {
      std::printf(
          "FAIL Sod flux [%s]: expected rightward mass/momentum "
          "flux, got rho=%.6g mom=%.6g\n",
          solver_name(solver), value.rho, value.mom_n);
      ++g_failures;
    }
  }

  // The pressure-jump sensor must leave HLLC-LM identical to HLLC on a shock.
  const Flux hllc = flux(RiemannSolver::Hllc, left, right, kGamma);
  const Flux low_mach = flux(RiemannSolver::HllcLowMach, left, right, kGamma);
  check_flux("shock sensor", "hllc_lm", low_mach, hllc, 1.0e-14);

  // AUSM+-up at a strong pressure jump uses a larger freestream Mach floor.
  const Flux ausm = saturn::riemann::ausm_plus_up(left, right, kGamma, 1.0);
  if (!(finite(ausm.rho) && finite(ausm.mom_n) && finite(ausm.energy))) {
    std::printf("FAIL Sod flux [ausm_plus_up] with Mach floor 1\n");
    ++g_failures;
  }
}

void test_low_mach_dissipation() {
  // Low-Mach pressure perturbation: HLLC-LM should be closer to the central
  // (non-dissipative) flux than plain HLLC.
  const PrimitiveState left{1.0, 1.0e-2, 0.0, 0.0, 1.0};
  const PrimitiveState right{0.8, 1.0e-2, 0.0, 0.0, 1.0001};
  const Flux central = saturn::riemann::average_flux(
      saturn::riemann::physical_flux(left, kGamma),
      saturn::riemann::physical_flux(right, kGamma));
  const Flux hllc = flux(RiemannSolver::Hllc, left, right, kGamma);
  const Flux low_mach = flux(RiemannSolver::HllcLowMach, left, right, kGamma);
  const double hllc_error = distance(hllc, central);
  const double low_mach_error = distance(low_mach, central);
  std::printf("  low-Mach distance to central: HLLC %.3e, HLLC-LM %.3e\n",
              hllc_error, low_mach_error);
  if (!(low_mach_error < hllc_error)) {
    std::printf(
        "FAIL low-Mach dissipation: HLLC-LM did not reduce the "
        "central-flux distance\n");
    ++g_failures;
  }

  // AUSM+-up in the same regime must stay finite and produce a small mass
  // flux instead of an O(1) artifact.
  const Flux ausm = saturn::riemann::ausm_plus_up(left, right, kGamma, 0.01);
  check_value("AUSM+-up low-Mach mass flux", "ausm_plus_up", ausm.rho, 0.0,
              0.05);
  check_value("AUSM+-up low-Mach pressure flux", "ausm_plus_up", ausm.mom_n,
              1.0, 0.05);
}

void test_device_kernel() {
  constexpr int kSolverCount = 4;
  Kokkos::View<double*> values("riemann_values", 10);
  Kokkos::View<double*> output("riemann_output", 5 * kSolverCount);
  auto host_values = Kokkos::create_mirror_view(values);
  // Left state.
  host_values(0) = 1.0;
  host_values(1) = 0.3;
  host_values(2) = 0.2;
  host_values(3) = -0.1;
  host_values(4) = 0.9;
  // Right state.
  host_values(5) = 0.4;
  host_values(6) = -0.2;
  host_values(7) = 0.5;
  host_values(8) = 0.3;
  host_values(9) = 0.2;
  Kokkos::deep_copy(values, host_values);

  Kokkos::parallel_for(
      "riemann_device", kSolverCount, KOKKOS_LAMBDA(int index) {
        const auto solver = static_cast<RiemannSolver>(index);
        const PrimitiveState left{values(0), values(1), values(2), values(3),
                                  values(4)};
        const PrimitiveState right{values(5), values(6), values(7), values(8),
                                   values(9)};
        const Flux result = flux(solver, left, right, kGamma);
        output(5 * index + 0) = result.rho;
        output(5 * index + 1) = result.mom_n;
        output(5 * index + 2) = result.mom_t1;
        output(5 * index + 3) = result.mom_t2;
        output(5 * index + 4) = result.energy;
      });

  auto host_output =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
  const PrimitiveState left{1.0, 0.3, 0.2, -0.1, 0.9};
  const PrimitiveState right{0.4, -0.2, 0.5, 0.3, 0.2};
  for (int index = 0; index < kSolverCount; ++index) {
    const auto solver = static_cast<RiemannSolver>(index);
    const Flux expected = flux(solver, left, right, kGamma);
    const Flux actual{host_output(5 * index + 0), host_output(5 * index + 1),
                      host_output(5 * index + 2), host_output(5 * index + 3),
                      host_output(5 * index + 4)};
    check_flux("device flux", solver_name(solver), actual, expected, 1.0e-14);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    test_uniform_consistency();
    test_stationary_contact();
    test_moving_contact();
    test_symmetry();
    test_sod_fluxes();
    test_low_mach_dissipation();
    test_device_kernel();

    if (g_failures == 0) {
      std::printf("saturnCFD Riemann tests passed\n");
    } else {
      std::printf("saturnCFD Riemann tests FAILED: %d failure(s)\n",
                  g_failures);
    }
  }
  Kokkos::finalize();
  return g_failures == 0 ? 0 : 1;
}
