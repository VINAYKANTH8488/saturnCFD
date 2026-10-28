#pragma once

#include <Kokkos_Core.hpp>

namespace saturn {

// Inviscid fluxes for the ideal-gas Euler equations in normal coordinates:
//
//   U = [rho, rho u, rho v, rho w, E]
//   F = [rho u, rho u^2 + p, rho u v, rho u w, u (E + p)]
//
// Here u is the velocity normal to the face and v, w are tangential. The caller
// rotates the state into this frame before calling a solver and rotates the
// returned flux back. All functions are device-callable.
enum class RiemannSolver {
  Llf,
  Hllc,
  HllcLowMach,
  AusmPlusUp,
};

struct PrimitiveState {
  double rho;
  double u;  // normal velocity
  double v;  // first tangential velocity
  double w;  // second tangential velocity
  double p;
};

struct ConservativeState {
  double rho;
  double mom_n;
  double mom_t1;
  double mom_t2;
  double energy;
};

struct Flux {
  double rho;
  double mom_n;
  double mom_t1;
  double mom_t2;
  double energy;
};

namespace riemann {

KOKKOS_INLINE_FUNCTION
bool valid(const PrimitiveState& state) {
  return state.rho > 0.0 && state.p > 0.0 && state.rho < 1.0e300 &&
         state.p < 1.0e300;
}

// Maximum-entropy wave speed estimate used by the local Lax-Friedrichs flux.
KOKKOS_INLINE_FUNCTION
double max_signal_speed(const PrimitiveState& left,
                        const PrimitiveState& right, double gamma) {
  const double c_left = Kokkos::sqrt(gamma * left.p / left.rho);
  const double c_right = Kokkos::sqrt(gamma * right.p / right.rho);
  const double a_left = Kokkos::fabs(left.u) + c_left;
  const double a_right = Kokkos::fabs(right.u) + c_right;
  return a_left > a_right ? a_left : a_right;
}

KOKKOS_INLINE_FUNCTION
ConservativeState to_conservative(const PrimitiveState& state, double gamma) {
  const double kinetic =
      0.5 * state.rho *
      (state.u * state.u + state.v * state.v + state.w * state.w);
  const double energy = state.p / (gamma - 1.0) + kinetic;
  return ConservativeState{state.rho, state.rho * state.u,
                           state.rho * state.v, state.rho * state.w, energy};
}

KOKKOS_INLINE_FUNCTION
PrimitiveState to_primitive(const ConservativeState& state, double gamma) {
  const double rho = state.rho;
  const double inv_rho = rho > 0.0 ? 1.0 / rho : 0.0;
  const double u = state.mom_n * inv_rho;
  const double v = state.mom_t1 * inv_rho;
  const double w = state.mom_t2 * inv_rho;
  const double kinetic =
      0.5 * (state.mom_n * state.mom_n + state.mom_t1 * state.mom_t1 +
             state.mom_t2 * state.mom_t2) *
      inv_rho;
  const double pressure = (gamma - 1.0) * (state.energy - kinetic);
  return PrimitiveState{rho, u, v, w, pressure};
}

KOKKOS_INLINE_FUNCTION
Flux physical_flux(const PrimitiveState& state, double gamma) {
  const double energy = state.p / (gamma - 1.0) +
                        0.5 * state.rho *
                            (state.u * state.u + state.v * state.v +
                             state.w * state.w);
  return Flux{state.rho * state.u, state.rho * state.u * state.u + state.p,
              state.rho * state.u * state.v, state.rho * state.u * state.w,
              state.u * (energy + state.p)};
}

KOKKOS_INLINE_FUNCTION
Flux average_flux(const Flux& left, const Flux& right) {
  return Flux{0.5 * (left.rho + right.rho),
              0.5 * (left.mom_n + right.mom_n),
              0.5 * (left.mom_t1 + right.mom_t1),
              0.5 * (left.mom_t2 + right.mom_t2),
              0.5 * (left.energy + right.energy)};
}

// Local Lax-Friedrichs / Rusanov flux. Robust fallback and comparison scheme.
KOKKOS_INLINE_FUNCTION
Flux llf(const PrimitiveState& left, const PrimitiveState& right,
         double gamma) {
  const Flux flux_left = physical_flux(left, gamma);
  const Flux flux_right = physical_flux(right, gamma);
  if (!valid(left) || !valid(right)) {
    return average_flux(flux_left, flux_right);
  }

  const ConservativeState u_left = to_conservative(left, gamma);
  const ConservativeState u_right = to_conservative(right, gamma);
  const double alpha = max_signal_speed(left, right, gamma);
  return Flux{
      0.5 * (flux_left.rho + flux_right.rho) -
          0.5 * alpha * (u_right.rho - u_left.rho),
      0.5 * (flux_left.mom_n + flux_right.mom_n) -
          0.5 * alpha * (u_right.mom_n - u_left.mom_n),
      0.5 * (flux_left.mom_t1 + flux_right.mom_t1) -
          0.5 * alpha * (u_right.mom_t1 - u_left.mom_t1),
      0.5 * (flux_left.mom_t2 + flux_right.mom_t2) -
          0.5 * alpha * (u_right.mom_t2 - u_left.mom_t2),
      0.5 * (flux_left.energy + flux_right.energy) -
          0.5 * alpha * (u_right.energy - u_left.energy)};
}

// HLLC with Roe-averaged wave-speed bounds.
KOKKOS_INLINE_FUNCTION
Flux hllc_impl(const PrimitiveState& left, const PrimitiveState& right,
               double gamma) {
  const Flux flux_left = physical_flux(left, gamma);
  const Flux flux_right = physical_flux(right, gamma);
  if (!valid(left) || !valid(right)) {
    return average_flux(flux_left, flux_right);
  }
  const ConservativeState u_left = to_conservative(left, gamma);
  const ConservativeState u_right = to_conservative(right, gamma);

  const double c_left = Kokkos::sqrt(gamma * left.p / left.rho);
  const double c_right = Kokkos::sqrt(gamma * right.p / right.rho);
  const double h_left = (u_left.energy + left.p) / left.rho;
  const double h_right = (u_right.energy + right.p) / right.rho;

  // Roe averages give tighter wave-speed bounds than the plain Davis
  // estimate while remaining cheap.
  const double sqrt_left = Kokkos::sqrt(left.rho);
  const double sqrt_right = Kokkos::sqrt(right.rho);
  const double inv_sum = 1.0 / (sqrt_left + sqrt_right);
  const double u_roe = (sqrt_left * left.u + sqrt_right * right.u) * inv_sum;
  const double v_roe = (sqrt_left * left.v + sqrt_right * right.v) * inv_sum;
  const double w_roe = (sqrt_left * left.w + sqrt_right * right.w) * inv_sum;
  const double h_roe =
      (sqrt_left * h_left + sqrt_right * h_right) * inv_sum;
  double c_roe_sq = (gamma - 1.0) *
                    (h_roe - 0.5 * (u_roe * u_roe + v_roe * v_roe +
                                    w_roe * w_roe));
  if (!(c_roe_sq > 0.0)) {
    c_roe_sq = 0.0;
  }
  const double c_roe = Kokkos::sqrt(c_roe_sq);

  double s_left = left.u - c_left;
  const double s_left_roe = u_roe - c_roe;
  if (s_left_roe < s_left) {
    s_left = s_left_roe;
  }
  double s_right = right.u + c_right;
  const double s_right_roe = u_roe + c_roe;
  if (s_right_roe > s_right) {
    s_right = s_right_roe;
  }

  if (!(s_left < s_right)) {
    return llf(left, right, gamma);
  }

  const double denom =
      left.rho * (s_left - left.u) - right.rho * (s_right - right.u);
  if (Kokkos::fabs(denom) < 1.0e-300) {
    return llf(left, right, gamma);
  }
  const double s_star =
      (right.p - left.p + left.rho * left.u * (s_left - left.u) -
       right.rho * right.u * (s_right - right.u)) /
      denom;

  if (s_left >= 0.0) {
    return flux_left;
  }
  if (s_right <= 0.0) {
    return flux_right;
  }
  if (!(s_star > s_left && s_star < s_right)) {
    return llf(left, right, gamma);
  }

  if (s_star >= 0.0) {
    const double delta = s_left - s_star;
    if (Kokkos::fabs(delta) < 1.0e-300 ||
        Kokkos::fabs(s_left - left.u) < 1.0e-300) {
      return llf(left, right, gamma);
    }
    const double factor = left.rho * (s_left - left.u) / delta;
    if (!(factor > 0.0)) {
      return llf(left, right, gamma);
    }
    const ConservativeState star{
        factor, factor * s_star, factor * left.v, factor * left.w,
        factor * (u_left.energy / left.rho +
                  (s_star - left.u) *
                      (s_star + left.p /
                                    (left.rho * (s_left - left.u))))};
    return Flux{flux_left.rho + s_left * (star.rho - u_left.rho),
                flux_left.mom_n + s_left * (star.mom_n - u_left.mom_n),
                flux_left.mom_t1 + s_left * (star.mom_t1 - u_left.mom_t1),
                flux_left.mom_t2 + s_left * (star.mom_t2 - u_left.mom_t2),
                flux_left.energy + s_left * (star.energy - u_left.energy)};
  }

  const double delta = s_right - s_star;
  if (Kokkos::fabs(delta) < 1.0e-300 ||
      Kokkos::fabs(s_right - right.u) < 1.0e-300) {
    return llf(left, right, gamma);
  }
  const double factor = right.rho * (s_right - right.u) / delta;
  if (!(factor > 0.0)) {
    return llf(left, right, gamma);
  }
  const ConservativeState star{
      factor, factor * s_star, factor * right.v, factor * right.w,
      factor * (u_right.energy / right.rho +
                (s_star - right.u) *
                    (s_star + right.p /
                                  (right.rho * (s_right - right.u))))};
  return Flux{flux_right.rho + s_right * (star.rho - u_right.rho),
              flux_right.mom_n + s_right * (star.mom_n - u_right.mom_n),
              flux_right.mom_t1 + s_right * (star.mom_t1 - u_right.mom_t1),
              flux_right.mom_t2 + s_right * (star.mom_t2 - u_right.mom_t2),
              flux_right.energy + s_right * (star.energy - u_right.energy)};
}

KOKKOS_INLINE_FUNCTION
Flux hllc(const PrimitiveState& left, const PrimitiveState& right,
          double gamma) {
  return hllc_impl(left, right, gamma);
}

// HLLC with a low-Mach correction in the spirit of Fleischmann et al.: the
// acoustic (HLLC) flux is blended with the central flux using
//
//   phi = sin(min(1, Ma_local / Ma_limit) * pi/2),
//   Ma_local = max(|u_L|/c_L, |u_R|/c_R).
//
// phi = 1 above Ma_limit, so the scheme is exactly HLLC in compressible flow.
// Below Ma_limit the blend removes the excessive acoustic dissipation that
// smears pressure waves at low Mach. A pressure-jump/velocity-jump sensor
// keeps full HLLC upwinding across shocks and strong expansions, where the
// central component would be unsafe. Ma_limit = 0.1 is the recommended value.
KOKKOS_INLINE_FUNCTION
Flux hllc_low_mach(const PrimitiveState& left, const PrimitiveState& right,
                   double gamma, double mach_limit = 0.1) {
  if (!valid(left) || !valid(right) || !(mach_limit > 0.0)) {
    return llf(left, right, gamma);
  }
  const double c_left = Kokkos::sqrt(gamma * left.p / left.rho);
  const double c_right = Kokkos::sqrt(gamma * right.p / right.rho);
  const double ma_left = Kokkos::fabs(left.u) / c_left;
  const double ma_right = Kokkos::fabs(right.u) / c_right;
  const double ma_local = ma_left > ma_right ? ma_left : ma_right;
  double ratio = ma_local / mach_limit;
  if (ratio > 1.0) {
    ratio = 1.0;
  }
  const double phi = Kokkos::sin(0.5 * 3.14159265358979323846 * ratio);
  if (phi >= 1.0) {
    return hllc_impl(left, right, gamma);
  }

  const double p_avg = 0.5 * (left.p + right.p);
  const double c_avg = 0.5 * (c_left + c_right);
  const double pressure_jump = Kokkos::fabs(right.p - left.p);
  const double velocity_jump = Kokkos::fabs(right.u - left.u);
  if (pressure_jump > 0.1 * p_avg || velocity_jump > 0.1 * c_avg) {
    return hllc_impl(left, right, gamma);
  }

  const Flux hllc_flux = hllc_impl(left, right, gamma);
  const Flux central_flux =
      average_flux(physical_flux(left, gamma), physical_flux(right, gamma));
  return Flux{phi * hllc_flux.rho + (1.0 - phi) * central_flux.rho,
              phi * hllc_flux.mom_n + (1.0 - phi) * central_flux.mom_n,
              phi * hllc_flux.mom_t1 + (1.0 - phi) * central_flux.mom_t1,
              phi * hllc_flux.mom_t2 + (1.0 - phi) * central_flux.mom_t2,
              phi * hllc_flux.energy + (1.0 - phi) * central_flux.energy};
}

// AUSM+ splitting polynomials.
KOKKOS_INLINE_FUNCTION
double mach_plus_1(double mach) { return 0.5 * (mach + Kokkos::fabs(mach)); }

KOKKOS_INLINE_FUNCTION
double mach_minus_1(double mach) { return 0.5 * (mach - Kokkos::fabs(mach)); }

KOKKOS_INLINE_FUNCTION
double mach_plus_2(double mach) {
  if (Kokkos::fabs(mach) >= 1.0) {
    return mach_plus_1(mach);
  }
  return 0.25 * (mach + 1.0) * (mach + 1.0);
}

KOKKOS_INLINE_FUNCTION
double mach_minus_2(double mach) {
  if (Kokkos::fabs(mach) >= 1.0) {
    return mach_minus_1(mach);
  }
  return -0.25 * (mach - 1.0) * (mach - 1.0);
}

KOKKOS_INLINE_FUNCTION
double pressure_plus_5(double mach, double alpha) {
  if (Kokkos::fabs(mach) >= 1.0) {
    return mach_plus_1(mach) / mach;
  }
  return mach_plus_2(mach) *
         ((2.0 - mach) - 16.0 * alpha * mach * mach_minus_2(mach));
}

KOKKOS_INLINE_FUNCTION
double pressure_minus_5(double mach, double alpha) {
  if (Kokkos::fabs(mach) >= 1.0) {
    return mach_minus_1(mach) / mach;
  }
  return mach_minus_2(mach) *
         ((-2.0 - mach) + 16.0 * alpha * mach * mach_plus_2(mach));
}

// AUSM+-up (Liou 2006) with the standard split Mach polynomials and the
// all-speed pressure/mass diffusion terms. `mach_cutoff` is the free-stream
// Mach number used to floor the low-Mach scaling function f_a.
KOKKOS_INLINE_FUNCTION
Flux ausm_plus_up(const PrimitiveState& left, const PrimitiveState& right,
                  double gamma, double mach_cutoff = 0.1) {
  if (!valid(left) || !valid(right)) {
    return llf(left, right, gamma);
  }
  const double c_left = Kokkos::sqrt(gamma * left.p / left.rho);
  const double c_right = Kokkos::sqrt(gamma * right.p / right.rho);
  const double a_face = 0.5 * (c_left + c_right);
  const double inv_a = 1.0 / a_face;

  const double mach_left = left.u * inv_a;
  const double mach_right = right.u * inv_a;

  double mbar_sq =
      (left.u * left.u + right.u * right.u) / (2.0 * a_face * a_face);
  const double cutoff = mach_cutoff > 0.0 ? mach_cutoff : 1.0e-4;
  const double cutoff_sq = cutoff * cutoff;
  if (mbar_sq < cutoff_sq) {
    mbar_sq = cutoff_sq;
  }
  if (mbar_sq > 1.0) {
    mbar_sq = 1.0;
  }
  const double m0 = Kokkos::sqrt(mbar_sq);
  double f_a = m0 * (2.0 - m0);
  if (f_a < 1.0e-8) {
    f_a = 1.0e-8;
  }

  const double alpha = (3.0 / 16.0) * (-4.0 + 5.0 * f_a * f_a);
  const double p_plus_left = pressure_plus_5(mach_left, alpha);
  const double p_minus_right = pressure_minus_5(mach_right, alpha);

  const double rho_face = 0.5 * (left.rho + right.rho);
  double pressure_diffusion =
      -0.75 * p_plus_left * p_minus_right * (left.rho + right.rho) *
      (f_a * a_face) * (right.u - left.u);

  double mass_diffusion = 0.0;
  const double pressure_jump = right.p - left.p;
  if (Kokkos::fabs(pressure_jump) > 0.0) {
    mass_diffusion = -(0.25 / f_a) * (1.0 - mbar_sq) * pressure_jump /
                     (rho_face * a_face * a_face);
  }

  const double mach_face = mach_plus_2(mach_left) +
                           mach_minus_2(mach_right) + mass_diffusion;
  const double mass_flux =
      a_face * mach_face * (mach_face > 0.0 ? left.rho : right.rho);
  const double pressure_face =
      p_plus_left * left.p + p_minus_right * right.p + pressure_diffusion;

  const PrimitiveState& upwind = mass_flux > 0.0 ? left : right;
  const double energy = upwind.p / (gamma - 1.0) +
                        0.5 * upwind.rho *
                            (upwind.u * upwind.u + upwind.v * upwind.v +
                             upwind.w * upwind.w);
  const double enthalpy = (energy + upwind.p) / upwind.rho;

  return Flux{mass_flux, mass_flux * upwind.u + pressure_face,
              mass_flux * upwind.v, mass_flux * upwind.w,
              mass_flux * enthalpy};
}

KOKKOS_INLINE_FUNCTION
Flux flux(RiemannSolver solver, const PrimitiveState& left,
          const PrimitiveState& right, double gamma) {
  switch (solver) {
    case RiemannSolver::Llf:
      return llf(left, right, gamma);
    case RiemannSolver::Hllc:
      return hllc(left, right, gamma);
    case RiemannSolver::HllcLowMach:
      return hllc_low_mach(left, right, gamma);
    case RiemannSolver::AusmPlusUp:
      return ausm_plus_up(left, right, gamma);
  }
  return llf(left, right, gamma);
}

}  // namespace riemann
}  // namespace saturn
