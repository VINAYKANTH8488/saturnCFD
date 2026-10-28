#pragma once

#include <Kokkos_Core.hpp>

namespace saturn {

// Face-reconstruction schemes for cell-centered finite volumes.
//
// All schemes reconstruct the state at the face between cells i and i+1 from
// the six-point stencil
//
//   v[-2] v[-1] v[0] | v[1] v[2] v[3]
//                     ^ face
//
// `reconstruct_left` returns the state on the left of the face (upwind for a
// positive velocity); `reconstruct_right` mirrors the stencil around the face.
// The caller supplies ghost cells; no boundary handling is done here.
//
// Schemes:
//   Central2  - two-point average, 2nd order.
//   Central4  - symmetric four-point reconstruction, 4th order.
//   Central6  - symmetric six-point reconstruction, 6th order.
//   Weno5Z    - WENO5 with Borges et al. (2008) Z weights, 5th order.
//   Teno6     - TENO6: three 3-point WENO candidates plus a symmetric
//               6-point candidate selected by an inverse-smoothness cutoff.
//               In smooth regions the 6-point candidate is retained, so the
//               scheme reduces to Central6; near discontinuities the cutoff
//               falls back to an ENO-like upwind candidate.
//
// TENO6 follows the incremental-stencil selection of Fu et al. (2021):
// gamma_k = (beta_k + eps)^-7, chi_k = gamma_k / sum(gamma), and a candidate
// is active when chi_k >= C_T. The six-point candidate is tried first; if it
// is rejected, the three small candidates are renormalized with their WENO5
// linear weights.
enum class Reconstruction {
  Central2,
  Central4,
  Central6,
  Weno5Z,
  Teno6,
};

// Number of cells touched by a scheme (the largest offset is 3 on each side).
constexpr int reconstruction_stencil_width(Reconstruction scheme) noexcept {
  switch (scheme) {
    case Reconstruction::Central2:
      return 2;
    case Reconstruction::Central4:
      return 4;
    case Reconstruction::Central6:
    case Reconstruction::Teno6:
      return 6;
    case Reconstruction::Weno5Z:
      return 5;
  }
  return 6;
}

namespace recon {

// WENO/TENO parameters. The inverse-smoothness exponent and cutoff are the
// values recommended for TENO6 (Fu et al., JCP 2021).
inline constexpr double kEpsilon = 1.0e-40;
inline constexpr double kTenoCutoff = 1.0e-5;
inline constexpr double kTenoExponent = 7.0;  // applied as x^7

// WENO5 finite-volume linear weights for the backward, central, and forward
// 3-point candidates. These combine to the 5th-order face value.
inline constexpr double kWenoWeight0 = 1.0 / 10.0;
inline constexpr double kWenoWeight1 = 3.0 / 5.0;
inline constexpr double kWenoWeight2 = 3.0 / 10.0;

KOKKOS_INLINE_FUNCTION
double central2(double /*vm2*/, double /*vm1*/, double v0, double vp1,
                double /*vp2*/, double /*vp3*/) {
  return 0.5 * (v0 + vp1);
}

KOKKOS_INLINE_FUNCTION
double central4(double /*vm2*/, double vm1, double v0, double vp1, double vp2,
                double /*vp3*/) {
  return (-vm1 + 7.0 * v0 + 7.0 * vp1 - vp2) / 12.0;
}

KOKKOS_INLINE_FUNCTION
double central6(double vm2, double vm1, double v0, double vp1, double vp2,
                double vp3) {
  return (vm2 - 8.0 * vm1 + 37.0 * v0 + 37.0 * vp1 - 8.0 * vp2 + vp3) / 60.0;
}

// Jiang-Shu smoothness indicators for the three 3-point WENO5 candidates.
// Ordered backward ({-2,-1,0}), central ({-1,0,1}), forward ({0,1,2}).
KOKKOS_INLINE_FUNCTION
void weno5_smoothness(double vm2, double vm1, double v0, double vp1, double vp2,
                      double& beta0, double& beta1, double& beta2) {
  const double d0 = vm2 - 2.0 * vm1 + v0;
  const double d1 = vm1 - 2.0 * v0 + vp1;
  const double d2 = v0 - 2.0 * vp1 + vp2;
  beta0 = (13.0 / 12.0) * d0 * d0 +
          0.25 * (vm2 - 4.0 * vm1 + 3.0 * v0) * (vm2 - 4.0 * vm1 + 3.0 * v0);
  beta1 = (13.0 / 12.0) * d1 * d1 + 0.25 * (vm1 - vp1) * (vm1 - vp1);
  beta2 = (13.0 / 12.0) * d2 * d2 +
          0.25 * (3.0 * v0 - 4.0 * vp1 + vp2) * (3.0 * v0 - 4.0 * vp1 + vp2);
}

// WENO5-Z (Borges et al. 2008) with finite-volume linear weights.
KOKKOS_INLINE_FUNCTION
double weno5z(double vm2, double vm1, double v0, double vp1, double vp2,
              double /*vp3*/) {
  const double q0 = (2.0 * vm2 - 7.0 * vm1 + 11.0 * v0) / 6.0;
  const double q1 = (-vm1 + 5.0 * v0 + 2.0 * vp1) / 6.0;
  const double q2 = (2.0 * v0 + 5.0 * vp1 - vp2) / 6.0;

  double beta0 = 0.0;
  double beta1 = 0.0;
  double beta2 = 0.0;
  weno5_smoothness(vm2, vm1, v0, vp1, vp2, beta0, beta1, beta2);

  // Normalize by the largest indicator so that the nonlinear weights are
  // scale invariant and cannot overflow for very large jumps.
  double beta_max = beta0 > beta1 ? beta0 : beta1;
  beta_max = beta_max > beta2 ? beta_max : beta2;
  double n0 = 0.0;
  double n1 = 0.0;
  double n2 = 0.0;
  double tau = 0.0;
  if (beta_max > 0.0) {
    n0 = beta0 / beta_max;
    n1 = beta1 / beta_max;
    n2 = beta2 / beta_max;
    tau = n0 > n2 ? n0 - n2 : n2 - n0;
  }

  const double a0 = kWenoWeight0 * (1.0 + tau / (n0 + kEpsilon)) *
                    (1.0 + tau / (n0 + kEpsilon));
  const double a1 = kWenoWeight1 * (1.0 + tau / (n1 + kEpsilon)) *
                    (1.0 + tau / (n1 + kEpsilon));
  const double a2 = kWenoWeight2 * (1.0 + tau / (n2 + kEpsilon)) *
                    (1.0 + tau / (n2 + kEpsilon));

  const double sum = a0 + a1 + a2;
  if (!(sum > 0.0)) {
    return q1;
  }
  return (a0 * q0 + a1 * q1 + a2 * q2) / sum;
}

// Jiang-Shu smoothness indicator for the symmetric six-point stencil
// {-2,...,3}, integrated over the cell adjacent to the face. The rational
// matrix is the exact derivative integral (denominator 120960).
KOKKOS_INLINE_FUNCTION
double stencil6_smoothness(double vm2, double vm1, double v0, double vp1,
                           double vp2, double vp3) {
  const double v[6] = {vm2, vm1, v0, vp1, vp2, vp3};
  const double kNumerator[6][6] = {
      {271779.0, -1190400.0, 2043176.0, -1731126.0, 729381.0, -122810.0},
      {-1190400.0, 5653317.0, -10213942.0, 8952516.0, -3863994.0, 662503.0},
      {2043176.0, -10213942.0, 19510972.0, -17908832.0, 7964956.0, -1396330.0},
      {-1731126.0, 8952516.0, -17908832.0, 17195652.0, -7940202.0, 1431992.0},
      {729381.0, -3863994.0, 7964956.0, -7940202.0, 3824847.0, -714988.0},
      {-122810.0, 662503.0, -1396330.0, 1431992.0, -714988.0, 139633.0}};

  double sum = 0.0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      sum += kNumerator[i][j] * v[i] * v[j];
    }
  }
  return sum / 120960.0;
}

// gamma = (normalized_beta + eps)^-7, evaluated with multiplications only.
KOKKOS_INLINE_FUNCTION
double teno_inverse_smoothness(double normalized_beta) {
  const double inv = 1.0 / (normalized_beta + kEpsilon);
  const double inv2 = inv * inv;
  const double inv4 = inv2 * inv2;
  return inv4 * inv2 * inv;
}

KOKKOS_INLINE_FUNCTION
double teno6(double vm2, double vm1, double v0, double vp1, double vp2,
             double vp3) {
  const double q0 = (2.0 * vm2 - 7.0 * vm1 + 11.0 * v0) / 6.0;
  const double q1 = (-vm1 + 5.0 * v0 + 2.0 * vp1) / 6.0;
  const double q2 = (2.0 * v0 + 5.0 * vp1 - vp2) / 6.0;
  const double q6 = central6(vm2, vm1, v0, vp1, vp2, vp3);

  double beta0 = 0.0;
  double beta1 = 0.0;
  double beta2 = 0.0;
  weno5_smoothness(vm2, vm1, v0, vp1, vp2, beta0, beta1, beta2);
  const double beta6 = stencil6_smoothness(vm2, vm1, v0, vp1, vp2, vp3);

  double beta_max = beta0 > beta1 ? beta0 : beta1;
  beta_max = beta_max > beta2 ? beta_max : beta2;
  beta_max = beta_max > beta6 ? beta_max : beta6;
  if (!(beta_max > 0.0)) {
    return q6;
  }

  // gamma_k = (beta_k / beta_max + eps)^-7. The common beta_max^-7 factor
  // cancels in the normalized chi_k, so this is scale invariant and safe.
  const double g0 = teno_inverse_smoothness(beta0 / beta_max);
  const double g1 = teno_inverse_smoothness(beta1 / beta_max);
  const double g2 = teno_inverse_smoothness(beta2 / beta_max);
  const double g6 = teno_inverse_smoothness(beta6 / beta_max);

  const double sum6 = g0 + g1 + g2 + g6;
  if (!(sum6 > 0.0)) {
    return q6;
  }
  if (g6 / sum6 >= kTenoCutoff) {
    return q6;
  }

  // Large stencil rejected: apply the TENO cutoff to the small candidates.
  const double sum3 = g0 + g1 + g2;
  if (!(sum3 > 0.0)) {
    return q1;
  }
  const double delta0 = (g0 / sum3 >= kTenoCutoff) ? 1.0 : 0.0;
  const double delta1 = (g1 / sum3 >= kTenoCutoff) ? 1.0 : 0.0;
  const double delta2 = (g2 / sum3 >= kTenoCutoff) ? 1.0 : 0.0;

  const double w0 = kWenoWeight0 * delta0;
  const double w1 = kWenoWeight1 * delta1;
  const double w2 = kWenoWeight2 * delta2;
  const double sum_w = w0 + w1 + w2;
  if (!(sum_w > 0.0)) {
    return q1;
  }
  return (w0 * q0 + w1 * q1 + w2 * q2) / sum_w;
}

// Reconstruct the left state at the face between v0 and vp1.
KOKKOS_INLINE_FUNCTION
double reconstruct_left(Reconstruction scheme, double vm2, double vm1,
                        double v0, double vp1, double vp2, double vp3) {
  switch (scheme) {
    case Reconstruction::Central2:
      return central2(vm2, vm1, v0, vp1, vp2, vp3);
    case Reconstruction::Central4:
      return central4(vm2, vm1, v0, vp1, vp2, vp3);
    case Reconstruction::Central6:
      return central6(vm2, vm1, v0, vp1, vp2, vp3);
    case Reconstruction::Weno5Z:
      return weno5z(vm2, vm1, v0, vp1, vp2, vp3);
    case Reconstruction::Teno6:
      return teno6(vm2, vm1, v0, vp1, vp2, vp3);
  }
  return central2(vm2, vm1, v0, vp1, vp2, vp3);
}

// Reconstruct the right state at the same face by mirroring the stencil.
KOKKOS_INLINE_FUNCTION
double reconstruct_right(Reconstruction scheme, double vm2, double vm1,
                         double v0, double vp1, double vp2, double vp3) {
  return reconstruct_left(scheme, vp3, vp2, vp1, v0, vm1, vm2);
}

// Convenience overload for a callable indexed by an offset in [-2, 3], e.g.
//
//   reconstruct_left(scheme, [&](int d) { return c(0, i + d, j, k); });
//
// The callable may be a device lambda.
template <class Accessor>
KOKKOS_INLINE_FUNCTION double reconstruct_left(Reconstruction scheme,
                                               const Accessor& accessor) {
  return reconstruct_left(scheme, accessor(-2), accessor(-1), accessor(0),
                          accessor(1), accessor(2), accessor(3));
}

template <class Accessor>
KOKKOS_INLINE_FUNCTION double reconstruct_right(Reconstruction scheme,
                                                const Accessor& accessor) {
  return reconstruct_right(scheme, accessor(-2), accessor(-1), accessor(0),
                           accessor(1), accessor(2), accessor(3));
}

}  // namespace recon
}  // namespace saturn
