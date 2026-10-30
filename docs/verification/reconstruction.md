# Verification: face reconstruction (central, WENO5-Z, TENO6)

**Date:** 2026-10-27
**Status:** unit and convergence tests pass on CPU (Kokkos Serial + OpenMP).
The same functions are device-callable; a CUDA run is pending GPU access in
the current environment.
**Code:** `src/numerics/compressible/reconstruction.hpp`, `tests/test_reconstruction.cpp`

## Purpose

Provide the dimension-by-dimension face reconstruction used by the
compressible solver. Every scheme reconstructs a state at the face between
cells `i` and `i+1` from cell averages on the stencil

```
v[-2]  v[-1]  v[0] | v[1]  v[2]  v[3]
                   ^ face i+1/2
```

`reconstruct_left` returns the left state; `reconstruct_right` mirrors the
stencil around the face. Ghost cells are supplied by the caller, so no
boundary condition is embedded in the operators.

## Schemes

| Scheme | Type | Formal order | Stencil |
|---|---|---|---|
| `central2` | symmetric average | 2 | 2 |
| `central4` | symmetric | 4 | 4 |
| `central6` | symmetric | 6 | 6 |
| `weno5z` | WENO5 with Z weights | 5 | 5 |
| `teno6` | targeted ENO | 6 (smooth) | 6 |

### Central schemes

```
central2 = (v0 + v1) / 2

central4 = (-v[-1] + 7 v0 + 7 v1 - v2) / 12

central6 = (v[-2] - 8 v[-1] + 37 v0 + 37 v1 - 8 v2 + v3) / 60
```

`central4` and `central6` are the standard symmetric finite-volume
reconstructions. They are non-dissipative but not shock-capturing.

### WENO5-Z

The three 3-point candidates are

```
q0 = (2 v[-2] - 7 v[-1] + 11 v0) / 6
q1 = (-v[-1] + 5 v0 + 2 v1) / 6
q2 = (2 v0 + 5 v1 - v2) / 6
```

with finite-volume linear weights `d = (1/10, 3/5, 3/10)`; their weighted sum
is the 5-point, 5th-order face value. The Jiang-Shu smoothness indicators
`beta0, beta1, beta2` feed the WENO-Z nonlinear weights

```
tau  = |beta0 - beta2|
alpha_k = d_k (1 + tau / (beta_k + eps))^2
w_k  = alpha_k / sum(alpha)
```

with `eps = 1e-40`. The indicators are normalized by their maximum before the
weight computation, which keeps the ratio scale invariant and prevents
overflow for very large jumps.

### TENO6

TENO6 follows the incremental-stencil selection of Fu et al. (2021):

1. Build the three WENO5 candidates above and the symmetric six-point value
   `q6 = central6`.
2. Compute the 3-point indicators and the six-point indicator
   `beta6` (the exact derivative integral, denominator 120960).
3. Form `gamma_k = (beta_k / beta_max + eps)^-7` and
   `chi_k = gamma_k / sum(gamma)`.
4. If `chi6 >= C_T` (`C_T = 1e-5`), retain the six-point candidate and return
   `q6`. In smooth flow this makes TENO6 identical to `central6`, i.e. it adds
   no nonlinear dissipation where none is needed.
5. Otherwise reject the large stencil and apply the same cutoff to the three
   small candidates, renormalizing their WENO5 linear weights:
   `w_k = d_k delta_k / sum(d_j delta_j)`, with `delta_k = 1` when
   `chi_k >= C_T`.
6. If every small candidate is rejected, return `q1` (the centered 3-point
   candidate) as the robust fallback.

Normalizing both the WENO-Z and TENO indicators by `beta_max` is a numerical
detail: it leaves every ratio unchanged but avoids overflow/underflow in
`beta^-7` for extreme states.

## Test results

`ctest -R saturn.reconstruction` runs exactness, convergence, symmetry,
shock-boundedness, TENO fallback, and a device-kernel check.

Polynomial exactness (both left and right states):

- constant and linear: all schemes;
- quadratic: `central4`, `central6`, `weno5z`, `teno6`;
- cubic: `central4`, `central6`, `teno6`;
- quintic: `central6`, `teno6` (TENO6 selects the six-point candidate).

Smooth periodic sine, unit box, phase shift 0.3, L1 face error:

| Scheme | N=32 | N=64 | observed order |
|---|---|---|---|
| `central2` | 4.089e-03 | 1.022e-03 | 2.00 |
| `central4` | 3.145e-05 | 1.968e-06 | 4.00 |
| `central6` | 2.590e-07 | 4.062e-09 | 5.99 |
| `weno5z` | 3.268e-06 | 9.783e-08 | 5.06 |
| `teno6` | 2.590e-07 | 4.062e-09 | 5.99 |

Across a unit step, both `weno5z` and `teno6` return face values bounded by
`[0, 1]` for every face within the stencil. At the face immediately left of
the jump, TENO6 rejects the six-point candidate and returns the upwind value
`0`, while `central6` returns `0.5`; this is the intended targeted-ENO
behaviour and is asserted by the test.

## Usage

```cpp
#include "numerics/compressible/reconstruction.hpp"

saturn::Reconstruction scheme = saturn::Reconstruction::Weno5Z;

// Inside a KOKKOS_LAMBDA over a PatchField c:
const double left = saturn::recon::reconstruct_left(
    scheme, [&](int d) { return c(0, i + d, j, k); });
const double right = saturn::recon::reconstruct_right(
    scheme, [&](int d) { return c(0, i + d, j, k); });
```

The explicit six-value overload is also available:

```cpp
reconstruct_left(scheme, vm2, vm1, v0, vp1, vp2, vp3);
```

## Remaining work

- Boundary closures that preserve the nominal order (currently the caller
  must provide valid ghost cells).
- Integration into the compressible flux/Riemann path and a high-order MMS.
- CUDA/HIP/SYCL verification once the backends are available; the operators
  use no host-only code.
- Cost measurement for fusing reconstruction with the flux and divergence
  kernels (the intended production kernel structure).
