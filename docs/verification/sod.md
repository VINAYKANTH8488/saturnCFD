# Verification: 1D Sod shock tube

**Date:** 2026-10-27
**Status:** passing on CPU for WENO5-Z, TENO6, and central4 reconstruction with
LLF, HLLC, HLLC-low-Mach, and AUSM+-up fluxes. The driver is a host-only 1D
verification tool, not the production 3D solver.
**Code:** `apps/euler1d.cpp`, `src/numerics/compressible/reconstruction.hpp`,
`src/numerics/compressible/riemann.hpp`

## Purpose

Gate the reconstruction and Riemann-solver modules end-to-end before they are
wired into the 3D compressible solver. The driver evolves the 1D ideal-gas
Euler equations,

```
U_t + F(U)_x = 0
U = [rho, rho u, E]
F = [rho u, rho u^2 + p, u (E + p)]
p = (gamma - 1) (E - 0.5 rho u^2)
```

with a cell-centered finite-volume update, component-wise primitive
reconstruction, a Riemann flux, and SSP-RK3.

## Setup

- Domain `[0, 1]`, 400 cells unless stated otherwise.
- Sod initial condition at `x0 = 0.5`:
  `(rho, u, p) = (1, 0, 1)` for `x < x0`, `(0.125, 0, 0.1)` otherwise.
- `gamma = 1.4`, `CFL = 0.4`, final time `t = 0.2`.
- Zero-order ghost cells; at `t = 0.2` the rarefaction head is near `x = 0.12`
  and the shock near `x = 0.85`, so the waves do not reach the boundaries.
- If a high-order reconstruction produces a non-positive density or pressure
  at a face, that face falls back to first-order states.
- SSP-RK3: `U1 = U + dt L(U)`,
  `U2 = 3/4 U + 1/4 (U1 + dt L(U1))`,
  `U = 1/3 U + 2/3 (U2 + dt L(U2))`.

The exact solution is the standard two-rarefaction/two-shock Riemann solution
with `p*` found by bisection; the driver samples it at every cell center.

## Exact reference

```
p*        = 0.30313018
u*        = 0.92745262
rho*_L    = 0.42631943    (left star, between rarefaction tail and contact)
rho*_R    = 0.26557371    (right star, between contact and shock)
```

## Results

`ctest -R saturn.sod` runs the default WENO5-Z + HLLC case. All combinations
below were also run manually with `--check`; the L1 errors are measured against
the exact solution at the final time.

| Reconstruction | Solver | L1 rho | L1 u | L1 p | Linf rho |
|---|---|---|---|---|---|
| WENO5-Z | HLLC | 1.113e-03 | 2.320e-03 | 8.258e-04 | 5.845e-02 |
| TENO6 | HLLC | 1.041e-03 | 2.292e-03 | 7.968e-04 | 5.848e-02 |
| central4 | HLLC | 1.640e-02 | 6.160e-02 | 2.046e-02 | 2.126e-01 |
| WENO5-Z | HLLC-LM | 1.091e-03 | 2.282e-03 | 7.978e-04 | 5.861e-02 |
| WENO5-Z | AUSM+-up | 2.054e-03 | 3.786e-03 | 1.798e-03 | 8.585e-02 |
| WENO5-Z | LLF | 1.381e-03 | 2.637e-03 | 9.758e-04 | 6.234e-02 |

Star-region densities at `x = 0.60` and `x = 0.75`:

| Case | rho(0.60) | rho(0.75) |
|---|---|---|
| exact | 0.426319 | 0.265574 |
| WENO5-Z + HLLC | 0.426348 | 0.265570 |
| TENO6 + HLLC | 0.426363 | 0.265543 |

The non-dissipative `central4` scheme is included only as a control: it is not
shock-capturing and its larger errors are expected. The production path will
default to WENO5-Z or TENO6 for shocks.

## Commands

```bash
./build/cpu-openmp/apps/saturn_euler1d --n 400 --check
./build/cpu-openmp/apps/saturn_euler1d \
  --n 400 --scheme teno6 --solver hllc_lm --check
./build/cpu-openmp/apps/saturn_euler1d \
  --n 400 --scheme weno5z --solver hllc --output /tmp/sod.csv
```

`--output` writes cell-center values and the exact solution as CSV.

```bash
ctest --test-dir build/cpu-openmp -R saturn.sod --output-on-failure
```

## Limits and next steps

- The driver is 1D and host-only. It does not exercise MPI halos, the GPU
  backend, or the 3D patch layout.
- Only the inviscid Euler equations are implemented; compressible
  Navier-Stokes, viscous fluxes, and physical boundary conditions are pending.
- Next verification targets: double Mach reflection, low-Mach vortex, acoustic
  pulse, and a smooth high-order MMS to measure the design order (Sod is
  dominated by discontinuities and therefore first-order in L1).
- Positivity preservation is currently a face-level first-order fallback; a
  cell-level positivity limiter should be added before strong blast-wave cases.
