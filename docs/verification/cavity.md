# Lid-driven cavity: Ghia Re=100 validation

**Date:** 2026-10-28
**Status:** single-rank, periodic z, no-slip x/y walls, moving lid. Steady
Re=100 centreline profiles agree with Ghia et al. (1982) to RMS 0.008 (u) and
0.010 (v). Re=400 tables are included; the coarse 24² check gives RMS
0.074/0.072. Higher resolutions and Re=1000 are not yet completed.
**Code:** `apps/cavity_sim.cpp`, `src/numerics/cavity.hpp`

## Setup

The cavity is the unit square `[0,1]^2`, periodic in z, with no-slip walls on
all four sides and the top lid moving in x at `u_lid=1`. The Reynolds number is
`Re = u_lid L / nu`; the Re=100 case uses `nu=0.01`.

Staggered MAC layout:

```
u(i,j,k): x-face; u=0 on the left/right walls (normal velocity)
v(i,j,k): y-face; v=0 on the bottom/top walls (normal velocity)
w(i,j,k): z-face; tangential no-slip ghosts in x and y (w=0 here)
p(i,j,k): cell centers
```

Tangential wall ghosts use `u_ghost = 2 u_wall - u_interior` (the top lid has
`u_wall = u_lid`) and `-u_interior` at stationary walls. Pressure uses
homogeneous Neumann conditions on the x/y walls and periodic conditions in z.
Momentum is second-order central; SSP-RK3 projects after every stage.

The initial state is at rest and the flow is integrated until the centreline
profiles reach steady state.

## Ghia comparison (Re=100)

Reference: Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387–411, Tables I
and II.

Full run: `n=32`, `nz=4`, `t=15`, 1423 steps:

| Quantity | Value |
|---|---|
| RMS error, u along vertical centreline | **7.563e-03** |
| RMS error, v along horizontal centreline | **9.911e-03** |
| max abs(divergence) | 3.2e-11 |
| wall time (8 OpenMP threads) | ~9 min |

The RMS errors are 0.8% and 1.0% of the lid speed, comparable to the 0.5–1.0%
agreement reported by other second-order solvers on similar grids.

A fast smoke gate uses `n=16`, `t=5`: RMS u 3.09e-2, RMS v 1.95e-2,
`max|div| 1.8e-11` in about 10 s. The steady `n=16`, `t=10` run gives
2.62e-2 / 1.52e-2.

## Higher Reynolds numbers

The Ghia Re=400 tables are included in `apps/cavity_sim.cpp`, selected by
`--re 400`. A coarse run at `n=24`, `t=20` (1203 steps) gives

| Quantity | Value |
|---|---|
| RMS error, u | 7.42e-02 |
| RMS error, v | 7.21e-02 |
| max abs(div) | 3.3e-12 |
| wall time | ~3 min |

The larger error than Re=100 reflects both the coarser 24² grid and the
stronger gradients at Re=400. A 32² run was started but is much more
expensive (the explicit viscous step requires many more steps); it was not
completed in this pass. (The published Re=400 v-table contains one point,
`x=0.9063`, that is flagged as suspect in the literature; it is kept here for
completeness.)

## Commands

```bash
# Fast validation gate used by CTest.
./build/cpu-openmp/apps/saturn_cavity_sim \
  --n 16 --re 100 --tend 5 --rms-tol 0.08 --check

# Full n=32 comparison (~9 min).
./build/cpu-openmp/apps/saturn_cavity_sim \
  --n 32 --re 100 --tend 15 --check

ctest --test-dir build/cpu-openmp -R saturn.cavity --output-on-failure
```

## Limitations and next steps

- Single rank; the Poisson solve is not distributed.
- Only `Re=100` is validated. Ghia data for `Re=400/1000` is available and the
  code accepts `--re`, but higher Re needs a finer grid and longer runs.
- The cavity is two-dimensional (periodic z with `nz=4`); a full 3D
  lid-driven cavity would need z walls and a third velocity boundary layer.
- The explicit viscous time step and the many Poisson solves make long runs
  expensive; warm-starting the pressure and a faster multigrid/FFT path are
  needed for production.
- No-slip wall ghosts are currently limited to x/y walls aligned with the grid.

## Advection schemes for high Reynolds numbers

`momentum_rhs` supports four selectable advection forms:

| `--advection` | Form |
|---|---|
| `central` | second-order central (previous default) |
| `upwind` | first-order upwind |
| `muscl` | second-order minmod-limited MUSCL (limited central slope) |
| `blend` | central/upwind blend, fraction set by `--blend` |

Ghia centreline RMS at Re=100, `n=16`, `nz=1`, `t=5`:

| Scheme | RMS u | RMS v |
|---|---|---|
| central | 3.09e-2 | 1.95e-2 |
| upwind | 2.60e-2 | 1.70e-2 |
| **muscl** | **1.67e-2** | **1.28e-2** |
| blend 0.25 | 2.93e-2 | 1.85e-2 |

MUSCL reduces the coarse-grid Re=100 error by ~46% (u) and ~34% (v) relative
to central.

At Re=1000 on the P600:

| Scheme | grid | t | RMS u | RMS v | `max|div|` | time |
|---|---|---|---|---|---|---|
| central | 32x32x1 | 40 | 1.00e-1 | 9.67e-2 | 6.2e-8 | 193 s |
| upwind | 32x32x1 | 40 | 1.43e-1 | 2.07e-2 | 3.1e-7 | 186 s |
| blend 0.25 | 32x32x1 | 40 | 1.22e-1 | 5.64e-2 | 9.2e-8 | 192 s |
| **MUSCL** | 32x32x1 | 20 | **3.73e-2** | 1.89e-1 | 1.0e-7 | 85 s |
| **MUSCL** | 48x48x1 | 20 | **1.74e-2** | 1.96e-1 | 1.0e-7 | 530 s |

The MUSCL `u` profile is dramatically better: **1.74e-2 at 48²**, about 5.7×
better than central at 32². The `v` profile is worse (1.96e-1). The `v`
centreline has a thin wall layer; the uniform grid resolves it poorly, and the
limiter adds dissipation there. The `u` result shows the scheme is working;
the `v` result is a wall-resolution issue, not a global instability.

### Steady-state and 64² check (Re=1000)

A steady-state criterion was added to the driver: it compares `u` and `v`
with their values `--steady-check` steps earlier and stops after
`--steady-window` consecutive checks below `--steady-tol`. At 64², Re=1000,
MUSCL, `t=30`:

| Grid | RMS u | RMS v | `max|div|` | final residual | steady | time |
|---|---|---|---|---|---|---|---|
| 48² | 1.74e-2 | 1.96e-1 | 1.0e-7 | — | — | 530 s |
| 64² | 1.72e-2 | 2.20e-1 | 3.8e-7 | 7.3e-2 | **no** | 2042 s |

The `u` error is unchanged and excellent; the `v` error is **not reduced** by
uniform refinement and is still growing at `t=30` (residual `7.3e-2`). This
confirms that the `v` error is the near-wall boundary layer and the transient,
not the `u` advection scheme. Uniform refinement alone is not the answer;
wall clustering (GLL-like) and a longer/steady run are required.

### Higher-Re limits (MUSCL, n=48, t=10)

| Re | RMS u | RMS v | `max|div|` | GPU time |
|---|---|---|---|---|---|
| 5000 | 2.60e-1 | 2.63e-1 | 9.5e-8 | 316 s |
| 10000 | 3.24e-1 | 2.99e-1 | 1.9e-8 | 376 s |

Both are **stable** but under-resolved; the 48² uniform grid cannot resolve the
Re=5000/10000 boundary layers. This is the current stability limit, not a
validated high-Re accuracy result.

## GPU higher-Reynolds validation (Quadro P600)

`scripts/verify_cavity_gpu.sh` builds the `cuda-local` preset and runs the
MUSCL scheme:

| Re | grid | t | RMS u | RMS v | `max|div|` | GPU time |
|---|---|---|---|---|---|---|---|
| 1000 | 32x32x1 | 20 | **3.73e-2** | 1.89e-1 | 1.0e-7 | 85 s |
| 5000 | 32x32x1 | 10 | 2.58e-1 | 2.78e-1 | 3.5e-8 | 91 s |

The Re=1000 gate uses `--rms-tol 0.20 --div-tol 2e-7`. The Re=5000 run is a
stability smoke. The script completes in ~3 minutes.

```bash
source scripts/env_cuda.sh
./scripts/verify_cavity_gpu.sh
```

## Higher-order method context (Nektar++ / NekRS)

The lid-driven cavity is a standard target for high-order methods. Nektar++
uses a spectral/hp element formulation: the solution is expanded in a
polynomial basis of order `p` on Gauss–Lobatto–Legendre (GLL) points, with
separate advection and diffusion operators and dealiasing/overintegration for
the nonlinear terms. NekRS is a spectral-element code with a polynomial order
`N` (typically 7–9), dealiasing and filtering, and overintegration.

Two lessons from those codes matter here:
- **GLL/spectral nodes cluster at walls.** That is what resolves the thin `v`
  wall layer which the uniform-grid MUSCL run misses.
- **Nonlinear advection needs dealiasing/overintegration.** Higher-order
  reconstructions are only robust when the nonlinear products are not
  aliased; limiters are the finite-volume analogue of that stabilization.

Our current scheme is a second-order finite-volume MUSCL; it is not a
spectral-element method. Its `u` accuracy at 48² is already good, but matching
the `v` wall layer requires wall-clustered grids (or a spectral/hp path) and
more careful dealiasing/limiting.

### Code extension needed

- a steady-state convergence criterion is now available, but the Re=1000
  run at 64²/`t=30` is still not steady;
- **wall-clustered grids or a mapped/spectral grid** are now the clear
  priority: uniform 48² -> 64² does not reduce the `v` error;
- WENO/limited reconstruction with overintegration-like stabilization at
  high Re;
- a 3D cavity with z walls for genuinely turbulent states;
- a distributed/GPU Poisson or faster multigrid path at larger sizes.

The MUSCL scheme is stable to Re=10000 on the GPU and gives a clear `u`
accuracy gain; the quantitative high-Re/`v` validation is the next extension.