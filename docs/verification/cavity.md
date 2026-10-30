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
| `muscl` | second-order minmod-limited MUSCL |
| `blend` | central/upwind blend, fraction set by `--blend` |

Ghia centreline RMS at Re=100, `n=16`, `nz=1`, `t=5`:

| Scheme | RMS u | RMS v |
|---|---|---|
| central | 3.09e-2 | 1.95e-2 |
| upwind | 2.60e-2 | 1.70e-2 |
| **muscl** | **1.67e-2** | **1.28e-2** |
| blend 0.25 | 2.93e-2 | 1.85e-2 |

> **Table correction.** An earlier revision of this page used shifted Ghia
> `v` columns for Re=1000/5000/10000 (the PetIBM table stores `v100` before
> `v1000`). That produced spurious `v` RMS values (e.g. MUSCL 1.96e-1,
> upwind 2.07e-2). The tables are fixed above; all `v` numbers below use the
> corrected columns. The `u` values were unaffected.

At Re=1000 on the P600:

| Scheme | grid | t | RMS u | RMS v | `max|div|` | time |
|---|---|---|---|---|---|---|
| central | 32x32x1 | 40 | 1.00e-1 | 9.67e-2 | 6.2e-8 | 193 s |
| upwind | 32x32x1 | 40 | 1.43e-1 | 1.88e-1 | 3.1e-7 | 164 s |
| blend 0.25 | 32x32x1 | 40 | 1.22e-1 | 1.39e-1 | 9.2e-8 | 170 s |
| **MUSCL** | 32x32x1 | 20 | **3.73e-2** | **6.37e-2** | 1.0e-7 | 79 s |
| **MUSCL** | 48x48x1 | 20 | **1.74e-2** | **4.69e-2** | 1.0e-7 | 517 s |

MUSCL is the clear winner: at 48² the combined centreline RMS is `0.050`,
against `0.139` for central at 32². The `v` RMS is `4.69e-2`, below the
`5e-2` target. The earlier apparent MUSCL `v` failure was the shifted table,
not the scheme. Uniform refinement plus MUSCL is working; wall clustering is
not required at Re=1000.

### Higher-Re limits (MUSCL, n=48, t=10)

| Re | RMS u | RMS v | `max|div|` | GPU time |
|---|---|---|---|---|---|
| 5000 | 2.60e-1 | 2.69e-1 | 9.5e-8 | 285 s |
| 10000 | 3.24e-1 | 2.95e-1 | 1.9e-8 | 346 s |

Both are stable but under-resolved at 48²; these are stability-limit results,
not validated high-Re accuracy. Wall clustering/spectral resolution is still
the likely requirement at these Reynolds numbers.

## GPU higher-Reynolds validation (Quadro P600)

`scripts/verify_cavity_gpu.sh` builds the `cuda-local` preset and runs:

| Re | scheme | grid | t | RMS u | RMS v | `max|div|` | time |
|---|---|---|---|---|---|---|---|
| 1000 | MUSCL | 32x32x1 | 20 | **3.73e-2** | **6.37e-2** | 1.0e-7 | 79 s |
| 5000 | MUSCL | 32x32x1 | 10 | 2.58e-1 | 2.83e-1 | 3.5e-8 | 89 s |

The Re=1000 gate uses `--rms-tol 0.20 --div-tol 2e-7`; the script completes in
~3 minutes.

```bash
source scripts/env_cuda.sh
./scripts/verify_cavity_gpu.sh
```

## Steady-state check and 64²

A steady-state criterion is available (`--steady-tol`, `--steady-check`,
`--steady-window`). At 64², Re=1000, MUSCL, `t=30`, the `u` RMS is
`1.72e-2` (consistent with 48²), but the run is **not steady**
(residual `7.3e-2`) after 2042 s. The `v` number from that run predates the
table correction and is not used. A corrected 64² run to steady state is the
next check; the corrected 48² result already meets the `v` target.

## Higher-order method context (Nektar++ / NekRS)

Nektar++ uses a spectral/hp element formulation: polynomial order `p` on
Gauss–Lobatto–Legendre points, separate advection and diffusion operators,
dealiasing/overintegration for nonlinear terms. NekRS uses spectral elements
with order `N` (typically 7–9), dealiasing/filtering, and overintegration.

Two lessons:
- **GLL/spectral nodes cluster at walls**, which resolves thin boundary
  layers without uniform refinement.
- **Nonlinear advection needs dealiasing/overintegration**; limiters are the
  finite-volume analogue.

Our MUSCL is second-order FV, not spectral. It already meets the Re=1000
centreline target at 48²; Re=5000/10000 and genuinely turbulent 3D states
will need the wall-clustered/spectral and dealiasing ideas.

### Code extension needed

- a corrected 64² steady-state run;
- wall-clustered or spectral/hp grids for Re=5000/10000;
- WENO/limited reconstruction with dealiasing-like stabilization;
- a 3D cavity with z walls for turbulence;
- a faster distributed/GPU Poisson at larger sizes.