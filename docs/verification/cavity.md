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

## GPU higher-Reynolds validation (Quadro P600)

The CUDA build selects the Kokkos Cuda backend with `--space cuda`. Ghia
centreline tables for Re=1000, 5000, and 10000 were added from the standard
PetIBM transcription of Ghia et al. (the Re=3200 column has a known typo at
`y=0.4531` and is not used).

`scripts/verify_cavity_gpu.sh` builds `saturn_cavity_sim` with the
`cuda-local` preset and runs:

| Re | grid | t | steps | RMS u | RMS v | `max|div|` | GPU time |
|---|---|---|---|---|---|---|---|
| 1000 | 32x32x1 | 40 | 3404 | **1.00e-1** | **9.67e-2** | `6.2e-8` | 192.6 s |
| 5000 | 32x32x1 | 10 | 729 | 2.88e-1 | 2.82e-1 | `1.1e-7` | 41.7 s |

The Re=1000 run passes `--check` with `--rms-tol 0.15`: the centreline RMS
error is ~10% of the lid speed on this coarse 32² grid, a first GPU
validation. The Re=5000 run is a stability smoke; it remains bounded and
mass-conserving but the 32² grid is far too coarse for a quantitative Ghia
comparison.

```bash
source scripts/env_cuda.sh
./scripts/verify_cavity_gpu.sh
```

### Code extension needed

The next step is not just a larger grid. The current explicit central
advection + explicit viscous step needs:
- a robust high-Re advection scheme (upwind/limiter or WENO) to avoid
  under-resolved central oscillations;
- finer grids (64²–128²) and longer transients, which the GPU makes
  affordable;
- a 3D cavity with z walls for genuinely turbulent states;
- a distributed/GPU Poisson or a faster multigrid path at larger sizes.

The Re=1000 GPU result is the first validated step; the Re=5000 run shows the
solver is stable in that direction, but the high-Re/turbulent validation
requires the extensions above.
