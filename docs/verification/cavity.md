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
| `weno5` | upwind-biased WENO5 derivative (sign-dependent stencil) |
| `blend` | central/upwind blend, fraction set by `--blend` |

Ghia centreline RMS at Re=100, `n=16`, `nz=1`, `t=5`:

| Scheme | RMS u | RMS v |
|---|---|---|
| central | 3.09e-2 | 1.95e-2 |
| upwind | 2.60e-2 | 1.70e-2 |
| **muscl** | **1.67e-2** | **1.28e-2** |
| **weno5** | 1.31e-2 | 1.66e-2 |
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

MUSCL is the clear winner over central: at 48² the combined centreline RMS is
`0.050`, against `0.139` for central at 32². The `v` RMS is `4.69e-2`, below
the `5e-2` target. The earlier apparent MUSCL `v` failure was the shifted
table, not the scheme.

### WENO5

`--advection weno5` uses a sign-dependent upwind-biased WENO5 derivative:
the stencil is mirrored for negative advective velocity. At Re=1000:

| Scheme | grid | t | RMS u | RMS v | combined | time |
|---|---|---|---|---|---|---|
| MUSCL | 32² | 20 | 3.73e-2 | 6.37e-2 | 0.0738 | 79 s |
| **WENO5** | 32² | 20 | **3.54e-2** | **6.24e-2** | **0.0716** | 70 s |
| MUSCL | 48² | 20 | **1.74e-2** | 4.69e-2 | 0.0501 | 517 s |
| **WENO5** | 48² | 20 | 1.84e-2 | **4.41e-2** | **0.0478** | 498 s |

WENO5 is about 4% better in combined RMS at 48²: the `v` profile improves,
while the `u` profile is marginally worse. The gain is modest because the
32²/48² grids are still coarse; WENO5 is the higher-order path for finer or
clustered grids.

A low-pass/dealiasing filter is available (`--filter`, `--filter-every`). It
is applied before the RK stage so the projection restores divergence. At
Re=1000, WENO5 + `--filter 0.01` gave `4.52e-2 / 6.99e-2`, worse than no
filter; the smooth cavity does not need it. It remains an option for
high-Re/turbulent runs.

### 3D cavity with z walls

`--z-wall` replaces the periodic z direction with no-slip walls and a
Neumann pressure condition. The 3D cavity runs stably with WENO5; the
reported RMS is the centre plane against the 2D Ghia data, so it is a
qualitative comparison (the end walls change the flow):

| grid | t | RMS u | RMS v | `max|div|` | GPU time |
|---|---|---|---|---|---|---|
| 24x24x24 | 5 | 1.84e-1 | 2.07e-1 | 5.8e-7 | 17 s |
| 24x24x48 | 10 | 1.22e-1 | 1.61e-1 | 2.6e-7 | 132 s |
| 32x32x48 | 10 | 1.17e-1 | 1.41e-1 | 3.5e-7 | 157 s |

Larger span moves the centre plane closer to the 2D solution, as expected.
A genuine 3D reference (aspect-ratio-resolved cavity) is still needed for
quantitative 3D validation.

### Clustered grid

Wall clustering is **not yet implemented**. The current Poisson solver is a
constant-coefficient uniform-grid multigrid; a clustered grid needs:
- face-coordinate arrays and metric factors in `Geometry`;
- metric-aware divergence, momentum, and Laplacian;
- a variable-coefficient/curvilinear Poisson (PCG or MG).

That is the next solver-wide extension. It is not required for Re=1000 at
48² (the target is met), but it is the right path for Re≥5000 and turbulence.
The Nektar++/NekRS GLL-clustering analogy is the design guide.

### Higher-Re limits (n=48, t=10 unless noted)

| Re | scheme | RMS u | RMS v | `max|div|` | GPU time |
|---|---|---|---|---|---|---|
| 5000 | MUSCL | 2.60e-1 | 2.69e-1 | 9.5e-8 | 285 s |
| 5000 | WENO5 | 2.16e-1 | 2.89e-1 | 1.5e-8 | 80 s (32²) |
| 10000 | MUSCL | 3.24e-1 | 2.95e-1 | 1.9e-8 | 346 s |
| 10000 | **WENO5** | **2.49e-1** | 3.08e-1 | 5.0e-8 | 413 s |
| 10000 | **WENO5, 64²** | **2.53e-1** | 3.05e-1 | 7.1e-8 | 1115 s |

WENO5 improves the Re=10000 `u` profile by ~23% over MUSCL, but the `v`
profile remains poor and **uniform refinement from 48² to 64² does not
improve the result** (`u` 2.49e-1 -> 2.53e-1, `v` 3.08e-1 -> 3.05e-1). The
Re=10000 boundary layer is not resolved by a uniform grid at this size.
These are stability/trend results, **not a validation**. Wall clustering with
a variable-coefficient Poisson is required before Re=5000/10000 can be
claimed as validated.

### Scheme choice

| Regime | Recommended scheme | Reason |
|---|---|---|
| Re≤1000, uniform 32²–48² | WENO5 or MUSCL | WENO5 combined 0.048 vs MUSCL 0.050 at 48²; both meet the target |
| Re=5000 | WENO5 + finer grid | WENO5 32² gives 2.16e-1/2.89e-1; not yet validated |
| Re=10000 | WENO5 + wall clustering | WENO5 `u` 2.49e-1 vs MUSCL 3.24e-1; uniform refinement stalls |
| High-Re oscillations | WENO5 + optional low-pass filter | filter is a dealiasing analogue; no benefit on the smooth cavity |
| Robust/cheap fallback | MUSCL | lower cost, similar combined error at Re=1000 |

Central advection is only competitive at low Re/coarse grids; upwind is
over-damped. The next quantitative step is a wall-clustered grid with a
variable-coefficient Poisson, not a larger uniform grid.

## GPU higher-Reynolds validation (Quadro P600)

`scripts/verify_cavity_gpu.sh` builds the `cuda-local` preset and runs:

| Re | scheme | grid | t | RMS u | RMS v | `max|div|` | time |
|---|---|---|---|---|---|---|---|
| 1000 | **WENO5** | 32x32x1 | 20 | **3.54e-2** | **6.24e-2** | 1.4e-7 | 75 s |
| 5000 | **WENO5** | 32x32x1 | 10 | 2.16e-1 | 2.89e-1 | 1.5e-8 | 80 s |

The Re=1000 gate uses `--rms-tol 0.20 --div-tol 2e-7`; the script completes in
~3 minutes. Both runs use WENO5.

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
- a **clustered/curvilinear grid** with a variable-coefficient Poisson
  (the current uniform-grid MG cannot do this);
- WENO5 on the clustered grid plus dealiasing-like filtering for high Re;
- a quantitative 3D cavity reference (the z-wall runs are qualitative);
- a faster distributed/GPU Poisson at larger sizes.