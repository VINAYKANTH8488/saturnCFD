# Wall-bounded channel flows

**Date:** 2026-10-28
**Status:** single-rank, periodic x/z, no-slip y walls; Couette and plane
Poiseuille verification at second order. Turbulent channel flow (Re_tau
180/395) is not yet run.
**Code:** `apps/channel_sim.cpp`, `src/numerics/channel.hpp`,
`src/numerics/poisson.hpp`

## Setup

The channel occupies `[0, 2pi] x [0, Ly] x [0, 2pi]` with periodic x and z and
no-slip walls at `y=0` and `y=Ly`. The staggered MAC layout is

```
u(i,j,k): x-face, y/z at cell centers
v(i,j,k): y-face; v(0) is the bottom wall, v(n1) (implicit) the top wall
w(i,j,k): z-face, x/y at cell centers
```

Boundary conditions:

- `u = 0` at the bottom, `u = u_top` at the top (Couette); `w = 0` at both;
- no penetration `v = 0` on both wall faces;
- pressure: homogeneous Neumann at both walls (`dp/dy = 0`);
- x/z: periodic.

The pressure Poisson solve uses the mixed periodic/Neumann configuration
verified in `docs/verification/poisson.md`. Momentum uses second-order central
advection/diffusion and SSP-RK3 with a projection after every stage.

## Couette flow

```
u(y) = u_top y / Ly,  v = w = 0
```

This is an exact steady solution of the discrete linear operator as well: the
linear profile has zero second derivative, the wall ghosts are exact
(`u_ghost = 2 u_wall - u_interior`), and there is no body force. With
`u_top=1`, `Ly=1`, `nu=0.1`, `t_end=0.1`:

| NY | steps | L2 error | max abs(div) | center abs(u) |
|---|---|---|---|---|
| 16 | 13 | 2.30e-17 | 0 | 0.53125 |
| 32 | 52 | 1.80e-17 | 0 | 0.515625 |

The solution is steady to round-off, confirming the moving-wall ghost, the
no-penetration faces, and the wall-aware pressure projection.

## Plane Poiseuille flow

```
u(y) = gx y (Ly - y) / (2 nu),  v = w = 0
```

driven by a constant streamwise body force `gx=1`, `nu=0.1`. The initial state
is the analytic profile; the run to `t=0.1` checks that the body force,
viscous term, and wall closure are consistent:

| NY | steps | L2 error | order | max abs(div) | center abs(u) |
|---|---|---|---|---|---|
| 16 | 13 | 9.955e-04 | — | 0 | 1.2451 |
| 32 | 52 | 2.545e-04 | 1.97 | 0 | 1.2488 |
| 64 | 205 | 6.395e-05 | 1.99 | 0 | 1.2497 |

The exact centerline velocity is `gx Ly^2 / (8 nu) = 1.25`, approached as the
grid is refined. The L2 error decreases at second order and the projected
divergence is zero to round-off.

## Statistics

`--stats FILE` writes plane-averaged profiles after the run:

```
y,yplus,uplus,urms,vrms,wrms,uv
```

The wall friction velocity is estimated from the no-slip gradient,
`u_tau = sqrt(nu |du/dy|_wall)`, and `Re_tau = u_tau (Ly/2)/nu`. The driver
prints both values. For the laminar Poiseuille case above (`nu=0.1`, `gx=1`),
`u_tau = 0.7066` versus the exact `sqrt(gx Ly/2) = 0.7071` (0.07% error), and
the first-cell `u+` equals `y+` as expected in the viscous sublayer.

A turbulent channel attempt uses a longer domain and initial perturbations:

```bash
./build/cpu-openmp/apps/saturn_channel_sim \
  --case poiseuille --n-list 64 --nx 32 --nz 16 \
  --lx 12.5663706144 --lz 4.1887902048 --ly 2 \
  --nu 0.0055555556 --gx 1 --perturb 0.1 --tend 10 \
  --stats channel_turb.csv
```

At `nu=1/180`, `Ly=2`, `gx=1` the target friction Reynolds number is 180.
The uniform 64-cell wall-normal grid gives `Delta y+ ~ 5.6`, which is far
coarser than a DNS wall layer; this run is an infrastructure demo, not a
validated turbulent-channel solution. Statistics for this case are reported
separately when available.

## Commands

```bash
./build/cpu-openmp/apps/saturn_channel_sim \
  --case couette --n-list 16 32 --tend 0.1 --nu 0.1 --u-top 1 --check

./build/cpu-openmp/apps/saturn_channel_sim \
  --case poiseuille --n-list 16 32 64 --tend 0.1 --nu 0.1 --gx 1 --check

ctest --test-dir build/cpu-openmp -R saturn.channel --output-on-failure
```

## Limitations and next steps

- Single rank; distributed channel flow needs the distributed Poisson solver
  and halo exchange.
- Laminar verification only. The turbulent channel at Re_tau=180/395 needs a
  longer domain, a driving pressure gradient or constant mass flux, and
  statistics collection.
- The walls are restricted to the y direction; x/z walls (cavity) and general
  wall normals are not implemented.
- The time step is explicit and diffusion-limited; an implicit viscous term or
  local time stepping is needed for high-Re production runs.
