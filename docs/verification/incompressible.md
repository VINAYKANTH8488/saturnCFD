# Incompressible projection solver: 2D Taylor–Green vortex

**Date:** 2026-10-28
**Status:** single-rank, periodic, staggered MAC prototype; second-order
convergence on the exact 2D Taylor–Green vortex and the 3D ABC (Beltrami)
flow. Variable density, physical boundaries, MPI, and the 3D TGV/channel/
cavity cases are next.
**Code:** `apps/incompressible_sim.cpp`, `src/numerics/incompressible.hpp`,
`src/numerics/projection.hpp`, `src/numerics/poisson.hpp`

## Formulation

Velocities live on faces of a staggered MAC grid and pressure on cell centers:

```
u(i,j,k): x-face at x_i, cell center y_j/z_k
v(i,j,k): y-face at y_j, cell center x_i/z_k
w(i,j,k): z-face at z_k, cell center x_i/y_j
```

The discrete divergence and pressure gradient are exact adjoints, so
`div(grad(p))` is the same 7-point Laplacian solved by the geometric-multigrid
solver in `src/numerics/poisson.hpp`. The staggered arrangement avoids the
pressure checkerboard mode, so Rhie–Chow interpolation is not needed on this
path.

Momentum uses second-order central advection and diffusion. SSP-RK3 advances
the three face velocities and the pressure projection is applied after every
RK stage:

```
laplacian(phi) = div(u*),   u = u* - grad(phi)
```

The projection operator is linear and independent of `dt`; the application
uses a unit scale internally. Projecting every stage (rather than only at the
end of the step) removed the first-order splitting error and restored the
expected second-order convergence.

## Exact test case

The two-dimensional Taylor–Green vortex is an exact solution of the
incompressible Navier–Stokes equations:

```
u =  sin(x) cos(y) exp(-2 nu t)
v = -cos(x) sin(y) exp(-2 nu t)
w = 0
p = (cos(2x) + cos(2y)) exp(-4 nu t) / 4 + const
```

on `[0,2pi]^2`, extended periodically in z with `nz = 8`. `nu = 0.05`,
`t_end = 0.5`, CFL 0.3. The initial field is projected once so that the
discrete divergence is zero.

## Results

OpenMP, 8 threads:

| N | steps | dt | L2 velocity | max abs(div) | order |
|---|---|---|---|---|---|
| 16 | 5 | 1.22e-02 | 7.218e-03 | 5.3e-13 | — |
| 32 | 9 | 1.65e-02 | 1.808e-03 | 6.4e-13 | 2.00 |
| 64 | 17 | 1.75e-02 | 4.522e-04 | 2.4e-12 | 2.00 |

The velocity error decreases at exactly second order and the projected
divergence is at solver/round-off level. The same errors are obtained at CFL
0.1, confirming that the time integration is not the limiting error at CFL 0.3.

## 3D ABC flow

The ABC (Beltrami) flow

```
u = sin(z) + cos(y)
v = sin(x) + cos(z)
w = sin(y) + cos(x)
```

has `curl(u) = u`, so `(u.grad)u = grad(|u|^2/2)` and `laplacian(u) = -u`. It
is therefore an exact decaying solution of the 3D incompressible
Navier–Stokes equations,

```
u(x,t) = u(x,0) exp(-nu t),   p = -|u|^2 exp(-2 nu t)/2 + const.
```

This exercises all three momentum components, the 3D pressure solve, and the
full nonlinear term. With `nu=0.1`, `t_end=0.5`, CFL 0.3, cubic grids:

| N | steps | L2 velocity | max abs(div) | order |
|---|---|---|---|---|
| 16 | 11 | 6.083e-04 | 1.7e-12 | — |
| 32 | 21 | 1.526e-04 | 2.2e-12 | 1.99 |

The 3D case converges at second order with divergence at round-off.

## 3D Taylor–Green vortex status

`--case tgv3d` implements the standard 3D Taylor–Green initial condition on a
cubic periodic grid (initial `K/V=0.125`). A short run at `n=16`, `t=1`,
`nu=1/1600` is stable and is covered by
`saturn.incompressible_tgv3d_smoke`.

Longer runs at `n=32` become unstable after transition, for both `Re=1600`
(`nu=1/1600`) and `Re=400` (`nu=1/400`): the central advection operator has no
grid-scale dissipation and the under-resolved 3D transition aliases energy into
the smallest modes. The solver now detects non-finite states and aborts instead
of continuing with a collapsed time step; `--check` fails in that case. A
dealiased/energy-conserving or upwind/high-order advection scheme, or a
production resolution (`n >= 64` with adequate dissipation), is required before
this case can be used as a quantitative TGV benchmark. The 2D TGV and 3D ABC
results above remain valid because those flows stay smooth.

Run the gates with:

```bash
./build/cpu-openmp/apps/saturn_incompressible_sim \
  --n-list 16 32 64 --tend 0.5 --nu 0.05 --cfl 0.3 --check

./build/cpu-openmp/apps/saturn_incompressible_sim \
  --case abc --n-list 16 32 --tend 0.5 --nu 0.1 --cfl 0.3 --check

ctest --test-dir build/cpu-openmp -R saturn.incompressible --output-on-failure
```

## Limitations and next steps

- Single rank; the multigrid solver and the MAC exchange are not distributed.
- Periodic boundaries only.
- Constant density/viscosity; variable-density projection and variable-
  coefficient Poisson operators are not implemented.
- No Rhie–Chow path: the MAC grid is used for this prototype. A collocated
  path can be added later if the production data layout requires it.
- The 3D TGV is implemented but the central advection scheme is unstable after
  transition at `n=32` (both `Re=400` and `Re=1600`); it needs a
  dealiased/energy-conserving scheme or a higher resolution.
- The 3D TGV, turbulent channel, and cavity cases from Phase 3 remain.
- Poisson solves currently start from zero each stage; warm starting and a
  faster MG/FFT path are needed for production runs.
