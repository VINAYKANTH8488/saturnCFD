# Incompressible projection solver: 2D Taylor–Green vortex

**Date:** 2026-10-28
**Status:** single-rank, periodic, staggered MAC prototype; second-order
convergence on the exact 2D Taylor–Green solution. Variable density, physical
boundaries, MPI, and the 3D TGV/channel/cavity cases are next.
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

Run the gate with:

```bash
./build/cpu-openmp/apps/saturn_incompressible_sim \
  --n-list 16 32 64 --tend 0.5 --nu 0.05 --cfl 0.3 --check

ctest --test-dir build/cpu-openmp -R saturn.incompressible --output-on-failure
```

## Limitations and next steps

- Single rank; the multigrid solver and the MAC exchange are not distributed.
- Periodic boundaries only.
- Constant density/viscosity; variable-density projection and variable-
  coefficient Poisson operators are not implemented.
- No Rhie–Chow path: the MAC grid is used for this prototype. A collocated
  path can be added later if the production data layout requires it.
- The 3D TGV, channel flow, and cavity cases from Phase 3 remain.
- Poisson solves currently start from zero each stage; warm starting and a
  faster MG/FFT path are needed for production runs.
