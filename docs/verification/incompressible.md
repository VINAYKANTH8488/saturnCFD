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

## 3D Taylor–Green vortex

`--case tgv3d` implements the standard 3D Taylor–Green initial condition on a
cubic periodic grid. The unfiltered central scheme is unstable after transition
at `n=32` for both `Re=1600` (`nu=1/1600`) and `Re=400` (`nu=1/400`): the
non-dissipative advection aliases energy into grid-scale modes. Non-finite
states are detected and the run aborts (the short `n=16`, `t=1` smoke test is
covered by `saturn.incompressible_tgv3d_smoke`).

A fourth-order periodic filter,

```
u <- u - alpha/16 (u_{i-2} - 4u_{i-1} + 6u_i - 4u_{i+1} + u_{i+2})
```

with symbol `1 - alpha sin^4(kh/2)`, stabilizes the `n=32` run. With
`--filter 0.1`, `Re=1600`, `t=10`:

| Quantity | Value |
|---|---|
| initial `K/V` | 0.1238 |
| final `K/V` | 0.0595 |
| peak `-dK/dt / V` | **1.520e-02** |
| peak time | 5.91 |
| steps | 174 |
| max abs(div) | 1.3e-15 |
| wall time | 20 s |

The filtered peak is larger and earlier than the low-Mach reference
(`~1.25e-2` at `t~9`); the filter adds numerical dissipation and the 32³ run is
under-resolved, so this is a stable transitional-flow exercise rather than a
quantitative DNS benchmark. A weaker filter (`alpha=0.02`) is already unstable.
Raw history and dissipation data are in
`benchmarks/results/tgv/tgv3d_n32_filter01_*`.

With the FFT pressure solver (`--poisson fft`) the run is fast enough for
`n=64`. At `n=64`, `Re=1600`, `filter=0.1`, `t=10` (368 steps, 34.6 s):

| Quantity | Value |
|---|---|
| initial `K/V` | 0.1247 |
| final `K/V` | 0.0694 |
| peak `-dK/dt / V` | **1.142e-02** |
| peak time | **8.96** |
| max abs(div) | 2.2e-15 |

The peak is 8.6% below the low-Mach reference (`~1.25e-2` near `t≈9`) and the
peak time now matches; the remaining difference is consistent with the filter
and the 64³ resolution. A weaker filter (`alpha=0.02`) is still unstable at
`n=64`. Data: `benchmarks/results/tgv/tgv3d_n64_filter01_*`.

Run the 3D cases with:

```bash
./build/cpu-openmp/apps/saturn_incompressible_sim \
  --case tgv3d --n 16 --tend 1 --nu 0.000625 --cfl 0.3 --check
./build/cpu-openmp/apps/saturn_incompressible_sim \
  --case tgv3d --n 64 --tend 10 --nu 0.000625 --cfl 0.3 \
  --filter 0.1 --poisson fft --diag-every 10 --history tgv3d.csv --check
```

The 2D TGV and 3D ABC results above remain valid because those flows stay
smooth.

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
- The 3D TGV is implemented but the unfiltered central advection is unstable
  after transition at `n=32`; a 4th-order filter with `alpha >= 0.1`
  stabilizes it (at the cost of added dissipation). A dealiased/upwind scheme
  and higher resolution are needed for a quantitative benchmark.
- Turbulent channel Re_tau=180/395 and cavity at higher Re remain.
- The pressure field is warm-started across projections; a periodic FFT path is
  available (`--poisson fft`), while wall-bounded cases still use multigrid. A
  distributed solve is still needed for production runs.
