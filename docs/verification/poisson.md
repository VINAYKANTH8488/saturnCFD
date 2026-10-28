# Poisson solver: geometric multigrid with periodic and wall boundaries

**Date:** 2026-10-28
**Status:** single-rank, constant-coefficient; periodic, homogeneous-Neumann,
and homogeneous-Dirichlet boundaries; second-order convergence verified on
periodic and Neumann boxes. Variable coefficients, MPI, and wall-aware MAC
projection are next.
**Code:** `src/numerics/poisson.hpp`, `tests/test_poisson.cpp`

## Discretization

Cell-centered 7-point finite-volume Laplacian on a uniform box:

```
laplacian(p)_ijk =
    (p_{i+1} - 2 p_i + p_{i-1}) / hx^2
  + (p_{j+1} - 2 p_j + p_{j-1}) / hy^2
  + (p_{k+1} - 2 p_k + p_{k-1}) / hz^2
```

Each dimension can be `Periodic`, `Neumann`, or `Dirichlet`. Wall ghosts are

```
Neumann:   p_ghost =  p_interior   (zero normal derivative at the face)
Dirichlet: p_ghost = -p_interior   (zero value at the face)
```

For periodic/Neumann combinations the operator has a constant null space;
the right-hand side and the initial guess are projected to zero mean. With
Dirichlet boundaries no projection is performed.

## Solver

`saturn::poisson::GeometricMultigrid` builds a hierarchy by halving each
dimension while it is even and at least 6 cells. Each V-cycle uses

- red-black Gauss-Seidel smoothing (2 pre- and 2 post-sweeps),
- full-weighting restriction (27-point, weights 1/4-1/2-1/4),
- cell-centered trilinear prolongation,
- 60 red-black sweeps on the coarsest level.

The `solve` call iterates V-cycles until the relative residual reaches
`Options::tolerance` or `max_cycles` is exhausted. The interface is
matrix-free and device-callable; the level hierarchy is built on the host.

## Periodic manufactured-solution test

```
p = sin(2 pi x) sin(2 pi y) sin(2 pi z)
f = -3 (2 pi)^2 p
```

on `[0,1]^3`, tolerance `1e-10`:

| N | levels | cycles | initial residual | final residual | L2 error | order |
|---|---|---|---|---|---|---|
| 16 | 3 | 20 | 4.187e+01 | 2.113e-09 | 4.579e-03 | — |
| 32 | 4 | 32 | 4.187e+01 | 2.062e-09 | 1.138e-03 | 2.01 |
| 48 | 5 | 38 | 4.187e+01 | 3.332e-09 | 5.053e-04 | 2.00 |

## Neumann-wall test

```
p = cos(pi x) cos(pi y) cos(pi z)
f = -3 pi^2 p
```

on `[0,1]^3` with homogeneous Neumann conditions on all six faces (the cosine
has zero integral, so the compatibility condition is satisfied):

| N | levels | cycles | initial residual | final residual | L2 error | order |
|---|---|---|---|---|---|---|
| 16 | 3 | 23 | 1.047e+01 | 8.028e-10 | 1.138e-03 | — |
| 32 | 4 | 28 | 1.047e+01 | 6.947e-10 | 2.841e-04 | 2.00 |
| 48 | 5 | 31 | 1.047e+01 | 7.032e-10 | 1.262e-04 | 2.00 |

The observed orders are 2.00–2.01 in both cases, matching the 7-point
discretization. The residual reduction is about 0.35–0.4 per V-cycle; this is
adequate for the projection prototype but should improve with a better coarse
solver or MG-preconditioned CG.

## Mixed channel configuration

The channel-flow pressure solve uses periodic x/z and Neumann y walls. The
corresponding manufactured solution is

```
p = sin(2 pi x) cos(pi y) sin(2 pi z)
f = -9 pi^2 p
```

| N | levels | cycles | initial residual | final residual | L2 error | order |
|---|---|---|---|---|---|---|
| 16 | 3 | 21 | 3.140e+01 | 2.117e-09 | 4.193e-03 | — |
| 32 | 4 | 30 | 3.140e+01 | 3.013e-09 | 1.043e-03 | 2.01 |
| 48 | 5 | 35 | 3.140e+01 | 3.030e-09 | 4.631e-04 | 2.00 |

This configuration is exercised directly by the channel-flow work.

Run the tests with:

```bash
ctest --test-dir build/cpu-openmp -R saturn.poisson --output-on-failure
```

## Limitations and next steps

- Constant coefficients only; variable-density projection needs a
  variable-coefficient operator and averaging of face coefficients.
- Homogeneous Neumann/Dirichlet walls are implemented for the cell-centered
  pressure solve. The MAC velocity still assumes periodicity in every
  dimension; wall-aware divergence/gradient and no-slip momentum ghosts are
  the next step for channel and cavity cases.
- Single rank. Distributed boxes need a halo exchange inside the smoother and
  coarse-grid solve, plus a global dot-product reduction.
- No FFT backend yet; for very large periodic grids the FFT path should be
  compared against MG.
