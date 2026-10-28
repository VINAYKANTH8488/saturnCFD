# Periodic Poisson solver: geometric multigrid

**Date:** 2026-10-28
**Status:** single-rank, constant-coefficient, periodic box; second-order
convergence verified. Variable coefficients, non-periodic boundaries, MPI, and
the projection coupling are next.
**Code:** `src/numerics/poisson.hpp`, `tests/test_poisson.cpp`

## Discretization

Cell-centered 7-point finite-volume Laplacian on a uniform periodic box:

```
laplacian(p)_ijk =
    (p_{i+1} - 2 p_i + p_{i-1}) / hx^2
  + (p_{j+1} - 2 p_j + p_{j-1}) / hy^2
  + (p_{k+1} - 2 p_k + p_{k-1}) / hz^2
```

The right-hand side must be zero-mean; both `b` and the initial guess are
projected onto the zero-mean subspace. The solution is therefore defined up to
a constant, as expected for a periodic Poisson problem.

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

## Manufactured-solution test

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

The observed order is 2.00–2.01, matching the 7-point discretization. The
residual reduction is about 0.35–0.4 per V-cycle; this is adequate for the
projection prototype but should improve with a better coarse solver or
MG-preconditioned CG.

Run the test with:

```bash
ctest --test-dir build/cpu-openmp -R saturn.poisson --output-on-failure
```

## Limitations and next steps

- Constant coefficients only; variable-density projection needs a
  variable-coefficient operator and averaging of face coefficients.
- Periodic boundaries only. Neumann/Dirichlet ghosts and the corresponding
  compatibility conditions are not implemented.
- Single rank. Distributed boxes need a halo exchange inside the smoother and
  coarse-grid solve, plus a global dot-product reduction.
- No FFT backend yet; for very large periodic grids the FFT path should be
  compared against MG.
- The Phase 3 projection step (Rhie–Chow face velocity, pressure correction,
  divergence-free projection) will use this solver.
