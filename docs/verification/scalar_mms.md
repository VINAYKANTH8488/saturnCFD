# Verification: 2nd-order scalar advection–diffusion

**Date:** 2026-10-26
**Status:** passing on Serial and OpenMP (single rank).
**Code:** `apps/scalar_mms.cpp`

## Problem

Periodic 3D box `[0, 2π]³`, passive scalar:

```
∂c/∂t + u·∇c = ν ∇²c
u = (1, 0.5, 0.25),  ν = 0.01
```

Manufactured solution:

```
c(x,t) = 1 + 0.5 exp(−ν|k|²t) sin(k·x − ωt)
k = (1, 2, 1),  ω = k·u = 2.25,  |k|² = 6
```

The constant offset makes total mass non-zero, so mass conservation is a
meaningful check.

## Discretization

- Cell-centered finite volume on a uniform Cartesian grid.
- 2nd-order central advection and diffusion.
- SSP-RK3 with `CFL = 0.2` and a viscous step limit.
- Periodic ghost fill on the padded field (`ng = 1`).
- `dt = t_end / ceil(t_end / dt_stable)`, so every resolution reaches the same
  final time.

## Results

`N` is the number of cells per direction; `h = 2π/N`.

| N | h | dt | steps | L2 error | observed order | Linf error | mass drift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3.927e-1 | 4.000e-2 | 5 | 9.195e-3 | — | 1.30e-2 | 0.00e0 |
| 32 | 1.963e-1 | 2.222e-2 | 9 | 2.343e-3 | 1.97 | 3.30e-3 | 0.00e0 |
| 64 | 9.817e-2 | 1.111e-2 | 18 | 5.884e-4 | 1.99 | 8.32e-4 | 0.00e0 |
| 128 | 4.909e-2 | 5.556e-3 | 36 | 1.473e-4 | 2.00 | 2.08e-4 | −9.2e-16 |

Serial and OpenMP runs produce the same norms to displayed precision.

## Interpretation

- The observed spatial order is **2.00**, as designed. The temporal error is
  smaller than the spatial error at these `dt` values (RK3).
- Total mass is conserved to **machine precision** over the run.
- The field/ghost-fill/RK3 path is validated; the next step is to replace the
  single-rank periodic fill with MPI halo exchange and rerun the same MMS on
  1–8 ranks.

## Reproduce

```bash
source scripts/env.sh
export OMP_PROC_BIND=spread OMP_PLACES=threads

./build/cpu-openmp/apps/saturn_scalar_mms \
  --space openmp --n-list 16 32 64 128 \
  --nu 0.01 --tend 0.2 --check \
  --json benchmarks/results/scalar_mms_2nd_order.json
```

CTest:

```bash
ctest --test-dir build/cpu-openmp --output-on-failure
```
