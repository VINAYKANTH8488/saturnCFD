# Verification: 2nd-order scalar advection–diffusion

**Date:** 2026-10-26
**Status:** passing on Serial, OpenMP, and CUDA (Quadro P600); 1–8 MPI ranks
on CPU and 2 ranks on the GPU.
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

CUDA (P600) produces the same norms to displayed precision; the N=128 run took
0.85 s and also reported order 2.00 and mass drift −1.8e-15.

Serial, OpenMP, and CUDA produce the same norms to displayed precision.

## MPI verification

The same case was run on 1, 2, 4, and 8 OpenMPI ranks (8 ranks oversubscribed
on the 4-core host), with one Kokkos thread per rank. The per-rank Cartesian
decomposition uses face-only halo exchange; a separate halo unit test fills a
global linear index field and checks every face ghost cell (2 ghost layers).

| Ranks | N | L2 error | Order | Mass drift |
|---:|---:|---:|---:|---:|
| 1 | 32 | 2.343e-3 | — | 0 |
| 1 | 64 | 5.884e-4 | 1.99 | 0 |
| 2 | 32 | 2.343e-3 | — | −2.3e-16 |
| 2 | 64 | 5.884e-4 | 1.99 | −1.2e-16 |
| 4 | 32 | 2.343e-3 | — | +2.3e-16 |
| 4 | 64 | 5.884e-4 | 1.99 | +1.2e-16 |
| 8 | 16 | 9.195e-3 | — | +2.3e-16 |
| 8 | 32 | 2.343e-3 | 1.97 | +2.3e-16 |

Halo test (`tests/test_halo.cpp`): **0 errors** on 1, 2, 4, and 8 ranks; the
same test passes with 2 CUDA ranks (host-staged MPI).

The two-rank CUDA MMS also matches the CPU values (N=16/32, order 1.97).

## MPI halo design

- Cartesian process grid from `MPI_Dims_create` with a greedy fallback for thin
  grids; 2:1-style partitioning of global cells per rank.
- Face-only exchange (edges/corners are never read by the 7-point stencil).
- Persistent pack/unpack buffers; `MPI_Isend`/`Irecv` for both faces posted
  before `MPI_Waitall` to avoid cross-side deadlock.
- Send tag uses the sender's side; receive tag uses the opposite side, so the
  message pairs with the neighbor's high/low face correctly.
- GPU builds stage through host buffers; GPU-aware MPI is a later optimization.
- Global reductions (`MPI_Allreduce`) for L2, Linf, mass, and timing; only rank
  0 writes tables and JSON.

## Remaining MPI work

- Halo bandwidth and weak-scaling measurements (1–8 ranks are correct, not yet
  benchmarked).
- Non-periodic physical boundary conditions (face exchange currently skips
  `MPI_PROC_NULL` sides).
- GPU-aware MPI to remove host staging.

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
