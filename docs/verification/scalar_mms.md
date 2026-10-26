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

- Overlap halo exchange with interior computation (currently serialized).
- GPU-aware MPI to remove host staging.
- Non-periodic physical boundary conditions (face exchange currently skips
  `MPI_PROC_NULL` sides).
- Weak scaling at fixed cells per rank.

## Halo benchmark

`apps/halo_bench.cpp`, periodic cube, `ng = 2`, one Kokkos thread per rank,
median of 20 exchanges (n=128) or 10 exchanges (n=256):

| Grid | Ranks | Dims | Local box | µs/exchange | GB/s | ns/cell |
|---:|---:|---|---|---:|---:|---:|
| 128³ | 1 | 1×1×1 | 128³ | 1428 | 0 | 0.681 |
| 128³ | 2 | 2×1×1 | 64×128×128 | 1151 | 1.82 | 0.549 |
| 128³ | 4 | 2×2×1 | 64×64×128 | 1419 | 2.96 | 0.677 |
| 128³ | 8 | 2×2×2 | 64³ | 1454 | 4.33 | 0.693 |
| 256³ | 1 | 1×1×1 | 256³ | 8097 | 0 | 0.483 |
| 256³ | 2 | 2×1×1 | 128×256×256 | 6808 | 1.23 | 0.406 |
| 256³ | 4 | 2×2×1 | 128×128×256 | 5850 | 2.87 | 0.349 |
| 256³ | 8 | 2×2×2 | 128³ | 6458 | 3.90 | 0.385 |

At these sizes the exchange is **latency-bound**, not bandwidth-bound: adding
ranks increases aggregate bytes but not throughput. For 256³ the cost is
0.35–0.48 ns/cell versus 0.97 ns/cell for the CPU stencil, i.e. a 35–50%
overhead if serialized. The fix is to **overlap** the exchange with the
interior stencil (planned) and, on GPUs, use GPU-aware MPI instead of host
staging. The surface-to-volume ratio improves at larger grids, so the fraction
falls with problem size.

On the P600 (128³, host-staged), one rank takes **418 µs** (0.20 ns/cell) and
two ranks **1426 µs** (0.68 ns/cell). Host staging dominates the multi-rank GPU
path; GPU-aware MPI is a priority for the scaling milestone.

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
