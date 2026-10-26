# Phase 0 microbenchmark results

**Date:** 2026-10-26
**Status:** CPU and local GPU (Quadro P600) complete.

## Environment

| Item | Value |
|---|---|
| CPU | Intel Core i7-7700, 4 cores / 8 threads, 3.6 GHz |
| Memory | 31 GiB DDR4 |
| GPU | NVIDIA Quadro P600, 2 GiB, sm_61 (functional/small-scale only) |
| OS | Ubuntu 20.04 |
| Compiler | GCC 11.5.0, CUDA 12.2 |
| CMake | 3.31.6 (user-local) |
| Kokkos (CPU) | 5.2.2, Serial + OpenMP |
| Kokkos (GPU) | 5.2.2, Serial + CUDA, `PASCAL61` |
| Build | Release, `-O3` |

## Configuration

- Stencil: 7-point 3D, FP64, 256³ grid, interior-only kernel.
- STREAM: 33,554,432 doubles (256 MiB per array), 30 iterations.
- OpenMP: 4 threads, `OMP_PROC_BIND=spread`, `OMP_PLACES=threads`.
- Three repetitions per configuration; median reported.

## Results

| Benchmark | Space | Threads | Median | Min | Max |
|---|---|---:|---:|---:|---:|
| stencil7 | serial | 1 | 2.66 ns/cell/iter | 2.589 | 2.697 |
| stencil7 | openmp | 4 | 0.972 ns/cell/iter | 0.950 | 0.993 |
| stencil7 | cuda | 6144 | 0.565 ns/cell/iter | 0.562 | 0.566 |
| STREAM copy | serial | 1 | 18.84 GB/s | 18.22 | 19.44 |
| STREAM copy | openmp | 4 | 19.56 GB/s | 19.34 | 20.01 |
| STREAM copy | cuda | 6144 | 54.68 GB/s | 54.00 | 54.71 |
| STREAM triad | serial | 1 | 20.71 GB/s | 20.64 | 21.36 |
| STREAM triad | openmp | 4 | 21.98 GB/s | 21.73 | 22.38 |
| STREAM triad | cuda | 6144 | 55.01 GB/s | 55.00 | 55.02 |

### Larger grid (512³, single run)

| Benchmark | Space | Threads | Median | Effective GB/s |
|---|---|---:|---:|---:|
| stencil7 | serial | 1 | 2.631 ns/cell/iter | 6.08 |
| stencil7 | openmp | 4 | 1.275 ns/cell/iter | 12.55 |

The 512³ grid is 8× larger and no longer fits in any cache. OpenMP throughput
drops from 0.972 to 1.275 ns/cell/iter (12.55 GB/s, ~64% of STREAM copy),
consistent with additional DRAM/TLB pressure. Serial throughput is unchanged,
confirming that the single-thread kernel is compute/latency-bound rather than
bandwidth-bound.

### GPU (Quadro P600, 256³)

| Benchmark | Space | Median | Effective GB/s |
|---|---|---:|---:|
| stencil7 | cuda | 0.565 ns/cell/iter | 28.3 |
| STREAM copy | cuda | 54.68 GB/s | — |
| STREAM triad | cuda | 55.01 GB/s | — |

The GPU stencil reaches **~52% of STREAM copy**. It is a 2 GiB, 384-core
display GPU with heavily reduced FP64 throughput, so this is a functional
baseline, not a scaling result. A crucial finding: Kokkos's **default
MDRangePolicy tile gave only ~5 GB/s** on this GPU; an explicit inner tile of
32 contiguous cells raised it to ~28 GB/s. The solver now uses a backend-aware
tile (Kokkos default on host, 32 on device).

### Stencil efficiency vs STREAM copy

| Space | Stencil GB/s | Copy GB/s | Ratio |
|---|---:|---:|---:|
| serial | 6.02 | 18.84 | 31.9% |
| openmp (4 threads) | 16.47 | 19.56 | 84.2% |
| cuda (P600) | 28.32 | 54.68 | 51.8% |

## MPI halo exchange

`apps/halo_bench.cpp`, periodic cube, `ng=2`, one Kokkos thread per rank
(OpenMPI, `--oversubscribe` for 8 ranks on the 4-core host):

| Grid | Ranks | µs/exchange | Aggregate GB/s | ns/cell |
|---:|---:|---:|---:|---:|
| 128³ | 1 | 1428 | 0 | 0.681 |
| 128³ | 2 | 1151 | 1.82 | 0.549 |
| 128³ | 4 | 1419 | 2.96 | 0.677 |
| 128³ | 8 | 1454 | 4.33 | 0.693 |
| 256³ | 1 | 8097 | 0 | 0.483 |
| 256³ | 2 | 6808 | 1.23 | 0.406 |
| 256³ | 4 | 5850 | 2.87 | 0.349 |
| 256³ | 8 | 6458 | 3.90 | 0.385 |

The exchange is latency-bound at these sizes; serialized it costs 35–50% of a
256³ CPU stencil step. On the P600 (128³), one rank takes 418 µs and two ranks
1426 µs with host staging. Overlap and GPU-aware MPI are the next
optimizations.

## Interpretation

1. **The CPU saturates at ~19–22 GB/s.** Serial and 4-thread STREAM are nearly
   identical, so one core can already saturate this memory subsystem
   (single-channel or bandwidth-limited configuration).
2. **The 4-thread stencil reaches 84% of STREAM copy**, at the top of the
   60–80% target range. The GPU reaches 52%, consistent with its reduced FP64
   throughput and small L2.
3. **Serial stencil is compute/latency-bound** at 32% of STREAM. The gap is the
   per-cell index arithmetic and neighbor loads, not memory bandwidth.
4. **Hyperthreading hurts this kernel.** 8 threads gave ~11.4 GB/s, worse than
   4 threads (~14.5–16 GB/s). Pin to physical cores for CPU runs.
5. **Tile size matters on GPU.** The Kokkos default MDRange tile produced
   ~5 GB/s; `{1,1,32}` produced ~28 GB/s. This is now the default device tile.
6. **Correctness:** all execution spaces produced the same checksum
   (`26239360.1663...`), and `--verify` reported zero error on 32³. The scalar
   MMS also matches on Serial, OpenMP, and CUDA.

## Caveats

- The P600 is a display GPU with 2 GiB and heavily reduced FP64 throughput. It
  is useful for functional validation and kernel-shape experiments, not for
  scaling claims. Scaling runs will use A100/H100/MI250 hardware.
- These numbers calibrate the stencil roofline only. They do not include halo
  exchange, reconstruction, Riemann solves, or the pressure coupling.
- The "naive" bandwidth (8 accesses/cell) is reported by the tool but is not a
  DRAM measurement; use the effective bandwidth and STREAM ratio.

## Reproduce

```bash
source scripts/env.sh
export OMP_PROC_BIND=spread OMP_PLACES=threads

./build/cpu-openmp/apps/saturn_stencil \
  --space openmp --nx 256 --ny 256 --nz 256 --iters 50 \
  --kokkos-num-threads=4 \
  --json benchmarks/results/runs/stencil_openmp4_256_run1.json

python3 benchmarks/scripts/summarize.py \
  'benchmarks/results/runs/*.json' \
  --markdown benchmarks/results/summary.md
```
