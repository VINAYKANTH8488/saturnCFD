# Phase 0 CPU microbenchmark results

**Date:** 2026-10-26
**Status:** CPU complete; CUDA pending.

## Environment

| Item | Value |
|---|---|
| CPU | Intel Core i7-7700, 4 cores / 8 threads, 3.6 GHz |
| Memory | 31 GiB DDR4 |
| GPU | NVIDIA Quadro P600, 2 GiB, sm_61 (functional tests only) |
| OS | Ubuntu 20.04 |
| Compiler | GCC 11.5.0 |
| CMake | 3.31.6 (user-local) |
| Kokkos | 5.2.2, Serial + OpenMP |
| Build | Release, `-O3` (CMake default for Release) |

## Configuration

- Stencil: 7-point 3D, FP64, 256³ grid, interior-only kernel.
- STREAM: 33,554,432 doubles (256 MiB per array), 30 iterations.
- OpenMP: 4 threads, `OMP_PROC_BIND=spread`, `OMP_PLACES=threads`.
- Three repetitions per configuration; median reported.

## Results

| Benchmark | Space | Threads | Median | Min | Max |
|---|---|---:|---:|---:|---:|
| stencil7 | serial | 1 | 2.66 ns/cell/iter | 2.589 | 2.697 |
| stencil7 | openmp | 4 | 0.985 ns/cell/iter | 0.950 | 0.993 |
| STREAM copy | serial | 1 | 18.84 GB/s | 18.22 | 19.44 |
| STREAM copy | openmp | 4 | 19.56 GB/s | 19.34 | 20.01 |
| STREAM triad | serial | 1 | 20.71 GB/s | 20.64 | 21.36 |
| STREAM triad | openmp | 4 | 21.98 GB/s | 21.73 | 22.38 |

### Stencil efficiency vs STREAM copy

| Space | Stencil GB/s | Copy GB/s | Ratio |
|---|---:|---:|---:|
| serial | 6.02 | 18.84 | 31.9% |
| openmp (4 threads) | 16.25 | 19.56 | 83.1% |

## Interpretation

1. **The machine saturates at ~19–22 GB/s.** Serial and 4-thread STREAM are
   nearly identical, so one core can already saturate this memory subsystem
   (single-channel or bandwidth-limited configuration).
2. **The 4-thread stencil reaches 83% of STREAM copy**, inside the 60–80%
   target range (slightly above it). This is a healthy starting point for a
   bandwidth-bound FVM kernel.
3. **Serial stencil is compute/latency-bound** at 32% of STREAM. The gap is the
   per-cell index arithmetic and neighbor loads, not memory bandwidth.
4. **Hyperthreading hurts this kernel.** 8 threads gave ~11.4 GB/s, worse than
   4 threads (~14.5–16 GB/s). Pin to physical cores for CPU runs.
5. **Correctness:** all execution spaces produced the same checksum
   (`26239360.1663...`), and `--verify` reported zero error on 32³.

## Caveats

- The P600 is a display GPU with 2 GiB and is not a meaningful scaling target.
  GPU numbers will be recorded on rented/HPCC A100/H100/MI250 hardware.
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
