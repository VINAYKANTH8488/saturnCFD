# GPU baseline: Quadro P600 (CUDA 12.2, sm_61)

**Date:** 2026-10-29
**Hardware:** NVIDIA Quadro P600, 2 GB, compute capability 6.1
**Driver:** 580.178.04
**Toolchain:** CUDA 12.2.140, GCC 12.4 host compiler, Kokkos 5.2.2 (CUDA +
Serial + OpenMP)
**Code:** `apps/stencil_bench.cpp`, `apps/stream_bench.cpp`

## Stencil (7-point FP64, 256³, 50 iterations)

Median of three runs:

| Metric | Value |
|---|---|
| effective bandwidth | **27.93 GB/s** |
| naive bandwidth (8 accesses) | 111.7 GB/s |
| throughput | 1.746 Gcell-updates/s |
| time per cell-update | 0.573 ns |
| max abs error (verify) | 0 |

Raw data: `benchmarks/results/gpu/stencil_cuda_p600_256_run{1,2,3}.json`.

## STREAM

16,777,216 elements (FP64), 50 iterations, median of three:

| Kernel | bytes/element | bandwidth | ns/element |
|---|---|---|---|
| copy | 16 | **54.24 GB/s** | 0.295 |
| triad | 24 | **54.31 GB/s** | 0.442 |

Raw data: `benchmarks/results/gpu/stream_{copy,triad}_cuda_p600_run{1,2,3}.json`.

The stencil reaches **51.5%** of the STREAM copy bandwidth, which is the
expected range for a 7-point stencil with two arrays on this class of GPU.

## Correctness status on CUDA

| Test | CUDA status |
|---|---|
| stencil verification | passes (`max_abs_error=0`) |
| scalar MMS, 16/32 | passes, order 1.97 |
| 1D Sod shock tube | passes (L1 rho 2.12e-3 at 200 cells) |
| incompressible 2D TGV | passes, same 2.00 order and errors as CPU |
| Poisson MMS / projection | passes after fixing a device-capture bug |
| 3D compressible diagnostics | **known issue**: the energy/velocity diagnostics and MMS error norm report stale/zero values on CUDA even though the 1D Sod path passes; needs a dedicated debugging pass |
| 3D compressible MMS | reports zero L2 on CUDA; not yet validated |

The 3D compressible issue does not affect the 1D Euler path or the
incompressible/scalar kernels. It must be resolved before claiming Phase 2 GPU
validation.

## Notes

- CUDA 12.2 does not support GCC 13; the local GCC 12 tree under
  `~/opt/gcc-12` is used as `NVCC_WRAPPER_DEFAULT_COMPILER`.
- `scripts/env_cuda.sh` sets the host compiler, CUDA library path, and Kokkos
  environment for `cmake --preset cuda-local`.
- The P600 has 2 GB, so large 3D cases (e.g. 256³ compressible) need care; the
  stencil/STREAM tests above fit comfortably.
