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

Raw data: `benchmarks/results/runs/stencil_cuda_p600_256_run{1,2,3}.json`.

## STREAM

16,777,216 elements (FP64), 50 iterations, median of three:

| Kernel | bytes/element | bandwidth | ns/element |
|---|---|---|---|
| copy | 16 | **54.24 GB/s** | 0.295 |
| triad | 24 | **54.31 GB/s** | 0.442 |

Raw data: `benchmarks/results/runs/stream_{copy,triad}_cuda_p600_run{1,2,3}.json`.

The stencil reaches **51.5%** of the STREAM copy bandwidth, which is the
expected range for a 7-point stencil with two arrays on this class of GPU.
A tile sweep (`--tile 16/32/64`) gives 0.853 / 0.573 / 0.669 ns/cell/iter, so
the existing inner tile of 32 is the best of those; larger tiles are invalid
for this grid. Two-dimensional tiles and kernel fusion remain future work.

## Correctness status on CUDA

| Test | CUDA status |
|---|---|
| stencil verification | passes (`max_abs_error=0`) |
| scalar MMS, 16/32 | passes, order 1.97 |
| 1D Sod shock tube | passes (L1 rho 2.12e-3 at 200 cells) |
| incompressible 2D TGV | passes, same 2.00 order and errors as CPU |
| Poisson MMS / projection | passes after fixing a device-capture bug |
| boundary conditions | passes after capturing BC scalars for the device |
| 3D compressible MMS | passes, WENO5-Z order 5.14 at 8/16 |
| 3D compressible driver | acoustic K `8.08e-10` matches CPU; TGV and double Mach run |
| full CUDA CTest suite | **32/32 pass** |
| CUDA + MPI suite | **44/44 pass** (2/4 ranks sharing the single GPU) |

Two CUDA-specific capture bugs were fixed: the generic RK helpers were
capturing `PatchField` objects instead of their views (and using the
destination's ghost offset for sources with different `ng`), and the boundary
condition kernel captured a C array of boundary types. Both produced stale or
host-memory reads on device. After the fixes the CUDA results match the CPU
results bit-for-bit for the tested cases.

## Compressible driver

3D Taylor–Green, `--tend 0.2`, identical diagnostics on CPU and GPU:

| Grid | steps | GPU wall | CPU (8 threads) | speedup |
|---|---|---|---|---|
| 64³ | 11 | 1.46 s | 2.61 s | 1.8x |
| 128³ | 21 | 19.39 s | 38.17 s | 2.0x |

The compressible kernels are not yet tuned: face fluxes, gradients, and the
viscous stencil are separate kernels with substantial workspace traffic and
launch overhead. Kernel fusion, a tuned tile size, and occupancy work are the
next performance steps. The GPU results match the CPU diagnostics bit-for-bit
for these cases.

## Notes

- CUDA 12.2 does not support GCC 13; the local GCC 12 tree under
  `~/opt/gcc-12` is used as `NVCC_WRAPPER_DEFAULT_COMPILER`.
- `scripts/envCuda.sh` sets the host compiler, CUDA library path, and Kokkos
  environment for `cmake --preset cuda-local`.
- The P600 has 2 GB, so large 3D cases (e.g. 256³ compressible) need care; the
  stencil/STREAM tests above fit comfortably.
