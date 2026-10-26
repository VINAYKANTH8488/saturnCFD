# saturnCFD

Performance-portable, block-structured, high-order finite-volume DNS code for
CPU and multiple GPUs, built on Kokkos. Basilisk-inspired in physics and
workflow, designed for GPU-first scaling.

**Status:** Phase 0 (foundation). The full design, numerics, and roadmap are in
[PLAN.md](PLAN.md).

Phase 0 progress:

- CPU toolchain and build presets verified (Kokkos 5.2.2, Serial + OpenMP).
- 7-point stencil microbenchmark: 0.985 ns/cell/iter on 4 threads, **83% of
  STREAM copy** on the i7-7700 (see `benchmarks/results/`).
- EDAC pressure-free prototype completed; projection + GMG remains the
  production default (see `experiments/edac_tgv/`).
- 2nd-order scalar advection–diffusion MMS passes on Serial/OpenMP: observed
  order **2.00**, mass drift ~1e-16 (see `docs/verification/scalar_mms.md`).
- CUDA/HIP/SYCL presets scaffolded; CUDA verification pending the local
  toolkit install.

## Requirements

- C++20 compiler (GCC >= 10.4, Clang >= 14)
- CMake >= 3.22
- Kokkos 5.x (Serial + OpenMP for CPU; CUDA/HIP/SYCL for GPUs)
- MPI, HDF5, Catch2, toml++ (from Phase 1)

## Toolchain

The project is designed to build against a user-local Kokkos installation, so no
root access is needed. On this machine the reference toolchain is:

```text
~/opt/cmake-3.31.6-linux-x86_64
~/opt/kokkos-5.2.2            # Serial + OpenMP
```

Generic install recipe:

```bash
# CMake (user-local)
curl -L -o /tmp/cmake.tar.gz \
  https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
tar -xzf /tmp/cmake.tar.gz -C "$HOME/opt"
ln -sfn "$HOME/opt/cmake-3.31.6-linux-x86_64" "$HOME/opt/cmake"

# Kokkos (Serial + OpenMP)
git clone --depth 1 --branch 5.2.2 https://github.com/kokkos/kokkos.git \
  "$HOME/opt/src/kokkos-5.2.2"
cmake -S "$HOME/opt/src/kokkos-5.2.2" -B "$HOME/opt/build/kokkos-5.2.2" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/kokkos-5.2.2" \
  -DCMAKE_CXX_STANDARD=20 \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build "$HOME/opt/build/kokkos-5.2.2" -j4
cmake --install "$HOME/opt/build/kokkos-5.2.2"
```

GPU builds need a separate Kokkos installation with the matching backend and
architecture, e.g.:

```bash
-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON   # Quadro P600 (sm_61)
-DKokkos_ENABLE_HIP=ON  -DKokkos_ARCH_AMD_GFX90A=ON # MI250
-DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON  # Ponte Vecchio
```

Point `KOKKOS_ROOT_CUDA`, `KOKKOS_ROOT_HIP`, or `KOKKOS_ROOT_SYCL` at those
installations.

## Build

```bash
source scripts/env.sh

cmake --preset cpu-openmp
cmake --build build/cpu-openmp -j4

ctest --test-dir build/cpu-openmp --output-on-failure
```

Presets: `cpu`, `cpu-serial`, `cpu-openmp`, `cuda-local`, `cuda-a100`,
`cuda-h100`, `hip-mi250`, `sycl-pvc`.

## Stencil microbenchmark

The first performance calibration is a 7-point 3D stencil in FP64:

```bash
export OMP_PROC_BIND=spread OMP_PLACES=threads   # avoids a Kokkos warning
./build/cpu-openmp/apps/saturn_stencil \
  --space openmp --nx 256 --ny 256 --nz 256 --iters 50 \
  --json benchmarks/results/stencil_cpu_openmp.json
```

Useful variants:

```bash
# Serial reference
./build/cpu-openmp/apps/saturn_stencil --space serial --nx 256 --ny 256 --nz 256

# Pin Kokkos/OpenMP threads
./build/cpu-openmp/apps/saturn_stencil --space openmp --kokkos-num-threads=4

# Correctness check on a small grid
./build/cpu-openmp/apps/saturn_stencil --verify --nx 32 --ny 32 --nz 32 --iters 2
```

Reported metrics:

- `ns/cell/iter`: inverse throughput, comparable across backends.
- `GB/s effective`: one read + one write per cell (the roofline traffic).
- `GB/s naive`: eight memory accesses per cell (upper bound on load traffic).

The effective number is the one to compare against STREAM bandwidth.

## STREAM microbenchmark

To separate machine bandwidth from stencil efficiency:

```bash
./build/cpu-openmp/apps/saturn_stream \
  --space openmp --kernel copy --n 33554432 --iters 50 \
  --json benchmarks/results/stream_copy_cpu_openmp.json

./build/cpu-openmp/apps/saturn_stream \
  --space openmp --kernel triad --n 33554432 --iters 50 \
  --json benchmarks/results/stream_triad_cpu_openmp.json
```

`copy` moves two arrays (16 bytes/element); `triad` reads two and writes one
(24 bytes/element). The stencil's effective GB/s should be interpreted as a
fraction of these numbers.

## Scalar verification (Phase 1)

The first numerical gate is a manufactured-solution test for 2nd-order
advection–diffusion on a periodic 3D box:

```bash
./build/cpu-openmp/apps/saturn_scalar_mms \
  --space openmp --n-list 16 32 64 128 \
  --nu 0.01 --tend 0.2 --check
```

Expected: observed order **2.00** and machine-precision mass conservation.
Details are in `docs/verification/scalar_mms.md`.

## Layout

```text
apps/        microbenchmarks and (later) simulation drivers
cmake/       options, warnings, toolchain helpers
src/core/    version, configuration, logging, timers
tests/       smoke and unit tests
benchmarks/  case definitions and recorded results
experiments/ numerical de-risk prototypes (EDAC TGV)
scripts/     environment setup and run scripts
docs/        design notes (added as the project grows)
```
