# saturnCFD

Performance-portable, block-structured, high-order finite-volume DNS code for
CPU and multiple GPUs, built on Kokkos. Basilisk-inspired in physics and
workflow, designed for GPU-first scaling.

**Status:** Phase 1 complete; Phase 2 (high-order compressible) in progress. The
full design, numerics, and roadmap are in [PLAN.md](PLAN.md).

Progress:

- CPU and CUDA toolchains verified: Kokkos 5.2.2 (Serial + OpenMP) and CUDA
  12.2 (sm_61, Quadro P600).
- 7-point stencil microbenchmark: **84% of STREAM copy** on 4 CPU threads and
  **52%** on the P600 (see `benchmarks/results/`).
- EDAC pressure-free prototype completed; projection + GMG remains the
  production default (see `experiments/edac_tgv/`).
- 2nd-order scalar advection–diffusion MMS passes on **Serial, OpenMP, and
  CUDA**: observed order **2.00**, mass drift ~1e-16 (see
  `docs/verification/scalar_mms.md`).
- **MPI halo exchange verified on 1, 2, 4, and 8 CPU ranks and 2 CUDA ranks**;
  the face-halo unit test reports zero errors.
- Halo/compute overlap implemented (`begin`/`end` + boundary shell) and
  correctness-tested; opt-in because it is not yet faster at the tested sizes
  (`docs/verification/halo_overlap.md`).
- VTK snapshots and binary checkpoint/restart pass on 1–2 ranks on CPU and CUDA
  (`docs/verification/io.md`); HDF5/ADIOS2 are later.
- HIP/SYCL presets scaffolded; tuning waits for hardware access.
- Phase 2 face reconstruction implemented: `central2/4/6`, WENO5-Z, and TENO6,
  with unit tests for polynomial exactness, 2nd–6th order convergence,
  symmetry, and shock-boundedness (`docs/verification/reconstruction.md`).
- Phase 2 Riemann solvers implemented: LLF, HLLC, HLLC-low-Mach, and
  AUSM+-up, with consistency, contact, symmetry, shock, and low-Mach tests
  (`docs/verification/riemann.md`).
- Phase 2 1D Sod shock tube passes end-to-end with WENO5-Z/TENO6
  reconstruction and every Riemann solver; L1 density error is 1.1e-3 at 400
  cells (`docs/verification/sod.md`).
- Phase 2 3D compressible Navier–Stokes path implemented: inviscid reconstruction
  + Riemann fluxes, 4th-order viscous stress/heat-flux terms, RK3/RK4. The
  entropy-wave and viscous-shear MMS give observed orders **3.92–3.98** with
  `central4`, **5.14–5.27** with WENO5-Z, and **5.87–5.95** with `central6`
  (`docs/verification/compressible_mms.md`).
- Periodic inviscid runs conserve mass and total energy to round-off
  (**≤1.1e-15** relative drift); `--check` enforces this via `--energy-tol`.
- Phase 2 3D compressible driver with TGV, acoustic-pulse, low-Mach-vortex, and
  standard Woodward–Colella double-Mach setups; VTK primitive snapshots, binary
  checkpoint/restart, per-step CSV diagnostics for energy-dissipation curves,
  exact time-dependent double-Mach inflow boundaries, and
  full 3D halo exchange (faces, edges, corners), verified on 2 and 4 MPI ranks;
  multi-rank non-periodic BCs use an all-gather fallback
  (`docs/verification/compressible_driver.md`).
- Long-time TGV diagnostics at 64³: WENO5-Z is stable to `t=10` for `M≈1` and
  the standard `M=0.1` setup. The low-Mach run peaks at `1.195e-2` at `t=7.04`
  (reference `~1.25e-2` near `t≈9`, 4.4% low as expected at 64³); total-energy
  drift `≤2e-13`. `central4` is unstable at this resolution
  (`docs/verification/tgv.md`).

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

CUDA build (after installing a CUDA toolkit and a matching Kokkos build):

```bash
export KOKKOS_ROOT_CUDA=$HOME/opt/kokkos-5.2.2-cuda
cmake --preset cuda-local
cmake --build build/cuda-local -j4

./build/cuda-local/apps/saturn_stencil --space cuda --nx 256 --ny 256 --nz 256
./build/cuda-local/apps/saturn_scalar_mms --space cuda --n-list 16 32 64 --check
```

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

## High-order reconstruction (Phase 2)

`src/numerics/reconstruction.hpp` provides device-callable left/right face
reconstruction for `central2`, `central4`, `central6`, `weno5z`, and `teno6`.
The unit test checks polynomial exactness, 2nd–6th order convergence, symmetry,
and non-oscillatory behaviour at a step:

```bash
ctest --test-dir build/cpu-openmp -R saturn.reconstruction --output-on-failure
```

Observed orders are 2.00, 4.00, 5.99, 5.06, and 5.99 respectively. Details and
the TENO6 formulation are in `docs/verification/reconstruction.md`.

## Riemann solvers (Phase 2)

`src/numerics/riemann.hpp` provides device-callable LLF, HLLC, HLLC-low-Mach,
and AUSM+-up fluxes for the ideal-gas Euler equations in face-normal
coordinates:

```cpp
auto f = saturn::riemann::flux(
    saturn::RiemannSolver::Hllc, left, right, /*gamma=*/1.4);
```

```bash
ctest --test-dir build/cpu-openmp -R saturn.riemann --output-on-failure
```

The tests cover uniform-state consistency, stationary and moving contacts,
mirror symmetry, Sod-state fluxes, and the low-Mach pressure-diffusion
behaviour. Details are in `docs/verification/riemann.md`.

## Sod verification (Phase 2)

The 1D Euler driver combines reconstruction, a Riemann solver, and SSP-RK3 and
compares against the exact Sod solution:

```bash
./build/cpu-openmp/apps/saturn_euler1d --n 400 --check
./build/cpu-openmp/apps/saturn_euler1d \
  --n 400 --scheme teno6 --solver hllc_lm --check
```

Observed L1 density error is `1.1e-3` (WENO5-Z + HLLC) and `1.0e-3`
(TENO6 + HLLC) at 400 cells; the left/right star densities match the exact
values to about 1e-4. Details and the full scheme/solver table are in
`docs/verification/sod.md`.

## Compressible Navier–Stokes (Phase 2)

`src/numerics/compressible_ns.hpp` implements the 3D ideal-gas compressible
Navier–Stokes RHS: reconstruction + Riemann inviscid fluxes, 4th-order viscous
stress and heat-flux terms, and a periodic multi-dimensional update. The
verification driver has two exact cases:

```bash
# Inviscid entropy wave: tests reconstruction + Riemann in 3D.
./build/cpu-openmp/apps/saturn_compressible_mms \
  --case entropy --n-list 8 16 32 --scheme central4 --time rk4 --cfl 0.3

# Viscous shear: tests the full stress tensor and energy terms.
./build/cpu-openmp/apps/saturn_compressible_mms \
  --case shear --n-list 8 16 32 --scheme central4 --time rk4 --cfl 0.3
```

Observed orders: `central4` 3.92–3.98, WENO5-Z 5.14–5.27, `central6`
5.87–5.95. On the periodic inviscid case mass and total energy are conserved to
round-off (`|dM/M| ≤ 1e-16`, `|dE/E| ≤ 1.1e-15`); `--check` enforces the
`--energy-tol` limit. See `docs/verification/compressible_mms.md`. The driver is
periodic-only in multi-rank mode; single-rank transmissive/slip boundaries are
supported, and TGV/acoustic/vortex/double-Mach drivers plus VTK snapshots and
checkpoint/restart are implemented. See `docs/verification/compressible_driver.md`.

## 3D compressible driver

`apps/compressible_sim.cpp` is the production-style 3D driver. It supports
`tgv`, `acoustic`, `vortex`, and `dm` cases, VTK snapshots of primitive fields,
and checkpoint/restart:

```bash
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 32 --tend 1.0 --output /tmp/tgv --plot-every 20

# Global diagnostics every 10 steps (CSV for dissipation curves).
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 64 --tend 10 --diag-every 10 --history tgv.csv
python3 scripts/tgv_dissipation.py tgv.csv --volume 248.0502134
```

Short stability gates:

```bash
ctest --test-dir build/cpu-openmp -R 'saturn.compressible_(tgv|acoustic|vortex|double_mach)'   --output-on-failure
```

The full halo exchange (`src/halo/box_exchange.hpp`) fills faces, edges, and
corners for the fourth-order viscous stencil. Multi-rank runs currently require
all-periodic boundaries; non-periodic single-rank runs use
`src/fields/boundary_conditions.hpp`. Details:
`docs/verification/compressible_driver.md`. Long-time Taylor–Green
dissipation diagnostics and the WENO5-Z/central4 stability comparison are in
`docs/verification/tgv.md`.

## MPI

The CPU build enables MPI by default. The scalar MMS can be run on multiple
ranks; each rank owns a Cartesian sub-box and exchanges face ghosts:

```bash
# 1, 2, 4, or 8 ranks
/usr/bin/mpiexec -n 4 ./build/cpu-openmp/apps/saturn_scalar_mms \
  --space openmp --n-list 32 64 --nu 0.01 --tend 0.2 --check \
  --kokkos-num-threads=1
```

Use the `mpiexec` from the same MPI installation as `mpicxx`. CTest also runs
a 2-rank halo test and a 2-rank MMS test when MPI is found:

```bash
ctest --test-dir build/cpu-openmp --output-on-failure
```

GPU builds stage halo buffers through host memory; GPU-aware MPI is a later
optimization. Non-periodic physical boundaries are not yet implemented (the
exchange skips `MPI_PROC_NULL` faces).

Halo cost:

```bash
/usr/bin/mpiexec -n 4 ./build/cpu-openmp/apps/saturn_halo \
  --space openmp --n 256 --ng 2 --iters 10 --warmup 2 \
  --kokkos-num-threads=1 --json benchmarks/results/halo/halo_np4.json
```

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
.github/     CI (Kokkos build + full CTest + clang-format check)
```

## Style and CI

C++ is formatted with `clang-format` 23.1.3 using the Google base style in
`.clang-format` (`IncludeBlocks: Preserve`). Format `src/`, `apps/`, and
`tests/` before submitting changes:

```bash
clang-format -i $(git ls-files 'src/*.hpp' 'apps/*.cpp' 'tests/*.cpp')
```

`.github/workflows/ci.yml` builds Kokkos 5.2.2 (Serial + OpenMP), configures
with MPI, runs the full CTest suite, and checks formatting.
