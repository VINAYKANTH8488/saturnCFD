#!/usr/bin/env bash
# Build and install Kokkos with the CUDA backend for the local Quadro P600.
#
# Prerequisite: the CUDA 12.2 toolkit is installed under $CUDA_ROOT (default
# ~/opt/cuda-12.2) and the NVIDIA driver is loaded (scripts/verify_gpu.sh).
set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-$HOME/opt/cuda-12.2}"
KOKKOS_SRC="${KOKKOS_SRC:-$HOME/opt/src/kokkos-5.2.2}"
KOKKOS_PREFIX="${KOKKOS_PREFIX:-$HOME/opt/kokkos-5.2.2-cuda}"
BUILD_DIR="${BUILD_DIR:-$HOME/opt/build/kokkos-5.2.2-cuda}"
ENV_NVCC="$CUDA_ROOT/bin/nvcc"
export PATH="$CUDA_ROOT/bin:$PATH"

# CUDA 12.2 does not support GCC 13. If a user-local GCC 12 tree exists, use it
# for the host side of nvcc.
GCC12_ROOT="${GCC12_ROOT:-$HOME/opt/gcc-12}"
if [ -x "$GCC12_ROOT/usr/bin/g++-12" ]; then
  export NVCC_WRAPPER_DEFAULT_COMPILER="$GCC12_ROOT/usr/bin/g++-12"
  export GCC_EXEC_PREFIX="$GCC12_ROOT/usr/lib/gcc/"
  export COMPILER_PATH="$GCC12_ROOT/usr/lib/gcc/x86_64-linux-gnu/12"
  export CPLUS_INCLUDE_PATH="$GCC12_ROOT/usr/include/c++/12:$GCC12_ROOT/usr/include/x86_64-linux-gnu/c++/12"
  export LIBRARY_PATH="$GCC12_ROOT/usr/lib/gcc/x86_64-linux-gnu/12:$GCC12_ROOT/usr/lib/x86_64-linux-gnu"
  echo "using GCC 12 host compiler: $GCC12_ROOT/usr/bin/g++-12"
else
  echo "note: no user-local GCC 12 found; nvcc will use the system compiler"
  echo "      (CUDA 12.2 + GCC 13 needs -allow-unsupported-compiler)"
fi

if [ ! -x "$ENV_NVCC" ]; then
  echo "error: nvcc not found at $ENV_NVCC" >&2
  echo "       install the CUDA 12.2 toolkit (--toolkit only) first" >&2
  exit 1
fi
if [ ! -x "$KOKKOS_SRC/bin/nvcc_wrapper" ]; then
  echo "error: Kokkos source not found at $KOKKOS_SRC" >&2
  exit 1
fi

cmake=/usr/bin/cmake
if [ -x "$HOME/opt/cmake/bin/cmake" ]; then
  cmake="$HOME/opt/cmake/bin/cmake"
fi

# CUDA 12.2 nominally supports up to GCC 12; this machine has GCC 13. The
# wrapper flag lets nvcc proceed with the host compiler (usually works for
# Kokkos and the project kernels).
"$cmake" -S "$KOKKOS_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$KOKKOS_PREFIX" \
  -DCMAKE_CXX_COMPILER="$KOKKOS_SRC/bin/nvcc_wrapper" \
  -DCMAKE_CXX_FLAGS="-allow-unsupported-compiler" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_PASCAL61=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  -DKokkos_ENABLE_EXAMPLES=OFF \
  -DKokkos_ENABLE_BENCHMARKS=OFF

"$cmake" --build "$BUILD_DIR" -j"$(nproc)"
"$cmake" --install "$BUILD_DIR"

echo
echo "Kokkos-CUDA installed at $KOKKOS_PREFIX"
echo "Now run:"
echo "  source scripts/env.sh"
echo "  cmake --preset cuda-local"
echo "  cmake --build build/cuda-local -j4"
