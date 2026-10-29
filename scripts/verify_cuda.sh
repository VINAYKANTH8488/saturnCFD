#!/usr/bin/env bash
# Build and smoke-test the CUDA configuration after the driver and toolkit are
# installed. Run from the repository root.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

# CUDA 12.2 + GCC 13 do not mix; use the user-local GCC 12 tree if present.
GCC12_ROOT="${GCC12_ROOT:-$HOME/opt/gcc-12}"
if [ -x "$GCC12_ROOT/usr/bin/g++-12" ]; then
  export NVCC_WRAPPER_DEFAULT_COMPILER="$GCC12_ROOT/usr/bin/g++-12"
  export GCC_EXEC_PREFIX="$GCC12_ROOT/usr/lib/gcc/"
  export COMPILER_PATH="$GCC12_ROOT/usr/lib/gcc/x86_64-linux-gnu/12"
  export CPLUS_INCLUDE_PATH="$GCC12_ROOT/usr/include/c++/12:$GCC12_ROOT/usr/include/x86_64-linux-gnu/c++/12"
  export LIBRARY_PATH="$GCC12_ROOT/usr/lib/gcc/x86_64-linux-gnu/12:$GCC12_ROOT/usr/lib/x86_64-linux-gnu"
fi

echo "== driver =="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader

echo "== nvcc =="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not on PATH; install the CUDA 12.2 toolkit and source scripts/env.sh" >&2
  exit 1
fi
nvcc --version | tail -2

echo "== configure/build cuda-local =="
cmake --preset cuda-local
cmake --build build/cuda-local -j4

echo "== stencil verification (CUDA) =="
./build/cuda-local/apps/saturn_stencil --space cuda --verify \
  --nx 32 --ny 32 --nz 32 --iters 2

echo "== STREAM (CUDA) =="
./build/cuda-local/apps/saturn_stream --space cuda --kernel copy \
  --n 1048576 --iters 5

echo "== CUDA test suite =="
ctest --test-dir build/cuda-local --output-on-failure

echo "CUDA smoke test passed."
