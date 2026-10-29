#!/usr/bin/env bash
# Build and smoke-test the CUDA configuration after the driver and toolkit are
# installed. Run from the repository root.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

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

echo "== selected GPU tests =="
ctest --test-dir build/cuda-local \
  -R 'saturn\.(smoke|reconstruction|riemann|poisson|fft_poisson)' \
  --output-on-failure

echo "CUDA smoke test passed."
