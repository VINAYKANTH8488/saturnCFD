#!/usr/bin/env bash
# GPU lid-driven cavity verification.
#
#   ./scripts/verifyCavityGpu.sh
#
# Quantitative Ghia Re=1000 check on the local Quadro P600 plus a Re=5000
# higher-Reynolds stability smoke. The CUDA build is created with the
# `cuda-local` preset if it does not exist.

set -euo pipefail

_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${_root}"
unset _root

# shellcheck source=envCuda.sh
source scripts/envCuda.sh

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found; a CUDA GPU is required" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d build/cuda-local ]; then
  cmake --preset cuda-local
fi
cmake --build build/cuda-local --target saturn_cavity_sim -j"${BUILD_JOBS:-4}"

BIN=build/cuda-local/apps/saturn_cavity_sim

echo
echo "== Re=1000 Ghia centreline validation, WENO5 (n=32, nz=1, t=20) =="
"${BIN}" --n 32 --nz 1 --re 1000 --tend 20 --cfl 0.4 \
  --poisson-tol 1e-6 --advection weno5 \
  --rms-tol 0.20 --div-tol 2e-7 --space cuda --check

echo
echo "== Re=5000 higher-Reynolds GPU stability smoke (n=32, nz=1, t=10) =="
"${BIN}" --n 32 --nz 1 --re 5000 --tend 10 --cfl 0.25 \
  --poisson-tol 1e-6 --advection weno5 --space cuda

echo
echo "cavity GPU verification passed"
