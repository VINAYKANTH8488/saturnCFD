#!/usr/bin/env bash
# Verify that the NVIDIA driver is loaded and usable by CUDA.
set -euo pipefail

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found (driver not installed)"
nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi cannot talk to the driver (reboot?)"

echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
major=${driver%%.*}
if [ "${major:-0}" -lt 525 ]; then
  fail "driver ${driver} is older than 525; CUDA 12.2 needs >= 525"
fi

ls /dev/nvidia0 >/dev/null 2>&1 || fail "/dev/nvidia0 missing"
echo "device nodes:"
ls -l /dev/nvidia* | head

if command -v nvcc >/dev/null 2>&1; then
  echo "CUDA compiler:"
  nvcc --version | tail -2
else
  echo "note: nvcc not on PATH; install the CUDA 12.2 toolkit under ~/opt/cuda-12.2"
  echo "      and source scripts/env.sh before building the CUDA preset."
fi

echo "GPU verification passed."
