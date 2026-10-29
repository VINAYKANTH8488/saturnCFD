#!/usr/bin/env bash
# Source this file to put the user-local toolchain on PATH.
#   source scripts/env.sh
#
# Override any KOKKOS_ROOT_* variable before sourcing if your installations
# live elsewhere.

_saturn_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# User-local CMake (>= 3.22 is required; system may be older).
if [ -d "${HOME}/opt/cmake/bin" ]; then
  export PATH="${HOME}/opt/cmake/bin:${PATH}"
fi

# User-local CUDA toolkit, when present.
if [ -d "${HOME}/opt/cuda-12.2/bin" ]; then
  export PATH="${HOME}/opt/cuda-12.2/bin:${PATH}"
  export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${HOME}/opt/cuda-12.2}"
fi

export SATURN_ROOT="${_saturn_root}"
export KOKKOS_ROOT="${KOKKOS_ROOT:-${HOME}/opt/kokkos-5.2.2}"
export KOKKOS_ROOT_CUDA="${KOKKOS_ROOT_CUDA:-${HOME}/opt/kokkos-5.2.2-cuda}"
export KOKKOS_ROOT_HIP="${KOKKOS_ROOT_HIP:-${HOME}/opt/kokkos-5.2.2-hip}"
export KOKKOS_ROOT_SYCL="${KOKKOS_ROOT_SYCL:-${HOME}/opt/kokkos-5.2.2-sycl}"

# Optional user-local FFTW3 development tree.
if [ -d "${HOME}/opt/fftw3-dev/usr/include" ]; then
  export FFTW3_ROOT="${FFTW3_ROOT:-${HOME}/opt/fftw3-dev/usr}"
fi

unset _saturn_root
