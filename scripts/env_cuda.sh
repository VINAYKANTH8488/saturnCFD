#!/usr/bin/env bash
# Environment for CUDA builds/runs on this machine.
#
#   source scripts/env_cuda.sh
#   cmake --preset cuda-local
#   cmake --build build/cuda-local -j4
#
# CUDA 12.2 does not support GCC 13, so the user-local GCC 12 tree (extracted
# under ~/opt/gcc-12) is used as nvcc's host compiler when present.

_saturn_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${_saturn_script_dir}/env.sh"
unset _saturn_script_dir

if [ -d "${HOME}/opt/cuda-12.2/targets/x86_64-linux/lib" ]; then
  export LD_LIBRARY_PATH="${HOME}/opt/cuda-12.2/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

_gcc12="${HOME}/opt/gcc-12"
if [ -x "${_gcc12}/usr/bin/g++-12" ]; then
  export NVCC_WRAPPER_DEFAULT_COMPILER="${_gcc12}/usr/bin/g++-12"
  export GCC_EXEC_PREFIX="${_gcc12}/usr/lib/gcc/"
  export COMPILER_PATH="${_gcc12}/usr/lib/gcc/x86_64-linux-gnu/12"
  export CPLUS_INCLUDE_PATH="${_gcc12}/usr/include/c++/12:${_gcc12}/usr/include/x86_64-linux-gnu/c++/12"
  export LIBRARY_PATH="${_gcc12}/usr/lib/gcc/x86_64-linux-gnu/12:${_gcc12}/usr/lib/x86_64-linux-gnu"
fi
unset _gcc12
