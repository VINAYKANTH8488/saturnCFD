#pragma once

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <string>
#include <utility>

namespace saturn {

inline int unavailableBackend(const std::string& name) {
  std::fprintf(stderr, "backend '%s' is not enabled in this Kokkos build\n",
               name.c_str());
  return 2;
}

// Dispatch a callable on the requested execution-space type.
//
// The callable is invoked as f(Kokkos::SomeExecutionSpace{}), so a generic
// lambda can recover the type with decltype and instantiate a template:
//
//   saturn::dispatchExecutionSpace(space, SATURN_DEFAULT_SPACE, [&](auto tag)
//   {
//     using space_type = decltype(tag);
//     return runImpl<space_type>(params);
//   });
template <class F>
int dispatchExecutionSpace(const std::string& requested,
                           const char* default_space, F&& f) {
  std::string space = requested;
  if (space == "default") {
    space = default_space;
  }

  if (space == "default") {
    return std::forward<F>(f)(Kokkos::DefaultExecutionSpace{});
  }
  if (space == "serial") {
#ifdef KOKKOS_ENABLE_SERIAL
    return std::forward<F>(f)(Kokkos::Serial{});
#else
    return unavailableBackend(space);
#endif
  }
  if (space == "openmp") {
#ifdef KOKKOS_ENABLE_OPENMP
    return std::forward<F>(f)(Kokkos::OpenMP{});
#else
    return unavailableBackend(space);
#endif
  }
  if (space == "cuda") {
#ifdef KOKKOS_ENABLE_CUDA
    return std::forward<F>(f)(Kokkos::Cuda{});
#else
    return unavailableBackend(space);
#endif
  }
  if (space == "hip") {
#ifdef KOKKOS_ENABLE_HIP
    return std::forward<F>(f)(Kokkos::HIP{});
#else
    return unavailableBackend(space);
#endif
  }
  if (space == "sycl") {
#ifdef KOKKOS_ENABLE_SYCL
    return std::forward<F>(f)(Kokkos::SYCL{});
#else
    return unavailableBackend(space);
#endif
  }

  std::fprintf(stderr, "unknown execution space '%s'\n", space.c_str());
  return 2;
}

}  // namespace saturn
