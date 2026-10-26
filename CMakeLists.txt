cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(saturnCFD
  VERSION 0.1.0
  DESCRIPTION "Performance-portable block-structured high-order FVM DNS code"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SaturnOptions)
include(SaturnWarnings)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Kokkos 5.0 CONFIG REQUIRED)

# Interface target carrying include paths, Kokkos, and project-wide flags.
add_library(saturn_core INTERFACE)
target_include_directories(saturn_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(saturn_core INTERFACE Kokkos::kokkos)
target_compile_features(saturn_core INTERFACE cxx_std_20)

if(SATURN_ENABLE_WARNINGS)
  saturn_enable_warnings(saturn_core)
endif()

if(SATURN_ENABLE_SANITIZERS)
  target_compile_options(saturn_core INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(saturn_core INTERFACE
    -fsanitize=address,undefined)
endif()

if(NOT DEFINED SATURN_DEFAULT_SPACE)
  set(SATURN_DEFAULT_SPACE "default")
endif()
message(STATUS "saturnCFD default execution space: ${SATURN_DEFAULT_SPACE}")

add_subdirectory(apps)

if(SATURN_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
