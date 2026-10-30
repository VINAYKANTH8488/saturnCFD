# Coding style and layout

## Naming

| Kind | Convention | Example |
|---|---|---|
| Functions | `camelCase`, no underscores | `momentumRhs`, `projectClustered` |
| Types / classes | `PascalCase` | `GeometricMultigrid`, `ClusteredPoisson2D` |
| Constants | `kPascalCase` | `kPi`, `kGhiaCount` |
| Local variables | `snake_case` | `max_speed`, `cell_volume` |
| Member data | trailing underscore | `nx_`, `u_` |
| Files | `snake_case` | `cavity.hpp`, `poisson.hpp` |

**Functions never contain underscores.** Third-party APIs (Kokkos, MPI, FFTW,
the C++ standard library, CMake) keep their published names and are the only
underscored calls in the code.

## Layout

```
src/core/            execution-space dispatch, MPI helpers, parallel loops
src/fields/          field metadata, patch fields, boundary conditions
src/halo/            halo exchange (box, face, global)
src/io/              VTK and checkpoint I/O
src/mesh/            boxes, uniform grid, decomposition
src/numerics/
  incompressible/    MAC projection, cavity, channel, Poisson, EDAC
  multiphase/        two-phase and dam-break solvers
  vof/               PLIC, curvature, redistribution
  compressible/      reconstruction, Riemann solvers, compressible NS
apps/                command-line drivers
tests/               CTest targets
docs/                verification notes, style, Doxygen output
```

## Doxygen

- Every public header starts with `/** @file ... @brief ... */`.
- Public classes and functions use `@brief`, `@param`, `@return`.
- Build the API docs with:

```bash
./scripts/build_docs.sh
```

The generated HTML is written to `docs/doxygen/html`.

## Formatting

`clang-format` 23.1.3 with the repository `.clang-format` (Google base,
80 columns). Format changed files before committing:

```bash
clang-format -i $(git ls-files 'src/*.hpp' 'apps/*.cpp' 'tests/*.cpp')
```

## Tests

```bash
source scripts/env.sh
cmake --build build/cpu-openmp -j8
ctest --test-dir build/cpu-openmp

source scripts/env_cuda.sh
cmake --build build/cuda-local -j4
ctest --test-dir build/cuda-local -R 'saturn.(vof|twophase|variable_poisson|dambreak|cavity)'
```
