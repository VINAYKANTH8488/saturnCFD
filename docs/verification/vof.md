# VOF: 2D PLIC reconstruction and geometric advection

**Date:** 2026-10-29
**Status:** 2D and 3D periodic prototypes (host-side) with PLIC reconstruction
and Strang-split geometric fluxes; mass-conservative and bounded. Curvature/CSF,
coupled momentum, and redistribution are next.
**Code:** `src/numerics/vof2d.hpp`, `tests/test_vof.cpp`

## Method

Each cell is reconstructed as a half-plane

```
n . (x - center) <= alpha
```

- **Normal:** Youngs gradient of the volume fraction, `n ∝ -grad(c)`, so the
  normal points from fluid to empty.
- **Alpha:** bisection on the clipped area of the unit square until it equals
  `c` (50 iterations).
- **Advection:** Strang splitting `x(dt/2), y(dt), x(dt/2)`. At every face the
  fluid area crossing the face is the intersection of the upwind cell's PLIC
  polygon with the swept strip of width `|u| dt`; the flux is signed by the
  face velocity. This is geometric (no interface smearing beyond the PLIC
  reconstruction) and conserves mass by construction.

All boundaries are periodic in this prototype.

## Tests

`n=32` for the 2D cases, `n=64` for the 1D step; CFL 0.25.

| Case | L1 error | mass drift | bounds |
|---|---|---|---|
| 1D step translation | 2.50e-03 | 0 | [0,1] |
| circle translation | 6.46e-03 | 1.97e-16 | [0,1] |
| circle solid-body rotation (one period) | 7.71e-03 | 5.92e-16 | [0,1] |

The mass drift is at round-off and the volume fraction stays in `[0,1]`; no
clipping or redistribution is applied.

Run:

```bash
./build/cpu-openmp/tests/saturn_test_vof
ctest --test-dir build/cpu-openmp -R saturn.vof --output-on-failure
```

## 3D PLIC

`src/numerics/vof3d.hpp` extends the reconstruction to 3D: the cell is a unit
cube, clipped by the PLIC half-plane; the plane constant is bisected on the
clipped volume. The cap face is built from the intersection points and sorted
around the plane normal; volume comes from a signed tetrahedral decomposition.
The advection is Strang-split in x, y, z with the same swept-slab geometric
flux.

| Case | L1 error | mass drift | range |
|---|---|---|---|
| clipping check (cube / half cube) | exact | — | — |
| 1D step translation (integer shift) | 1.42e-14 | 0 | [0,1] |
| sphere translation | 3.95e-03 | 3.81e-16 | [0,1] |
| sphere solid-body rotation (one period, 16³) | 3.76e-03 | 1.27e-15 | [-1.1e-13,1] |

Run:

```bash
./build/cpu-openmp/tests/saturn_test_vof3d
ctest --test-dir build/cpu-openmp -R saturn.vof3d --output-on-failure
```

## Limitations and next steps

- Host-side 2D/3D prototypes; no Kokkos/GPU version yet.
- Dimensional splitting is first order in time (Strang reduces the splitting
  error but the flux is still explicit).
- No bounded redistribution/normalization or interface compression; the
  3D rotation range is bounded to ~1e-13 here, but a generic redistribution
  pass is still needed.
- No curvature (height function), no CSF surface tension, no variable-density
  coupling.
- Next: 3D split PLIC, 3D translation/rotation tests, then curvature and
  balanced-force CSF (dam break, RTI, rising bubble).
