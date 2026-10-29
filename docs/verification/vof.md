# VOF: 2D PLIC reconstruction and geometric advection

**Date:** 2026-10-29
**Status:** 2D periodic prototype (host-side) with PLIC reconstruction and
Strang-split geometric fluxes; mass-conservative and bounded. 3D split,
curvature/CSF, and redistribution are next.
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

## Limitations and next steps

- Host-side 2D prototype; no Kokkos/GPU version yet.
- Dimensional splitting is first order in time (Strang reduces the splitting
  error but the flux is still explicit).
- No 3D PLIC, no bounded redistribution/normalization, no interface
  compression.
- No curvature (height function), no CSF surface tension, no variable-density
  coupling.
- Next: 3D split PLIC, 3D translation/rotation tests, then curvature and
  balanced-force CSF (dam break, RTI, rising bubble).
