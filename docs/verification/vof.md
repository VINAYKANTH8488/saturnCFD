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

## Height-function curvature (2D)

`src/numerics/vof_curvature2d.hpp` computes the column (or row) height
function

```
h_i = sum_j c_ij hy
```

and the interface curvature

```
kappa = h'' / (1 + h'^2)^(3/2)
```

with 5-point derivatives. On a periodic sine interface
`y = 0.5 + 0.1 sin(2 pi x)` at `n=128`, initialized with exact cell fractions,
the maximum curvature error is **2.54e-07** (RMS 1.19e-07) over 124 sample
columns. This validates the geometric formula and the height-function
derivative stencil; curvature from noisy/advected interfaces and the 3D
height-function version are the next steps.

Run:

```bash
./build/cpu-openmp/tests/saturn_test_vof_curvature
ctest --test-dir build/cpu-openmp -R saturn.vof_curvature --output-on-failure
```

## 3D height-function curvature

`src/numerics/vof_curvature3d.hpp` computes the height function
`h_ij = sum_k c_ijk hz`, 5-point derivatives in x/y, the mixed derivative, and

```
kappa = [h_xx (1+h_y^2) + h_yy (1+h_x^2) - 2 h_x h_y h_xy]
        / (1 + h_x^2 + h_y^2)^(3/2).
```

On `z = 0.5 + 0.1 sin(2 pi x) sin(2 pi y)` at `n=48` (exact cell
fractions), the maximum curvature error is **2.08e-03** (RMS 9.19e-04) over
1936 samples. As with the 2D test, this validates the geometric formula and
derivative stencils; curvature from noisy/advected interfaces is next.

Run:

```bash
./build/cpu-openmp/tests/saturn_test_vof_curvature3d
ctest --test-dir build/cpu-openmp -R saturn.vof_curvature3d --output-on-failure
```

## Bounded redistribution

`src/numerics/vof_redistribute.hpp` clamps any overshoot and restores the
pre-clamp mass by distributing the error to cells with available capacity
(weighted by `1-c` or `c`), iterating until the mass matches. It is a safety
net for sharp interfaces, not a replacement for bounded advection.

Test with a perturbed field (`c` spanning roughly `[-0.05, 1.05]`):

| Grid | range after | relative mass error |
|---|---|---|
| 32² | [2.2e-04, 1] | 4.4e-16 |
| 16³ | [2.0e-14, 0.999] | 8.0e-16 |

Run:

```bash
./build/cpu-openmp/tests/saturn_test_vof_redistribute
ctest --test-dir build/cpu-openmp -R saturn.vof_redistribute --output-on-failure
```

## Limitations and next steps

- Host-side 2D/3D prototypes; no Kokkos/GPU version yet.
- Dimensional splitting is first order in time (Strang reduces the splitting
  error but the flux is still explicit).
- Bounded redistribution is available as a safety net; interface compression
  is not implemented.
- 2D and 3D height-function curvature are verified; CSF surface tension and
  variable-density coupling are not implemented yet.
- Next: 3D split PLIC, 3D translation/rotation tests, then curvature and
  balanced-force CSF (dam break, RTI, rising bubble).
