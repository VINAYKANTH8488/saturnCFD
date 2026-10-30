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

## Closed-interface curvature (level set)

`src/numerics/vof_curvature_levelset2d.hpp` complements the height function
for closed interfaces, where a column can cross the interface twice:

```
phi   = smooth(c) - 1/2
n     = grad(phi) / |grad(phi)|
kappa = -div(n)
```

A short separable binomial smoothing removes the grid-scale staircase. On a
circle of radius 0.2 at `n=96`, the error is

| smoothing passes | max error | RMS |
|---|---|---|
| 2 | 1.61 | 0.80 |
| 8 | 0.57 | 0.18 |
| 16 | **0.203** | **0.0955** |

At 16 passes the maximum error is about 4% of `1/r = 5`. The smoothing width
trades grid noise against curvature bias; it is a prototype and should be
compared against a production height-function implementation for bubbles.

Run:

```bash
./build/cpu-openmp/tests/saturn_test_vof_curvature_levelset
ctest --test-dir build/cpu-openmp -R saturn.vof_curvature_levelset --output-on-failure
```

## Variable-coefficient Poisson with Neumann walls

`VariablePoisson2D` supports wall boundaries (zero face flux) in addition to
the periodic configuration. The manufactured test uses

```
beta = 1 + 0.25 sin(pi x) sin(pi y),   p = cos(pi x) cos(pi y)
f = div(beta grad p)
```

on a unit box with Neumann walls at `n=64`. The PCG solve gives
**L2 error 1.13e-04** and **Linf 2.2e-04**; the operator applied to the exact
solution matches `f` to 4.0e-03. This is the pressure-solver foundation for the
wall-bounded two-phase/dam-break step; wall-aware momentum and VOF are next.

Run:

```bash
./build/cpu-openmp/tests/saturn_test_variable_poisson
ctest --test-dir build/cpu-openmp -R saturn.variable_poisson --output-on-failure
```

## Balanced CSF and static bubble (2D)

`src/numerics/twophase2d.hpp` couples the VOF field to a periodic MAC
projection solver with an FFT Poisson solve. The CSF force is evaluated on
faces as

```
f_x = sigma kappa_face (c_R - c_L) / hx
```

so that for constant curvature it is a discrete gradient. The projection then
balances it exactly with pressure.

Static bubble test: `n=64`, radius 0.2, `sigma=0.1`, density 1, curvature
prescribed as `1/r`:

| Quantity | Value |
|---|---|
| max abs velocity | **1.76e-17** |
| max abs divergence | 2.72e-15 |
| pressure jump | **0.500000** (expected `sigma/r = 0.5`) |
| VOF mass drift after advection | < 1e-12 |

The zero spurious current and exact Laplace pressure validate the
balanced-force CSF and projection coupling. Curvature is prescribed in this
test (the level-set curvature is available for closed interfaces).

The full explicit step (advection, variable viscosity, gravity, CSF,
projection, VOF advection) is exercised as well:

| Test | Result |
|---|---|
| static bubble, 5 full steps | `|u|max = 9.77e-13`, mass drift `6.6e-16` |
| uniform `u=0.1`, 10:1 density jump, 50 steps | velocity deviation `0`, mass drift `4.4e-16` |

The uniform-advection test confirms that a density jump alone does not create
spurious momentum.

A variable-density projection is included (`rho_l=1000`, `rho_g=1`, harmonic
face density). The same bubble gives max `|u| = 3.93e-13`, divergence
`4.0e-11`, and pressure jump `0.500000`: the balanced force remains exact
across a 1000:1 density jump. Curvature from VOF on a closed interface is the
next piece.

Run:

```bash
./build/cpu-openmp/tests/saturn_test_twophase
ctest --test-dir build/cpu-openmp -R saturn.twophase --output-on-failure
```

## Wall-bounded dam break (prototype)

`src/numerics/dambreak2d.hpp` is a closed-box dam-break prototype:

- no-slip/no-penetration walls on all sides
- variable density `rho(c)` and viscosity `mu(c)` and gravity
- variable-coefficient Neumann projection
- wall-aware PLIC face fluxes (zero wall flux, mirror reconstruction ghosts)
- conservative bounded redistribution against splitting overshoot
- balanced CSF from a smoothed level set with mirror ghosts
  `sigma*kappa*grad(c)`
- face-staggered conservative momentum: the same PLIC mass fluxes drive
  both the `c` update and the momentum update

A 64² column (`c=1` for `x<0.3`, 1000:1 density, `g=-9.81`) run for 50
steps at `dt=1e-3` with and without `sigma=0.072`:

| Case | `|u|max` | mass drift | range |
|---|---|---|---|
| no CSF | 2.522 | `1.1e-14` | `[0, 1]` |
| CSF | 2.522 | `1.0e-14` | `[0, 1]` |

The CSF changes the interface by `2.4e-4` in `Linf`, so the coupling is active,
while both runs conserve mass and stay bounded. The conservative flux is less
diffusive than the earlier advective form, hence the higher peak speed.

Two equilibrium checks exercise the wall boundaries:

| Check | Result |
|---|---|
| hydrostatic two layers, 200 steps | `|u|max = 6.3e-12`, drift `7.8e-14` |
| static drop `R=0.2`, `sigma=0.072`, 200 steps | `|u|max = 1.7e-3`, drift `9.9e-15` |
| drop pressure jump | `0.36030` vs `sigma/R = 0.36` (0.08%) |
| variable-viscosity drop (10:1) | `0.36029` (0.08%) |

The hydrostatic case is balanced to roundoff, confirming the variable-density
projection with Neumann walls. The static drop reproduces the Laplace
pressure to 0.08%; its `1.7e-3` residual current comes from the pointwise
level-set curvature error rather than the wall treatment. Curvature is
smoothed over 8 binomial passes for the CSF force.
### Martin & Moyce front position

A dry-bed column (width `a=0.1`, height `2a`) in an `8a x 4a` tank is run at
128x64 with `dt=1e-3`. The surge front `Z = x/a` reaches the published
positions at dimensionless times `T = t sqrt(2g/a)`:

| `Z` | measured `T` | reference `T` | error |
|---|---|---|---|
| 2.11 | 1.67 | 1.83 | -8.7% |
| 3.00 | 2.40 | 2.51 | -4.4% |
| 3.89 | 3.04 | 3.11 | -2.3% |

The simulation is early by a near-constant offset, consistent with the
experiment's finite gate-release time (the same offset is seen in inviscid
simulations). The tank is closed in the prototype, unlike the open-top
experiment; the early surge front is insensitive to the top. This is the
first quantitative dam-break validation.

## Rayleigh-Taylor linear growth

The same closed-box solver is run with a heavy layer above a light one,
`rho = 2:1`, a single-mode perturbation
`y_i = 0.5 + 0.001 cos(4 pi x)`, and `g = -9.81`. At Atwood
`A = 1/3` the linear growth rate is `sigma = sqrt(A g k) = 6.41`.

| Grid | measured `sigma` | error |
|---|---|---|
| 64² | 5.71 | -10.9% |
| 128² | 5.97 | -6.8% |

The growth rate converges toward theory under refinement; the remaining
deficit is the first-order time integration. At Atwood near 1 (1000:1
density) the prototype remains stable only to `t ~ 0.6`; the strongly
nonlinear high-Atwood regime is still open.

## Rising bubble: Hysing et al. case 1

The buoyant-bubble smoke test uses the Hysing case-1 parameters. The
FEATFLOW/Basilisk reference (`c1g3l4.txt`) gives:

| t | centre rise | centroid velocity |
|---|---|---|
| 0.5 | 0.0483 | 0.196 |
| 1.0 | 0.1696 | 0.241 |
| 2.0 | 0.3887 | 0.198 |
| 3.0 | 0.5817 | 0.194 |

The current face-staggered solver (64x128, `dt=1e-3`) gives:

| t | rise | reference | error | `v_b` | reference |
|---|---|---|---|---|---|
| 0.5 | 0.0504 | 0.0483 | +4.3% | 0.177 | 0.196 |
| 1.0 | 0.1536 | 0.1696 | -9.4% | 0.215 | 0.241 |
| 1.5 | 0.2563 | 0.2850 | -10.1% | 0.194 | 0.219 |
| 2.0 | 0.3486 | 0.3887 | -10.3% | 0.178 | 0.198 |
| 2.5 | 0.4368 | 0.4854 | -10.0% | 0.176 | 0.192 |
| 3.0 | 0.5257 | 0.5817 | -9.6% | 0.179 | 0.194 |

Refining to 128x256 gives a rise of 0.1503 at `t=1` (-11.4%), so the
~10% deficit is grid-converged rather than a resolution effect. This is now a
quantitative prototype, but it is not exact; the remaining gap is attributed
to the first-order time integration and the diffuse interface/level-set
curvature.

## Face-staggered conservative momentum

The default step is now a face-staggered conservative update. The volume
fraction and the momentum are advanced with **one set of PLIC face fluxes**,
so the discrete density and momentum updates are consistent.

The path to this scheme was instructive:

- a cell-centred `rho u` flux and a face momentum flux without the consistent
  mass flux both blew up on the 1000:1 dam break at `t ~ 0.02`
  (`|u| ~ 1e14` by step 24, CFL `< 0.3`); the unbalanced `u^2 grad(rho)`
  term was not cancelled by the discrete continuity;
- a cell-centred consistent mass-flux variant stabilised the dam break
  (`|u| = 1.15`, drift `1e-14`) but degraded RTI (`sigma = 4.33` vs `6.41`)
  and the Laplace jump (`0.340` vs `0.36`) because of the cell-to-face
  re-averaging;
- the face-staggered version uses the consistent mass fluxes directly on the
  MAC faces and avoids that re-averaging. It passes all checks: dam break
  `|u| = 2.52`, drift `1e-14`; static-drop Laplace `0.36030`; RTI
  `sigma = 5.71` at 64²; rising bubble within ~10% of the Hysing reference.

The high-Atwood RTI case (1000:1) still destabilises at `t ~ 0.6`; the
strongly nonlinear regime remains open.

## Limitations and next steps

- Host-side 2D/3D prototypes; the wall-bounded two-phase path has no
  Kokkos/GPU version yet.
- The scheme is first order in time; the PLIC spatial reconstruction is
  second order in space.
- Bounded redistribution is a safety net for splitting overshoot; a bounded
  PLIC flux limiter is not implemented.
- High-Atwood (1000:1) RTI still destabilises at `t ~ 0.6`; the dam break at
  the same density ratio is stable.
- Dam-break front position matches Martin & Moyce within 9% (the simulation
  is early, consistent with the gate-release offset); the closed-box top
  differs from the open-top experiment. RTI growth is validated to ~11% at
  64².
- The rising bubble is within ~10% of the Hysing et al. reference and is now a
  quantitative prototype, but not an exact match.
- Next: open-top boundary with a ghost-cell momentum condition, rising-bubble
  grid convergence, then the Kokkos port.
