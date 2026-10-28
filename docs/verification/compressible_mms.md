# Verification: 3D compressible Navier–Stokes MMS

**Date:** 2026-10-27
**Status:** passing on CPU (Kokkos Serial + OpenMP) and on 2/4 MPI ranks for
all-periodic cases. The 3D path is device-callable; CUDA verification is
pending hardware access.
**Code:** `src/numerics/compressible_ns.hpp`, `apps/compressible_mms.cpp`

## Equations

Ideal-gas compressible Navier–Stokes in conservative form:

```
U_t + div(F_inv) = div(F_visc)

U = [rho, rho u, rho v, rho w, E]
F_inv = [rho u, rho u^2 + p, rho u v, rho u w, u(E+p)]   (x-face example)
E = p/(gamma-1) + 0.5 rho |u|^2
p = rho R T
```

Viscous stress and heat flux:

```
tau_ij = mu (du_i/dx_j + du_j/dx_i - 2/3 delta_ij div u)
q_j    = -k dT/dx_j,    k = mu cp / Pr
cp     = gamma R/(gamma-1)
```

The inviscid flux uses dimension-by-dimension primitive reconstruction and the
Riemann module. The viscous operator is cell-centered and fourth order:

- velocity and temperature gradients: 4-point central differences,
- stress divergence: 4-point central differences of the cell-centered stress,
- energy: `u . div(tau) + tau : grad(u) - div(q)`.

This avoids the second-order error that appears when two fourth-order
cell-centered gradients are merely averaged to a face; the resulting operator
is fourth-order in space. The face-flux arrays are used for the inviscid flux
only.

## Cases

### Entropy wave (inviscid)

```
rho = 1 + 0.2 sin(2 pi (x+y+z) - omega t)
u   = (1, 0.5, 0.25),  p = 1
omega = 2 pi (1 + 0.5 + 0.25)
```

This is an exact solution of the Euler equations. `mu = 0`. It exercises the
3D reconstruction, the contact-preserving HLLC flux, and the multi-dimensional
update.

### Viscous shear (Navier–Stokes)

```
u = 0.1 sin(2 pi y) sin(2 pi z) exp(-nu (2 pi)^2 * 2 t)
v = w = 0,  rho = 1,  p = 1,  nu = mu/rho,  mu = 0.05
```

This is an exact solution of the momentum equation. The physical viscous
heating is removed by an isothermal energy sink (`--sink`, enabled by default)
so that `p` remains uniform; this is a verification device, not a production
model. It tests the full viscous stress tensor, the energy work terms, and the
4th-order viscous operator.

## Results

All runs below use `CFL = 0.3`, `RK4`, and `n-list 8 16 32`.

Entropy wave, `central4 + HLLC`:

| N | L2 rho | order |
|---|---|---|
| 8 | 3.667e-03 | — |
| 16 | 2.422e-04 | 3.92 |
| 32 | 1.535e-05 | 3.98 |

Entropy wave, `weno5z + HLLC`:

| N | L2 rho | order |
|---|---|---|
| 8 | 2.207e-03 | — |
| 16 | 6.253e-05 | 5.14 |
| 32 | 1.622e-06 | 5.27 |

Entropy wave, `central6 + HLLC`:

| N | L2 rho | order |
|---|---|---|
| 8 | 4.629e-04 | — |
| 16 | 7.938e-06 | 5.87 |
| 32 | 1.283e-07 | 5.95 |

Viscous shear, `central4 + HLLC` (all three resolutions are stable; the
viscous operator is fourth order independent of the reconstruction):

| N | L2 u | order | L2 rho | L2 p |
|---|---|---|---|---|
| 8 | 4.239e-04 | — | 6.865e-09 | 8.987e-09 |
| 16 | 2.791e-05 | 3.93 | 1.064e-10 | 3.121e-10 |
| 32 | 1.768e-06 | 3.98 | 3.888e-13 | 1.218e-12 |

The density and pressure errors in the shear case are at round-off level,
confirming that the isothermal sink removes the heating and that the continuity
and energy balances are consistent.

## Multi-rank results

`BoxExchange` (full 3D halo, including edges and corners) is exercised by the
MPI CTest entries:

- `saturn.compressible_entropy_mpi2`: entropy wave, 2 ranks; same order and
  errors as the serial run.
- `saturn.compressible_shear_mpi2`: viscous shear, 2 ranks; same 4th-order
  convergence.
- `saturn.box_exchange_mpi2/4`: every ghost cell (including corners) checked on
  2 and 4 ranks.

## Conservation

On a periodic domain the conservative face fluxes telescope exactly, so an
inviscid run must conserve mass and total energy to round-off. This holds
independently of the reconstruction, the Riemann solver, and the RK order; it
is a direct check that the flux assembly is conservative. The app reports
`|dM/M|` and `|dE/E|` for every resolution, and `--check` enforces
`--energy-tol` (default `1e-10`) for the entropy case.

| Case | Scheme | N | t_end | steps | abs(dM/M) | abs(dE/E) |
|---|---|---|---|---|---|---|
| entropy | central4 | 8 | 0.2 | 13 | 0 | 2.8e-16 |
| entropy | central4 | 16 | 0.2 | 25 | 1.1e-16 | 1.1e-15 |
| entropy | central4 | 32 | 0.2 | 50 | 0 | 0 |
| entropy | weno5z | 16 | 0.5 | 62 | 0 | 1.1e-15 |

The shear case reports the drift as well, but `--check` does not enforce it:
the optional isothermal sink and physical viscous dissipation change the total
energy by design.

## Commands

```bash
./build/cpu-openmp/apps/saturn_compressible_mms \
  --case entropy --n-list 8 16 32 --scheme central4 --time rk4 --cfl 0.3

./build/cpu-openmp/apps/saturn_compressible_mms \
  --case shear --n-list 8 16 32 --scheme central4 --time rk4 --cfl 0.3

ctest --test-dir build/cpu-openmp -R saturn.compressible --output-on-failure
```

`--check` fails if the observed order is below `--min-order` (default 3.7),
or if the periodic inviscid entropy case drifts in mass or total energy above
`--energy-tol`. `--json` records the convergence table and both drifts.

## Limitations and next steps

- Multi-rank non-periodic runs use the `GlobalExchange` all-gather fallback;
  this is correct but not scalable.
- Constant `mu` and `Pr`; variable transport coefficients and Sutherland's law
  are future work.
- The isothermal energy sink is a manufactured-verification device. Production
  runs use the physical viscous heating term.
- CUDA verification and GPU scaling are pending hardware access; the kernels
  are device-callable.
- The remaining physics gates are quantitative comparisons against reference
  data for the double Mach reflection, low-Mach vortex, acoustic pulse, and
  Taylor–Green vortex dissipation curve (`docs/verification/compressible_driver.md`).
