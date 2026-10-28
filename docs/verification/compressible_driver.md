# 3D compressible driver and cases

**Date:** 2026-10-27
**Status:** CPU Serial/OpenMP; single- and multi-rank non-periodic cases.
All-periodic runs use the full 3D halo exchange (`BoxExchange`), verified on 2
and 4 ranks; non-periodic runs use the `GlobalExchange` all-gather fallback.
The double-Mach analytic inflow is applied on top of either path and gives
bitwise-identical diagnostics on 1, 2, and 4 ranks. CUDA validation is
pending.
**Code:** `apps/compressible_sim.cpp`, `src/halo/box_exchange.hpp`,
`src/fields/boundary_conditions.hpp`

## What the driver does

`saturn_compressible_sim` evolves the 3D ideal-gas compressible
Navier–Stokes equations with:

- WENO5-Z/TENO6/central4/central6 reconstruction,
- LLF/HLLC/HLLC-low-Mach/AUSM+-up face fluxes,
- 4th-order cell-centered viscous stress and heat flux,
- SSP-RK3 or RK4,
- periodic, transmissive, and slip-wall boundaries,
- VTK snapshots of `rho, u, v, w, p`,
- binary checkpoint/restart of the five conservative variables.

The full 3D halo (`BoxExchange`) fills faces, edges, and corners, which the
4th-order viscous stencil requires. It exchanges all 26 neighbor blocks and is
host-staged; this is correct but not yet a fused/scalable exchange.

## Cases

| Case | Domain | Default BC | Notes |
|---|---|---|---|
| `tgv` | `[0,2pi]^3` | periodic | 3D Taylor–Green vortex |
| `acoustic` | `[0,1]^3` | periodic | small pressure pulse |
| `vortex` | `[0,10]x[0,10]x[0,1]` | periodic | low-Mach isentropic vortex |
| `dm` | `[0,4]x[0,1]x[0,0.25]` | inflow/slip/transmissive/periodic | Woodward–Colella double Mach |

### Double Mach

Standard Woodward–Colella setup: a Mach-10 shock inclined 30° to the x-axis
impinges on the reflecting bottom wall at `x = 1/6`; `gamma = 1.4`. The
pre-shock state is `rho=1.4`, `p=1`, `u=v=0`. The oblique post-shock state has
normal velocity 8.25, i.e. `u = 8.25 cos(30°) = 7.1447`,
`v = -8.25 sin(30°) = -4.125`, `rho=8`, `p=116.5`. Initially the post-shock
state fills `x < 1/6 + y/sqrt(3)`.

Boundaries (implemented in `apply_double_mach_bc`):

- left: fixed post-shock inflow;
- bottom: fixed post-shock inflow for `x <= 1/6`, reflecting slip wall after;
- top: time-dependent oblique shock at
  `x_f(t) = 1/6 + (1 + 20 t)/sqrt(3)`, post-shock for `x < x_f` and pre-shock
  after;
- right: transmissive outflow; z: periodic.

The stage-time boundary is evaluated at the correct RK stage time (the time
integrators pass the stage time to the RHS). The wave structure is the
standard double-Mach pattern (incident shock, Mach stem, reflected shock,
triple point, slip line and Kelvin–Helmholtz roll-up). Quantitative comparison
against reference data (e.g. along the bottom wall at `t=0.2`) is still
pending.

### Taylor–Green vortex

```
u = sin(x) cos(y) cos(z)
v = -cos(x) sin(y) cos(z)
w = 0
rho = 1
p = 1/gamma + (cos(2x)+cos(2y))(cos(2z)+2)/16
```

Default `mu = 1/1600` (Re = 1600 on the unit box), `t_end = 1` for quick runs;
production TGV runs typically use `t_end = 10` and `n = 64..256`. The base
pressure `1/gamma` gives sound speed 1, so the default is `M ≈ 1`;
`--mach 0.1` sets `p0 = 1/(gamma M^2)` for the standard low-Mach reference
case (at the cost of ~10x more time steps).

### Acoustic pulse

```
rho = 1 + eps exp(-r^2/sigma^2)
p   = 1 + gamma eps exp(-r^2/sigma^2)
u = 0,  eps = 1e-3,  sigma = 0.1
```

on `[0,1]^3`, periodic. This checks that the acoustic branch is stable and
that kinetic energy remains small.

### Low-Mach isentropic vortex

Standard isentropic vortex with `beta = 0.1` and base velocity `u = 0.1` on a
`10 x 10` periodic domain; it is advected and should remain smooth.

### Double Mach (continued)

The default case uses `128 x 32 x 4` cells and runs to `t=0.2`; the resolution
study in the original paper uses `240 x 60` and `480 x 120`. The thin periodic
z direction and `w=0` initialization make the problem two-dimensional while
exercising the 3D solver.

## Diagnostics

Every snapshot and the final state print:

```
t, step, total mass, |momentum|, kinetic energy, internal energy,
total energy, max Mach
```

`--history FILE --diag-every N` additionally appends the same quantities as CSV
(`t,step,mass,momentum,kinetic,internal,total_energy,max_mach`) every `N`
steps, which gives the Taylor–Green kinetic-energy dissipation curve:

```bash
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 64 --tend 10 --diag-every 10 --history tgv_dissipation.csv

python3 scripts/tgv_dissipation.py tgv_dissipation.csv \
  --volume 248.0502134 --out tgv_eps.csv
```

The script differentiates the kinetic energy and reports the peak `-dK/dt` and
its time; `--volume` normalizes by the box volume (`(2 pi)^3` for TGV).

`--check` fails if any final cell has non-positive density/pressure or a
non-finite state. For an all-periodic inviscid run (`mu = 0`) it also requires
total energy to be conserved to `--energy-tol` (default `1e-10`); the acoustic
and low-Mach-vortex cases exercise this.

## VTK and restart

```bash
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 32 --tend 1.0 --output /tmp/tgv --plot-every 20 \
  --checkpoint /tmp/tgv_ck --checkpoint-every 100

./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 32 --restart /tmp/tgv_ck --tend 2.0
```

Snapshots are `<base>_NNNNNN.vti` (or `.pvti` + per-rank `.vti` in MPI).
Checkpoints are `<base>.rankNNNN.bin`, compatible with `read_checkpoint`.

## Short verification runs

```bash
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 16 --tend 0.05 --check
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case acoustic --n 16 --tend 0.05 --check
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case vortex --nx 16 --ny 16 --nz 4 --tend 0.05 --check
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case dm --nx 32 --ny 16 --nz 4 --tend 0.005 --cfl 0.2 --check
```

Observed on OpenMP (16³ unless noted):

| Case | steps | mass drift | final K | max Mach |
|---|---|---|---|---|
| TGV, t=0.05 | 1 | <1e-12 | 3.10004e+01 | 0.950 |
| acoustic, t=0.05 | 3 | <1e-12 | 8.08e-10 | 2.95e-04 |
| vortex, t=0.05 | 5 | <1e-12 | 5.00007e+01 | ~0.1 |
| double Mach, t=0.005 (32x16x4) | 5 | inflow/outflow | 3.403e+01 | 1.83 |
| double Mach, t=0.2 (256x64x4) | 653 | inflow/outflow | 2.264e+02 | 3.85 |

Mass is conserved to round-off in the periodic cases. The double-Mach mass
grows because post-shock inflow enters through the left, bottom (`x <= 1/6`),
and top boundaries.

## Multi-rank notes

- `BoxExchange` fills all 26 neighbor blocks (faces, edges, corners) and is
  covered by `saturn.box_exchange_mpi2/4`.
- Multi-rank all-periodic runs pass the entropy MMS, viscous-shear MMS, and
  TGV stability tests on 2 and 4 ranks.
- Multi-rank non-periodic boundaries use `GlobalExchange`, an all-gather
  fallback that fills all ghosts/BCs consistently; the double-Mach analytic
  inflow is applied on top of it. A 64x16x4 short run gives bitwise-identical
  mass, momentum, K, e, and max Mach on 1, 2, and 4 ranks
  (`saturn.compressible_double_mach_mpi2/mpi4`).
- All-periodic runs use the 26-neighbor `BoxExchange`.
- MPI is host-staged, as in `FaceExchange`. A fused neighbor exchange and
  GPU-aware MPI are later optimizations; `GlobalExchange` is O(N) and should be
  replaced by a corner-aware neighbor exchange for large runs.

## Remaining work

- Compare the double-Mach wave structure and wall pressure against reference
  data at `t=0.2`.
- Compare the Taylor–Green energy-dissipation curve from `--history` against
  reference data at production resolutions (64–256); the 64³ low-Mach
  comparison is in `docs/verification/tgv.md`.
- HDF5/ADIOS2 output and parallel checkpointing.
- GPU runs and performance tuning of reconstruction + flux fusion.
