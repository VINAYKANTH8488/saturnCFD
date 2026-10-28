# 3D compressible driver and cases

**Date:** 2026-10-27
**Status:** CPU Serial/OpenMP; single-rank non-periodic cases; multi-rank
all-periodic runs use the full 3D halo exchange, verified on 2 and 4 ranks.
CUDA validation pending.
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
| `dm` | `[0,4]x[0,1]x[0,0.25]` | transmissive/slip/periodic | double Mach setup; approximate top BC |

The `dm` case is an initial-condition/stability setup, not yet a validated
double-Mach benchmark: the top boundary uses a simple transmissive condition
instead of the exact post-shock inflow profile.

### Taylor–Green vortex

```
u = sin(x) cos(y) cos(z)
v = -cos(x) sin(y) cos(z)
w = 0
rho = 1
p = 1/gamma + (cos(2x)+cos(2y))(cos(2z)+2)/16
```

Default `mu = 1/1600` (Re = 1600 on the unit box), `t_end = 1` for quick runs;
production TGV runs typically use `t_end = 10` and `n = 64..256`.

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

### Double Mach

Post-shock state (`rho=8`, `p=116.5`, `u=8.25`) for `x < 1/6`, pre-shock
(`rho=1.4`, `p=1`, `u=0`) otherwise; `gamma=1.4`. Bottom is a slip wall, z is
periodic, other sides are transmissive.

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
```

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
| double Mach, t=0.005 (32x16x4) | 6 | inflow/outflow | 1.112e+01 | 1.83 |

Mass is conserved to round-off in the periodic cases. The double-Mach mass
changes because the left/right boundaries are transmissive.

## Multi-rank notes

- `BoxExchange` fills all 26 neighbor blocks (faces, edges, corners) and is
  covered by `saturn.box_exchange_mpi2/4`.
- Multi-rank all-periodic runs pass the entropy MMS, viscous-shear MMS, and
  TGV stability tests on 2 and 4 ranks.
- Multi-rank non-periodic boundaries use `GlobalExchange`, an all-gather
  fallback that fills all ghosts/BCs consistently; the double-Mach case gives
  the same result on 2/4 ranks as on one rank.
- All-periodic runs use the 26-neighbor `BoxExchange`.
- MPI is host-staged, as in `FaceExchange`. A fused neighbor exchange and
  GPU-aware MPI are later optimizations; `GlobalExchange` is O(N) and should be
  replaced by a corner-aware neighbor exchange for large runs.

## Remaining work

- Validate the double-Mach top boundary and compare against reference data.
- Compare the Taylor–Green energy-dissipation curve from `--history` against
  reference data at production resolutions (64–256).
- HDF5/ADIOS2 output and parallel checkpointing.
- Physical inflow/outflow/wall boundary conditions for multi-rank runs.
- GPU runs and performance tuning of reconstruction + flux fusion.
