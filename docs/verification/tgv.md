# Taylor–Green vortex: dissipation diagnostics

**Date:** 2026-10-28
**Status:** stable long-time run at `n=64` with WENO5-Z; `central4` is unstable
at this resolution. This is a diagnostic/sanity check, not yet a formal
low-Mach benchmark comparison.
**Code:** `apps/compressible_sim.cpp`, `scripts/tgv_dissipation.py`

## Setup

The `tgv` case uses

```
u = sin(x) cos(y) cos(z)
v = -cos(x) sin(y) cos(z)
w = 0
rho = 1
p = 1/gamma + (cos(2x)+cos(2y))(cos(2z)+2)/16
```

on `[0,2pi]^3`, with `mu = 1/1600` (Re = 1600 for unit velocity and length) and
periodic boundaries. With `p0 = 1/gamma` and `rho = 1`, the sound speed is 1,
so the initial velocity amplitude corresponds to **M ≈ 1** (a compressible
Taylor–Green vortex, not the standard M = 0.1 incompressible benchmark).

## Commands

```bash
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 64 --tend 10 --cfl 0.3 --scheme weno5z \
  --diag-every 10 --history tgv_n64_weno5z_history.csv --check

python3 scripts/tgv_dissipation.py tgv_n64_weno5z_history.csv \
  --volume 248.0502134 --out tgv_n64_weno5z_dissipation.csv
```

Recorded data:
`benchmarks/results/tgv/tgv_n64_weno5z_history.csv` and
`benchmarks/results/tgv/tgv_n64_weno5z_dissipation.csv`.

## Results

OpenMP, 8 threads, `n=64`, WENO5-Z + HLLC, SSP-RK3, CFL 0.3:

| Quantity | Value |
|---|---|
| steps to `t=10` | 674 |
| wall time | 253 s |
| initial `K/V` | 1.250000e-01 |
| final `K/V` | 6.941211e-02 |
| peak `-dK/dt / V` | **1.278861e-02** |
| peak time | **t = 8.489** |
| mass drift | 0 |
| total-energy drift | 7.0e-14 |
| final max Mach | 0.742 |

The kinetic energy is flat during the initial pressure-adjustment phase, decays
through transition, and peaks in dissipation at `t ≈ 8.5`. Total energy is
conserved to round-off, as expected for a periodic viscous run with a
conservative energy flux.

## Resolution and scheme sensitivity

The same case with `central4`, CFL 0.4, becomes unstable at `t ≈ 6.67`
(kinetic energy jumps from 28.5 to 955 while the total energy is 474, so the
internal energy goes negative; the run then produces NaNs). This is the
expected behaviour of a non-dissipative central scheme on an under-resolved
transitional flow, not a mass/energy conservation failure: mass stays exactly
constant until the state becomes invalid.

WENO5-Z remains stable to `t=10`. `central4` is appropriate for smooth-flow
MMS but needs a filter or a higher resolution for long TGV runs.

The low-Mach option is verified separately with a short `n=32`, `M=0.1` run
(`--mach 0.1`, WENO5-Z, 187 steps to `t=1`): max Mach 0.0986, mass drift 0,
total-energy drift 1.7e-14. A full `n=64`, `t=10` low-Mach run needs roughly
10x more steps than the M≈1 case (about 30 minutes on 8 OpenMP threads).

## Comparison with reference data

Published low-Mach (`M → 0`) Re = 1600 reference solutions place the
dissipation peak near `1.25e-2` at `t ≈ 9` (e.g. the NASA TGV workshop
reference; the 64³ grid is typically ~10% off the peak). The present M ≈ 1 run
gives `1.279e-2` at `t = 8.49`; the closeness is encouraging but **not** a
formal validation because the Mach number differs.

Next steps for a quantitative TGV gate:

1. run at `--mach 0.1` (the option sets `p0 = 1/(gamma M^2)`; sound speed 10
   means roughly 10x more steps than the M≈1 run);
2. compare `K(t)` and `-dK/dt` against tabulated reference data at
   `64^3/128^3/256^3`;
3. repeat with `central6`/TENO6 at fixed resolution to separate scheme
   dissipation from physical dissipation.
