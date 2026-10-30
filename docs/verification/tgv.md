# Taylor–Green vortex: dissipation diagnostics

**Date:** 2026-10-28
**Status:** stable long-time runs at `n=64` with WENO5-Z, including the
standard low-Mach (`M=0.1`) setup; `central4` is unstable at this resolution.
A 64³/128³/256³ convergence study against tabulated reference data is still
pending.
**Code:** `apps/compressible_sim.cpp`, `scripts/tgvDissipation.py`

## Setup

The `tgv` case uses

```
u = sin(x) cos(y) cos(z)
v = -cos(x) sin(y) cos(z)
w = 0
rho = 1
p = p0 + (cos(2x)+cos(2y))(cos(2z)+2)/16
```

on `[0,2pi]^3`, with `mu = 1/1600` (Re = 1600 for unit velocity and length) and
periodic boundaries. The base pressure is `p0 = 1/(gamma M^2)`:

- default `--mach 1`: `p0 = 1/gamma`, sound speed 1, initial max Mach ≈ 1;
- `--mach 0.1`: `p0 = 1/(gamma M^2) ≈ 71.43`, sound speed 10, initial max
  Mach ≈ 0.1 (the standard low-Mach reference setup).

## Commands

```bash
# M ~ 1 quick compressible run (674 steps)
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 64 --tend 10 --cfl 0.3 --scheme weno5z \
  --diag-every 10 --history tgv_n64_weno5z_history.csv --check

# Low-Mach reference setup (3686 steps, ~22 min on 8 threads)
./build/cpu-openmp/apps/saturn_compressible_sim \
  --case tgv --n 64 --mach 0.1 --tend 10 --cfl 0.3 --scheme weno5z \
  --diag-every 10 --history tgv_n64_m0.1_weno5z_history.csv --check

python3 scripts/tgvDissipation.py <history.csv> \
  --volume 248.0502134 --out <dissipation.csv>
```

Recorded data in `benchmarks/results/tgv/`:

| History | Processed |
|---|---|
| `tgv_n64_weno5z_history.csv` | `tgv_n64_weno5z_dissipation.csv` |
| `tgv_n64_m0.1_weno5z_history.csv` | `tgv_n64_m0.1_weno5z_dissipation.csv` |

## Results

OpenMP, 8 threads, `n=64`, WENO5-Z + HLLC, SSP-RK3, CFL 0.3:

| Quantity | M ≈ 1 | M = 0.1 |
|---|---|---|
| steps to `t=10` | 674 | 3686 |
| wall time | 253 s | 1322 s |
| initial `K/V` | 1.250000e-01 | 1.250000e-01 |
| final `K/V` | 6.941211e-02 | 5.974016e-02 |
| peak `-dK/dt / V` | **1.278861e-02** | **1.195450e-02** |
| peak time | 8.489 | 7.042 |
| mass drift | 0 | 0 |
| total-energy drift | 7.0e-14 | 1.9e-13 |
| final max Mach | 0.742 | 0.060 |

In both runs the kinetic energy is flat during the initial pressure-adjustment
phase, decays through transition, and peaks in dissipation near `t = 7–9`.
Total energy is conserved to round-off, as expected for a periodic viscous run
with a conservative energy flux.

## Resolution and scheme sensitivity

The same case with `central4`, CFL 0.4, becomes unstable at `t ≈ 6.67`
(kinetic energy jumps from 28.5 to 955 while the total energy is 474, so the
internal energy goes negative; the run then produces NaNs). This is the
expected behaviour of a non-dissipative central scheme on an under-resolved
transitional flow, not a mass/energy conservation failure: mass stays exactly
constant until the state becomes invalid.

WENO5-Z remains stable to `t=10` for both Mach numbers. `central4` is
appropriate for smooth-flow MMS but needs a filter or a higher resolution for
long TGV runs.

## Comparison with reference data

Published low-Mach (`M → 0`) Re = 1600 reference solutions place the
dissipation peak near `1.25e-2` at `t ≈ 9` (NASA TGV workshop reference; the
64³ grid is typically ~10% off the peak, with the peak shifted early).

The new low-Mach run gives `1.195e-2` at `t = 7.04`: the peak magnitude is
**4.4% below** the reference, while the peak time is about two time units early
— the expected signature of numerical dissipation on a 64³ grid. The M ≈ 1 run
happens to land closer in time (`8.49`) but is a different physical case and is
not used for the quantitative comparison.

Next steps for a quantitative TGV gate:

1. run `64³/128³/256³` at `M=0.1` and check convergence of the peak value,
   peak time, and `K(t)`;
2. compare against tabulated reference data (not just the peak);
3. repeat with `central6`/TENO6 at fixed resolution to separate scheme
   dissipation from physical dissipation.
