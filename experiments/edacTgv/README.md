# EDAC vs projection: 2D Taylor–Green vortex prototype

**Status:** completed de-risk experiment (2026-10-26).
**Question:** can a pressure-free EDAC formulation replace the Poisson solve for
incompressible DNS without losing accuracy or performance?

## Test setup

- 2D periodic Taylor–Green vortex on `[0, 2π]²`, `ν = 0.01`, `t_end = 2`.
- Cell-centered finite differences, 2nd-order central operators for both
  methods. Skew-symmetric advection, SSP-RK3, explicit viscous term.
- **Projection reference:** FFT Poisson solve after every RK3 stage. This is a
  best-case Poisson solver on a single CPU (no MPI, no GMG iteration).
- **EDAC:** local pressure evolution, no global solve:
  ```
  ∂u/∂t + ∇·(u⊗u) = −∇p + ν∇²u
  ∂p/∂t + c² ∇·u  = ν_p ∇²p,     c = 1/M_c,  ν_p = c² τ
  τ = tau_factor · h / c
  ```
- Exact solution used for the initial condition and error norm:
  ```
  u =  sin(x) cos(y) e^{−2νt}
  v = −cos(x) sin(y) e^{−2νt}
  p =  (cos 2x + cos 2y)/4 · e^{−4νt}
  ```
- EDAC pressure initialized from the exact pressure. Initializing `p = 0`
  leaves a long-lived acoustic transient (see below), so a consistent initial
  pressure (or one initial projection) is required.

## Results

### Accuracy and cost

| Method | N | `M_c` | `τ` factor | Steps | Rel. L2 error | `‖∇·u‖₂` | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| projection | 64 | — | — | 40 | 3.21e-5 | 8.9e-16 | 0.12 s |
| EDAC | 64 | 0.10 | 0.01 | 448 | 7.53e-5 | 9.4e-5 | 0.84 s |
| EDAC | 64 | 0.05 | 0.01 | 855 | 3.61e-5 | 9.9e-6 | 1.51 s |
| projection | 128 | — | — | 80 | 8.03e-6 | 1.5e-15 | 0.61 s |
| EDAC | 128 | 0.10 | 0.01 | 895 | 9.53e-5 | 5.7e-5 | 3.45 s |
| EDAC | 128 | 0.05 | 0.01 | 1710 | 1.33e-5 | 1.4e-5 | 6.54 s |

**At `M_c = 0.05`, EDAC matches projection accuracy to within ~2× at N=128,
but takes 21× more steps and ~11× more wall time** on one CPU with an FFT
Poisson solve.

### Damping sensitivity (`N=64`, `M_c=0.1`, `p₀` exact)

| `tau_factor` | Rel. L2 error | `‖∇·u‖₂` | Steps |
|---:|---:|---:|---:|
| 0.001 | 1.61e-4 | 3.0e-5 | 448 |
| **0.01** | **7.53e-5** | **9.4e-5** | 448 |
| 0.05 | 7.62e-4 | 5.9e-4 | 448 |
| 0.1 | 1.66e-3 | 1.2e-3 | 448 |
| 0.5 | 8.53e-3 | 4.6e-3 | 448 |
| 1.0 | unstable | — | 632 |

The damping coefficient is the critical EDAC parameter. Too little damping
leaves acoustic noise; too much over-damps and destroys accuracy. It must be
calibrated per flow/resolution; `tau_factor = 0.01` was optimal here, not the
0.1 used in the initial sweep.

### Initial-pressure sensitivity (`N=64`, `M_c=0.1`)

| `p₀` | Rel. L2 error (best `τ`) |
|---|---:|
| exact | 7.5e-5 |
| zero | ~2e-2 |

Starting from `p = 0` is not acceptable for this flow: the missing initial
pressure gradient excites a slowly decaying acoustic transient. A production
EDAC solver needs a consistent initial pressure (one Poisson solve at startup,
or a spin-up phase).

### `M_c` scaling (`N=64`, `tau_factor=0.01`, `p₀` exact)

| `M_c` | Steps | Rel. L2 error | `‖∇·u‖₂` |
|---:|---:|---:|---:|
| 0.20 | 244 | 3.36e-4 | 9.9e-5 |
| 0.10 | 448 | 7.53e-5 | 9.4e-5 |
| 0.05 | 855 | 3.61e-5 | 9.9e-6 |
| 0.025 | 1670 | 4.41e-5 | 4.5e-5 |
| 0.0125 | 3300 | 2.71e-5 | 9.4e-6 |

- Error drops with `M_c` until it reaches the **spatial discretization floor**
  (~3e-5 at N=64).
- Steps scale as `1/M_c`, as expected from the acoustic CFL.
- With the standard damping scaling `ν_p = c² τ`, `τ ∝ h/c`, the divergence is
  `∇·u ≈ τ ∇²p ∝ M_c` at fixed `tau_factor`; it is **not** `O(M_c²)`.
  Choosing `τ ∝ M_c` (constant `ν_p`) changes the constant but not the
  fundamental step-count penalty.

## Conclusion

1. **EDAC is a real, local, explicit method** and can reproduce incompressible
   TGV results without a Poisson solve.
2. **It is not a free replacement for projection in DNS.** The acoustic CFL
   costs `1/M_c` more steps; matching projection accuracy required `M_c≈0.05`
   and ~20× more steps in this test.
3. **Projection + GMG remains the production default** for incompressible DNS.
   EDAC is worth keeping as an experimental path for extreme-scale runs where
   the Poisson solve dominates (>~95% of step time) and moderate accuracy is
   acceptable.
4. **If the goal is removing the global solve at all costs**, the alternatives
   are not better for high-order FVM DNS: artificial compressibility has the
   same CFL penalty; low-Mach preconditioning reintroduces an inner elliptic
   solve; vorticity/streamfunction formulations still need a global solve;
   LBM/SPH abandon the FVM/high-order requirement.
5. **The practical scaling lever is the Poisson solver itself**: GMG (O(N),
   local communication) and, for uniform periodic grids, FFT. Optimize those
   before accepting an order-of-magnitude step-count penalty.

## Reproduce

```bash
python3 experiments/edacTgv/edacTgv.py \
  --n-list 64 128 --mc-list 0.2 0.1 0.05 0.025 \
  --nu 0.01 --tend 2.0 --tau 0.01 --method both \
  --json experiments/edacTgv/results/sweep.json \
  --plot experiments/edacTgv/results/sweep.png
```

Single run:

```bash
python3 experiments/edacTgv/edacTgv.py --n 128 --nu 0.01 --tend 2 \
  --mc 0.05 --tau 0.01 --method both
```

## Caveats

- 2D, single CPU, FFT projection. A GMG projection would be slower per step, so
  the EDAC/projection wall-time ratio would improve somewhat; the step-count
  ratio would not.
- No MPI or GPU communication effects. The crossover at extreme scale must be
  measured on the target hardware, not inferred from this laptop.
- The damping parameter was tuned on this case. A production EDAC path needs
  auto-calibration and a robustness study across flows.
