# Verification: ideal-gas Riemann solvers

**Date:** 2026-10-27
**Status:** unit tests pass on CPU (Kokkos Serial + OpenMP); device-callable.
Full shock-tube/MMS validation in a time-dependent solver is the next step.
**Code:** `src/numerics/riemann.hpp`, `tests/test_riemann.cpp`

## Purpose

Provide the inviscid face flux for the compressible Euler equations behind a
single interface:

```cpp
saturn::riemann::Flux flux(
    saturn::RiemannSolver::Hllc, left, right, gamma);
```

The states are in face-normal coordinates:

```
U = [rho, rho u, rho v, rho w, E]
F = [rho u, rho u^2 + p, rho u v, rho u w, u (E + p)]
p = (gamma - 1) (E - 0.5 rho (u^2 + v^2 + w^2))
```

`u` is normal to the face and `v, w` are tangential. The caller rotates the
state into this frame and rotates the flux back, which keeps one implementation
for x, y, and z sweeps. Ghost cells and reconstruction are outside this module.

## Solvers

| Solver | Description |
|---|---|
| `llf` | Local Lax-Friedrichs / Rusanov; robust fallback |
| `hllc` | HLLC with Roe-averaged wave-speed bounds |
| `hllc_lm` | HLLC with a low-Mach correction |
| `ausm_plus_up` | AUSM+-up (Liou 2006) for all-speed flows |

### HLLC

Wave speeds use the Roe average for tighter bounds than the plain Davis
estimate:

```
s_L = min(u_L - c_L, u_Roe - c_Roe)
s_R = max(u_R + c_R, u_Roe + c_Roe)
s_* = [p_R - p_L + rho_L u_L (s_L - u_L) - rho_R u_R (s_R - u_R)]
      / [rho_L (s_L - u_L) - rho_R (s_R - u_R)]
```

The intermediate state on side K is

```
U*_K = rho_K (s_K - u_K)/(s_K - s_*) *
       [1, s_*, v_K, w_K,
        E_K/rho_K + (s_* - u_K)(s_* + p_K/(rho_K (s_K - u_K)))]
```

and `F*_K = F_K + s_K (U*_K - U_K)`. The implementation rejects an intermediate
state that is not positive or not bracketed by the wave speeds and falls back
to LLF. LLF is also the fallback for non-positive input states.

### HLLC-low-Mach

The acoustic part of HLLC over-dissipates pressure waves as `M -> 0`. The
low-Mach variant blends the HLLC flux with the central flux,

```
F = phi F_HLLC + (1 - phi) F_central
phi = sin(min(1, Ma_local / Ma_limit) * pi/2)
Ma_local = max(|u_L|/c_L, |u_R|/c_R)
```

with `Ma_limit = 0.1`. Above the limit `phi = 1` and the scheme is exactly
HLLC. A pressure-jump/velocity-jump sensor (`|p_R-p_L| > 0.1 p_avg` or
`|u_R-u_L| > 0.1 c_avg`) also restores full HLLC, so shocks and strong
expansions keep their upwinding. The blend is symmetric under a mirror of the
Riemann problem.

### AUSM+-up

Uses the standard split Mach polynomials

```
M+(M) = 0.5 (M + |M|)
M-(M) = 0.5 (M - |M|)

M+_(2)(M) =  0.25 (M + 1)^2        (|M| < 1)
M-_(2)(M) = -0.25 (M - 1)^2        (|M| < 1)

P+_(5)(M) = M+_(2) [( 2 - M) - 16 alpha M M-_(2)]   (|M| < 1)
P-_(5)(M) = M-_(2) [(-2 - M) + 16 alpha M M+_(2)]   (|M| < 1)
alpha = 3/16 (-4 + 5 f_a^2)
```

with the AUSM+-up interface Mach number and pressure

```
M_f = M+_(2)(M_L) + M-_(2)(M_R) + M_p
p_f = P+_(5)(M_L) p_L + P-_(5)(M_R) p_R + p_u

M_p = -(K_p / f_a) max(1 - sigma Mbar^2, 0) (p_R - p_L) / (rho_f a_f^2)
p_u = -K_u P+_(5)(M_L) P-_(5)(M_R) (rho_L + rho_R) (f_a a_f) (u_R - u_L)

a_f   = 0.5 (c_L + c_R)
rho_f = 0.5 (rho_L + rho_R)
Mbar^2 = (u_L^2 + u_R^2) / (2 a_f^2)
f_a = M0 (2 - M0),   M0^2 = min(1, max(Mbar^2, M_cut^2))

K_p = 0.25, K_u = 0.75, sigma = 1
```

`M_cut` is the free-stream Mach number. It defaults to 0.1; for shock-dominated
problems a larger value (e.g. 1) gives a less diffusive pressure-diffusion
term. The mass flux is

```
m_dot = a_f M_f * (rho_L if M_f > 0 else rho_R)
F = m_dot [1, u, v, w, H]_upwind + [0, p_f, 0, 0, 0]
```

## Test results

`ctest -R saturn.riemann` checks:

- **Uniform consistency:** for four states and all four solvers, the flux of
  identical left/right states equals the physical flux to 1e-12.
- **Stationary contact:** `rho = 1/3`, `u = 0`, `p = 1` gives
  `[0, p, 0, 0, 0]` for HLLC, HLLC-LM, and AUSM+-up. (LLF diffuses it by
  construction and is excluded.)
- **Moving contact:** `u = 0.5`, `p = 1`, `rho = 1/3` gives the exact upwind
  flux `[0.5, 1.25, 0, 0, 1.8125]` for HLLC, HLLC-LM, and AUSM+-up.
- **Mirror symmetry:** swapping left/right and reflecting the normal velocity
  maps the flux to `(-F_rho, F_mom_n, -F_mom_t1, -F_mom_t2, -F_E)` to 1e-12,
  including the low-Mach blend.
- **Sod state:** all solvers return finite rightward mass and momentum fluxes;
  the HLLC-LM pressure sensor makes it identical to HLLC to 1e-14.
- **Low-Mach dissipation:** for a pressure perturbation at `M = 0.01`, the
  distance to the central flux is `1.122e-03` for HLLC and `1.485e-04` for
  HLLC-LM, i.e. the correction reduces the acoustic smearing as intended.
- **Device kernel:** all four solvers run in a Kokkos kernel and match the host
  results.

## Remaining work

- A 1D/3D time-dependent Euler solver with reconstruction + SSP-RK3 to compare
  Sod against the exact solution (the current tests exercise the flux, not the
  full update).
- Double Mach reflection, low-Mach vortex, and acoustic-pulse cases.
- Tune `M_cut` and the HLLC-LM sensor against those cases; document the
  measured dissipation.
- Viscous terms (compressible Navier-Stokes) after the inviscid path is gated.
