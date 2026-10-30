# EDAC prototype: 2D Taylor–Green vortex

**Date:** 2026-10-29
**Status:** periodic 2D TGV verification of the C++ EDAC path; matches the
projection spatial error at `M_c=0.05` but needs ~12x more steps. Projection +
GMG/FFT remains the production path.
**Code:** `apps/edac_sim.cpp`, `src/numerics/incompressible/edac.hpp`

## Formulation

EDAC evolves a pseudo-pressure instead of solving a Poisson equation:

```
du/dt = -(u.grad)u + nu laplacian(u) - grad(p)
dp/dt = -c^2 div(u) + nu_p laplacian(p)

c = 1/M_c,   nu_p = c^2 tau,   tau = tau_factor h/c
```

The velocity is on a staggered MAC grid, pressure at cell centers, and all
boundaries are periodic. Time integration is SSP-RK3 on `(u,v,w,p)`. No global
solve or communication is required.

## Test case

Exact 2D Taylor–Green vortex on `[0,2pi]^2`, periodic in z:

```
u =  sin(x) cos(y) exp(-2 nu t)
v = -cos(x) sin(y) exp(-2 nu t)
p = (cos(2x) + cos(2y))/4 exp(-4 nu t)
```

`nu=0.01`, `t_end=1`, `tau_factor=0.01`, CFL 0.5. The exact pressure is used
for the initial condition; the `--zero-p` option starts from `p=0` to show the
acoustic transient.

## Results versus projection

Projection uses the FFT pressure solver (same MAC discretization):

| Method | N | `M_c` | steps | L2 velocity | L2 divergence |
|---|---|---|---|---|---|
| projection | 32 | — | 17 | 1.901e-03 | ~8e-16 |
| projection | 64 | — | 34 | 4.756e-04 | ~2e-15 |
| EDAC | 32 | 0.05 | 214 | 1.914e-03 | 2.49e-05 |
| EDAC | 64 | 0.05 | 428 | 4.800e-04 | 1.53e-05 |

At `M_c=0.05` the EDAC velocity error matches projection to within ~1% at both
resolutions, while the step count is 12.6x larger (the acoustic CFL). EDAC's
divergence is `O(M_c)` and is not driven to round-off.

### `M_c` sweep (`N=32`, `t=0.5`)

| `M_c` | steps | L2 velocity | L2 divergence |
|---|---|---|---|
| 0.20 | 31 | 1.929e-03 | 4.94e-04 |
| 0.10 | 56 | 1.940e-03 | 3.27e-04 |
| 0.05 | 107 | 1.940e-03 | 6.45e-05 |
| 0.025 | 209 | 1.937e-03 | 1.57e-05 |

The velocity error sits on the spatial floor for this short run; the divergence
falls roughly linearly with `M_c`, and the step count scales as `1/M_c`, as in
the Python prototype (`experiments/edacTgv/`).

### Initial-pressure sensitivity

Starting from `p=0` (`--zero-p`, `N=32`, `M_c=0.1`, `t=0.5`) gives L2 velocity
`7.40e-03` and divergence `2.38e-02`, versus `1.94e-03` and `3.27e-04` with the
exact initial pressure. A production EDAC path needs one initial projection or
a spin-up phase.

## GPU cost crossover (Quadro P600)

Measured on the CUDA build, `N=64`, `t_end=1`, `nu=0.01`:

| Method | steps | L2 velocity | L2 divergence | wall time |
|---|---|---|---|---|
| projection + FFT | 34 | 4.756e-04 | ~1e-15 | **0.227 s** |
| projection + MG | 34 | 4.756e-04 | 2.7e-12 | 3.615 s |
| EDAC (`M_c=0.05`) | 428 | 4.800e-04 | 1.53e-05 | 1.529 s |

EDAC is **6.7x slower** than projection + FFT on this GPU. The gap is smaller
than on the CPU (~12x) because EDAC is fully local while the current FFT path
copies the field to the host; a device FFT or distributed FFT would widen
projection's lead further. The Poisson fraction would have to exceed roughly
85% of the step time before EDAC wins, which is not the case here. Projection +
GMG/FFT remains the production default.

## Commands

```bash
./build/cpu-openmp/apps/saturn_edac_sim \
  --n-list 32 64 --mc 0.05 --tau 0.01 --tend 1 --check

ctest --test-dir build/cpu-openmp -R saturn.edac_tgv --output-on-failure
```

## Conclusion

- EDAC is implemented and reproduces the projection velocity error at
  `M_c≈0.05` without any Poisson solve.
- The cost is ~`1/M_c` more steps; in this test ~12.6x at `M_c=0.05`.
- Divergence is `O(M_c)`, not `O(M_c^2)`.
- Projection + GMG/FFT remains the production default. EDAC remains an
  experimental extreme-scale path, as decided in `PLAN.md` Appendix C.

## Limitations and next steps

- Periodic, constant-density 2D TGV only; walls, variable density, and 3D
  turbulence are not covered.
- The pressure field is not compared quantitatively yet; only velocity and
  divergence are diagnostics.
- Auto-calibration of `tau` and `M_c` from `h`, `U`, and `nu` is not
  implemented.
- The cost crossover requires timing the Poisson fraction on the target GPU
  hardware; that measurement is still pending.
