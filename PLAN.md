# saturnCFD — Design & Execution Plan

> A performance-portable, block-structured, high-order finite-volume DNS code
> for CPU and multiple GPUs, built on Kokkos. Basilisk-inspired in physics and
> workflow, but designed for exascale-era hardware from the start.

**Status:** planning (no code yet)
**Owner:** _(you)_
**Last updated:** 2026-10-26

**Decisions locked on 2026-10-26:**

- **First production solver:** compressible Navier–Stokes (density-based,
  high-order).
- **Incompressible path:** projection + GMG is the production default. EDAC
  (pressure-free) is implemented as an experimental extreme-scale path; the 2D
  prototype showed a ~20× step-count penalty for DNS-level accuracy.
- **AMR:** uniform first, AMR-ready abstractions, block AMR in Phase 5.
- **Backends:** CUDA, HIP, and SYCL all buildable from day 1; CUDA is the
  first performance-tuned target.
- **Dependencies:** Kokkos, MPI, HDF5, Catch2, toml++.
- **Geometry:** Cartesian; immersed boundaries later.
- **License:** Apache-2.0 proposed (pending confirmation).

---

## 1. Purpose and scope

saturnCFD is a **generic CFD framework** for direct numerical simulation (DNS)
of incompressible and compressible flows, with multiphase (VOF) capability, that:

1. Runs the **same C++ source** on Serial, OpenMP, CUDA, HIP, and SYCL by
   changing only the CMake configuration.
2. Scales from a laptop to **many GPUs and many CPU nodes** via MPI + Kokkos.
3. Uses **high-order finite-volume** discretizations suitable for DNS
   (4th–6th order in smooth regions, shock-capturing where needed).
4. Is **generic**: dimension-generic, physics-generic, boundary-condition-generic,
   and backend-generic; adding a new equation set should not touch the grid,
   halo, or linear-solver code.
5. Is **verifiable and benchmarkable**: every scheme has an MMS/convergence test,
   every solver has a validation case, and every performance claim has a
   reproducible scaling script.
6. Provides **three pressure-coupling strategies** — compressible NS for
   acoustics and shocks, projection + GMG as the production incompressible
   solver, and EDAC as an experimental pressure-free path — selected per case
   and compared at fixed accuracy.

### 1.1 Non-goals (for v1)

- Unstructured body-fitted meshes and general polyhedral FVM.
- RANS/LES turbulence models, combustion chemistry, radiation.
- GUI, CAD import, or a full pre/post-processing suite.
- Bitwise reproducibility across backends (float contraction differs).
- Production multiphase at the fidelity of 20 years of Basilisk development.

### 1.2 Assumptions

- 3D is the production target; 2D and 1D exist for fast verification.
- Double precision by default (`float` only as an explicit experiment).
- Non-dimensional equations by default; dimensional input via a units layer.
- Cartesian cells; complex geometry via immersed boundaries later.
- Ideal-gas EOS for compressible; two-phase incompressible with VOF.

---

## 2. What we borrow from Basilisk, and what we change

| Aspect | Basilisk | saturnCFD |
|---|---|---|
| Grid | Per-cell quadtree/octree | **Block-structured AMR** (AMReX-style boxes) |
| Data layout | Tree of cells, pointer chasing | Contiguous `Kokkos::View`, fixed-size boxes |
| Parallelism | MPI + OpenMP, host-centric | **MPI + Kokkos** (CUDA/HIP/SYCL/OpenMP) |
| Order | 2nd order | **4th–6th order** FVM + WENO/TENO for shocks |
| Multiphase | Geometric VOF, height functions | Geometric VOF + balanced-force CSF (later CLSVOF) |
| Pressure | Multigrid, approximate projection | **Three strategies:** EDAC (no Poisson), projection + GMG, low-Mach preconditioned compressible |
| Adaptivity | Wavelet error estimation | Gradient/wavelet tagging + Berger–Rigoutsos clustering |
| Philosophy | Concise, event-driven, single source | **Layered, generic, test-driven, GPU-first** |

**Why not copy the octree?** A per-cell tree is the worst possible data
structure for GPUs: divergent traversal, no coalescing, pointer indirection,
irregular work. Block-structured AMR gives 95% of the adaptivity benefit with
regular stencils inside blocks, which is what makes GPU performance possible.

---

## 3. Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────┐
│ applications: tgv, channel, cavity, sod, rti, dam_break, bubble  │
├──────────────────────────────────────────────────────────────────┤
│ drivers: time loop, subcycling, diagnostics, restart, benchmarks │
├──────────────────────────────────────────────────────────────────┤
│ solvers: EDAC | compressible | projection | VOF | advection      │
│          (each owns its time integrator and pressure coupler)    │
├──────────────────────────────────────────────────────────────────┤
│ numerics: reconstruction | Riemann | limiters | operators        │
│           (stateless, device-callable, templated on Model)       │
├──────────────────────────────────────────────────────────────────┤
│ physics: EOS | transport | surface tension | body force | IBM    │
├──────────────────────────────────────────────────────────────────┤
│ mesh: Box | BoxArray | Patch | Level | Distribution | SFC | BC   │
│       halo exchange | coarse-fine interpolation | refluxing      │
├──────────────────────────────────────────────────────────────────┤
│ core: Kokkos views | config | logging | timers | errors | units  │
└──────────────────────────────────────────────────────────────────┘
```

**Dependency rule:** arrows point downward only. `numerics` never calls MPI;
`mesh` never knows about physics; `solvers` compose the rest. Device code has
no virtual functions, no exceptions, no dynamic allocation.

**Data flow per stage:** `State` (named fields on patches) → operator kernels
(read ghosts, write fluxes/rhs) → `State` update → halo fill → next stage.
The `State` is the only thing that crosses layer boundaries.

---

## 4. Grid: the most important decision

### 4.1 Options considered

| Grid | GPU efficiency | High-order | AMR memory | Complex geom. | Implementation cost |
|---|---|---|---|---|---|
| Uniform Cartesian | ★★★★★ | ★★★★★ | none | ★ | ★ |
| Per-cell octree (Basilisk) | ★★ | ★★★ | ★★★★★ | ★★★ | ★★★★ |
| **Block-structured AMR** | ★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★ |
| Unstructured polyhedral | ★★ | ★★ | n/a | ★★★★★ | ★★★★★ |
| Cartesian + immersed boundary | ★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★ |

### 4.2 Recommendation: block-structured Cartesian AMR

**Use an AMReX-style `BoxArray` of non-overlapping rectangular boxes.**

- Every box has **uniform spacing**; resolution changes only between levels.
- Refinement ratio `r = 2` (optionally 4), with **2:1 level balance**.
- Cells are **cell-centered**; fields carry `G` ghost layers, where `G` is the
  stencil radius (2 for 4th order, 3 for WENO5/6th order).
- **Uniform DNS is just a single level** with large boxes — no AMR overhead.
- AMR boxes are fixed-size (default `32³` or `64³`) so that allocation,
  halo exchange, and kernel launch geometry are regular.
- Box IDs are **Morton/Z-order keys**; load balancing is space-filling-curve
  based.
- Optional later: immersed boundary method (IBM) for complex geometry on the
  same grid, so we never need a body-fitted mesh.

**Why this beats the alternatives for this project:**

1. It preserves the **regular stencil** that makes high-order FVM and GPU
   kernels fast.
2. It gives **local refinement** where VOF interfaces, shear layers, and
   shocks need it, without refining the whole domain.
3. It is the proven design for GPU AMR (AMReX, FLASH-X, Castro, incflo).
4. It keeps a clean path to **conservative refluxing** and **subcycling**.
5. It can degrade gracefully to a uniform grid for baseline DNS benchmarks.

### 4.3 Two operating modes, one mesh abstraction

| Mode | Decomposition | Use |
|---|---|---|
| `Uniform` | 3D Cartesian process grid (pencils), one large patch per rank | DNS benchmarks, scaling, TGV/channel |
| `Amr` | SFC distribution of fixed boxes over ranks | multiphase, shocks, local refinement |

Both produce a `BoxArray` + `Distribution`; operators only see boxes and
ghosts. The uniform mode uses a pencil decomposition because it minimizes
halo surface area; AMR uses SFC because box counts are dynamic.

### 4.4 Memory layout

```cpp
// One patch, one field set. i is fastest -> coalesced on GPU.
// data(var, k, j, i), extents [NVAR][nz+2G][ny+2G][nx+2G]
using PatchData = Kokkos::View<double****, Kokkos::LayoutRight, MemorySpace>;
```

- **Variable-major SoA** (`var` is the slowest index) so each 3D field is
  contiguous and stencils stream through memory.
- **One allocation per patch** (not per variable) to reduce allocator pressure.
- Ghost cells are stored **inside** the patch (padded), so kernels need no
  branch for interior vs boundary; only fill ghosts before use.
- Ghost overhead: `(B+2G)³/B³`. With `B=32, G=3` it is 1.67×; with `B=64` it is
  1.30×. Use `B=64` at level 0 and `B=32` on refined levels.
- For uniform runs with `G=3` and `B=128`, overhead is only 1.15×.
- **Do not** store a `std::vector<Patch>` of separate views in the hot path if
  it causes pointer chasing; use a flat `View<PatchData*>` of patch handles
  (device-copyable struct of views) or group patches per rank.

### 4.5 Domain decomposition and load balance

**Uniform mode**
- `MPI_Dims_create` + `MPI_Cart_create` for a 3D process grid.
- Face-only halo exchange; edge/corner ghosts are unnecessary for
  dimension-by-dimension stencils (this saves ~2× communication).
- Optional pencil–pencil transpose for FFT pressure solves.

**AMR mode**
- Compute a Morton key per box; sort boxes by key on device (`Kokkos::sort`).
- Weight each box by `cells × level_cost` (fine levels cost more per cell due
  to smaller `dt` and more halo work).
- Partition the sorted sequence into contiguous ranges with roughly equal
  weight (`Kokkos::parallel_scan`).
- Rebalance at every regrid; optionally every `N` steps for dynamic loads.
- Target **≥ 10–50 boxes per rank** so each GPU has enough parallelism; avoid
  one-box-per-rank configurations.

### 4.6 Halo exchange design

Per level, in order:

1. **Intra-level:** intersect boxes, pack faces into contiguous buffers,
   `MPI_Isend/Irecv`, unpack.
2. **Coarse→fine:** fill fine ghosts by high-order conservative interpolation
   from the coarse level (limited slopes).
3. **Fine→coarse:** average fine state onto coarse cells under fine boxes, and
   **reflux** coarse fluxes at coarse–fine interfaces to preserve conservation.
4. **Physical BCs** on domain boundaries (see §6.7).

Implementation notes:

- Exchange **faces only**, all variables in one message per direction/neighbor.
- Pack/unpack are Kokkos kernels; buffers are persistent per patch/neighbor.
- Overlap: launch pack → `MPI_Isend` → compute interior → `MPI_Wait` → unpack.
  This requires operators to separate "interior box" from "boundary box".
- GPU-aware MPI when available (`MPIX_Query_cuda_support`); otherwise stage
  through pinned host memory. Make this a runtime option, benchmark both.
- Use `MPI_Ineighbor_alltoallw` for the uniform Cartesian case.

---

## 5. Numerics

### 5.1 Equation sets (physics models)

| Model | Variables | Status |
|---|---|---|
| `ScalarAdvectionDiffusion` | `c` (+ optional `N` scalars) | v0 — verification workhorse |
| `CompressibleNS` | `ρ,ρu,ρv,ρw,E` (+ species) | v1 — first solver, shocks/acoustics |
| `AllMachEDAC` | `u,v,w,p` (+ `c`) | v2 — experimental pressure-free |
| `ProjectionIncompressible` | `u,v,w,p` (collocated) | v2 — reference/fallback |
| `VariableDensityVOF` | `u,v,w,p,c` | v3 — Basilisk parity |
| `LowMachReacting` | `u,v,w,p,T,Y` | later |

Incompressible one-fluid formulation (v3):

```
∇·u = 0
∂u/∂t + ∇·(u⊗u) = −∇p/ρ + ∇·(ν(∇u+∇uᵀ)) + g + f_σ/ρ
∂c/∂t + ∇·(cu) = 0
ρ = c ρ₁ + (1−c) ρ₂,   1/ρ = c/ρ₁ + (1−c)/ρ₂
f_σ = σ κ ∇c            (balanced-force CSF)
```

### 5.2 Spatial discretization (FVM)

Cell-centered finite volume; face fluxes from reconstruction.

| Term | Smooth DNS | Shock-capturing |
|---|---|---|
| Advection | 4th-order central / 6th-order compact reconstruction | WENO5-Z or TENO6 |
| Riemann | not needed (central) | HLLC (default), LLF fallback |
| Viscous | 4th-order central face gradients | same |
| Pressure gradient | 4th-order central; Rhie–Chow (projection) or pressure evolution (EDAC) | same |
| VOF | geometric PLIC, directionally split | same |

Rules:

- **Dimension-by-dimension** reconstruction: simpler, cheaper, and adequate for
  Cartesian grids. Avoid full multidimensional WENO in v1.
- **Skew-symmetric advection** splitting
  `½(∇·(uφ) + u·∇φ)` for kinetic-energy conservation in smooth incompressible
  flow. Add a `--energy-conserving` switch and verify on TGV.
- **Rhie–Chow** interpolation for the collocated velocity–pressure coupling to
  suppress checkerboarding, consistent with the pressure gradient operator.
  (Only in the projection path; EDAC does not need it.)
- **All-Mach consistency:** standard HLLC becomes excessively diffusive as the
  Mach number `M → 0`. Use a low-Mach-corrected Riemann solver (HLLC-low-Mach
  or AUSM⁺-up) for compressible flows; verify with a low-Mach vortex and an
  acoustic pulse. For the incompressible limit, use projection + GMG for
  production; EDAC (§5.4) is an experimental pressure-free alternative, not a
  way to make a compressible solver low-Mach for free.
- **Balanced-force** surface tension: use the *same* discrete gradient for
  `∇p` and `σκ∇c` so a static bubble has zero spurious currents to machine
  precision.
- **VOF:** PLIC reconstruction from cell-centered `c` and a least-squares
  normal; Strang-split directional sweeps; conservative flux; bounded
  redistribution. Curvature from height functions (Basilisk-style) initially,
  CLSVOF later if spurious currents are too high.
- **AMR order caveat:** coarse–fine interpolation and refluxing must be at
  least as high order as the interior scheme, otherwise the global order drops
  to 2nd near level boundaries. Verify with an MMS on a two-level grid. If
  high-order AMR proves too costly, document the local order reduction and use
  AMR only in regions where 2nd-order error is acceptable.

### 5.3 Time integration

| Physics | Integrator | Notes |
|---|---|---|
| Advection (explicit) | SSP-RK3 | default, positivity-preserving |
| Compressible | SSP-RK3 / low-storage RK4 | CFL from `U + c` (sound speed) |
| Incompressible (projection) | RK3 + projection | project each stage (variable density) |
| Incompressible (pressure-free) | SSP-RK3 + pressure relaxation | EDAC; CFL from `U + c`, not `U` |
| Viscous (stiff) | implicit | backward Euler / CN via multigrid |
| AMR | subcycling | fine levels take `r` steps per coarse step |

Time-step constraints:

```
dt_adv   = CFL · h / (|u| + c)        (compressible, EDAC)
dt_visc  = 0.5 · h² / ν               (explicit diffusion)
dt_proj  = CFL · h / |u|              (projection: no acoustic CFL)
dt = min(dt_adv, dt_visc)             (then reduce for AMR/subcycling)
```

The price of removing the Poisson solve is the acoustic CFL: EDAC with
`c = U/M_c` takes about `1/M_c` more steps than projection. This is the central
trade-off to measure (§5.4, §8.3), not to assume.

Default `CFL = 0.4–0.8` for high-order schemes; tune per scheme and verify with
a temporal convergence test.

### 5.4 Pressure coupling: three strategies, one interface

The incompressible limit is **elliptic**. "No Poisson solve" is not free: you
pay with an acoustic CFL, an inner iteration, or a divergence error. We
therefore implement three strategies behind a `PressureCoupler` policy and let
**time-to-solution at fixed accuracy** decide which one a case uses.

| Strategy | Global solve | Timestep | Divergence | Scaling | Role |
|---|---|---|---|---|---|
| Projection + GMG | `O(N)` V-cycles | advective CFL | zero to solver tol | good (GMG); FFT all-to-all | **production default** |
| **EDAC** | **none** | acoustic CFL | `O(M_c)` (measured) | excellent (local) | experimental, extreme scale |
| Low-Mach preconditioned compressible | inner iterations | advective | good if converged | inner solve | research option |

#### 5.4.1 EDAC — the pressure-free path

Entropically Damped Artificial Compressibility (Clausen 2013) evolves a
pseudo-pressure instead of solving a Poisson equation:

```
∂u/∂t + ∇·(u⊗u) = −∇p/ρ + ∇·(ν(∇u+∇uᵀ)) + f
∂p/∂t + c² ∇·u  = ν_p ∇²p,        ν_p = c² τ
```

Properties:

- **Fully local and explicit** — only nearest-neighbour stencils; no global
  communication, no V-cycle, no FFT transpose. This is exactly the scaling
  property you want.
- `c = U/M_c` is a numerical sound speed; `M_c` controls how incompressible the
  flow behaves. With the standard damping `ν_p = c²τ` and `τ ∝ h/c`, the
  discrete divergence scales as **`O(M_c)`**, not `O(M_c²)`; this was verified
  in `experiments/edac_tgv`. At `M_c = 0.05` the TGV divergence is `~1e-5`.
- `τ` is a damping time; `ν_p = c²τ` stabilizes the pressure field and damps
  acoustic modes. The damping coefficient is the **critical** EDAC parameter:
  in the TGV prototype, `tau_factor = 0.01` (`τ = 0.01 h/c`) was optimal,
  `0.1` was ~20× less accurate, and `1.0` was unstable. Auto-calibrate it.
- **Cost:** `dt ≈ CFL · h/(|u|+c) ≈ M_c · h/U`, i.e. roughly `1/M_c` more
  steps than projection. Measured on the 128² TGV: at `M_c = 0.05`, matching
  projection accuracy required **21× more steps and ~11× more wall time** on
  one CPU with an FFT projection. EDAC wins only when the projection's Poisson
  fraction exceeds the Appendix C threshold (~95% at `M_c=0.05`) — an
  extreme-scale regime. **Measure it; do not assume it.**
- **Prototype verdict:** projection + GMG is the production default. EDAC
  remains an experimental path for extreme-scale runs and moderate accuracy.
- **Initial-pressure caveat:** starting from `p = 0` leaves a long-lived
  acoustic transient (relative error ~2e-2 in the TGV test). A production EDAC
  path needs a consistent initial pressure (one initial projection or a
  spin-up).
- **Scope caveat:** EDAC here is formulated for **constant density**. Coupling
  it to variable density and CSF surface tension (`ρ(c)`, `f_σ`) is a research
  extension; the VOF path defaults to projection + GMG until EDAC-VOF is
  validated against Basilisk.
- **Variants:** Chorin artificial compressibility with dual time (steady
  flows), EDAC with local time stepping, and EDAC + low-Mach correction.

Verification for EDAC:

1. Divergence norm scales linearly with `M_c` at fixed `tau_factor`; document
   the constant.
2. TGV and channel match projection within the spatial error floor at
   `M_c ≈ 0.05`.
3. Cost crossover vs GPU count; require a measured Poisson fraction above the
   Appendix C threshold.
4. Initial-pressure consistency: start from a projection or spin up.

#### 5.4.2 Projection + GMG — the production path

Solve `∇·(1/ρ ∇p) = (1/Δt) ∇·u*` with appropriate BCs. This is the production
default and the accuracy reference for the incompressible solver.

1. **Geometric multigrid (GMG)** — default for the projection path.
   - Cell-centered 7-point stencil, face-centered coefficients.
   - Red-black Gauss–Seidel or Chebyshev smoothing; V-cycle (W-cycle option).
   - Coarse-grid direct solve (small) or extra smoothing.
   - Coarse–fine BCs consistent with AMR; composite solve for multi-level.
   - Fused smoothing kernels; keep the whole V-cycle on device.
2. **FFT** — fast path for uniform periodic/periodic-in-2D grids.
   - cuFFT/hipFFT/oneMKL FFT with pencil decomposition and MPI transposes.
   - Great for TGV/channel; poor scaling beyond ~O(10³) GPUs.

Guidance:

- Expose `pressure.model = "edac" | "projection"` and
  `pressure.poisson_backend = "gmg" | "fft"`.
- Report `pressure_fraction`, `steps`, and divergence in every scaling run.
- For variable density, GMG uses face coefficients `1/ρ_f`; FFT is only valid
  for constant density (or as a preconditioner).

#### 5.4.3 Low-Mach preconditioned compressible — why it is not a free lunch

Preconditioning removes the acoustic stiffness of the compressible system for
**steady** flows. For **unsteady** DNS it requires dual time stepping: each
physical step performs inner pseudo-time iterations until convergence, which is
again an elliptic (Poisson-like) solve. It is a valid research direction, but
it does **not** eliminate the global solve — it renames it. We keep it as an
optional research path, not the default.

#### 5.4.4 Decision rule

| Regime | Default |
|---|---|
| `M > 0.3` (shocks, acoustics) | compressible NS + HLLC/low-Mach Riemann |
| `0.01 < M < 0.3` | compressible or projection; benchmark both |
| `M < 0.01` (strict incompressible DNS) | **projection + GMG** (production) |
| Divergence-sensitive / exact projection needed | projection + GMG |
| Extreme-scale trial where measured Poisson fraction > ~95% | EDAC (experimental) |

The point of the architecture is that **this is a runtime/config choice**, not
a rewrite: all three share the grid, reconstruction, time integrator, I/O, and
diagnostics.

### 5.5 Boundary conditions

Generic, per-patch, per-side, per-field descriptors:

```cpp
enum class BCType { Periodic, Dirichlet, Neumann, Robin, Wall, Inflow, Outflow,
                    Symmetry, WaveTransmissive, IBM };
struct BCParams { BCType type; double value; double t0, ramp; ... };
```

- Ghost-fill kernels switch on a **uniform** per-face value (no divergence
  within a face), which is cheap on GPU.
- Periodic faces are handled by the decomposition (no special kernel).
- Time-dependent inflow via a small device functor table.
- Wall: no-slip for velocity, Neumann for pressure/scalars.
- Outflow: zero-gradient + optional wave-transmissive for compressible.
- Symmetry: mirror normal component, copy tangential.

### 5.6 Verification targets

| Test | What it proves | Target |
|---|---|---|
| MMS advection–diffusion | spatial order | observed order ≥ 3.8 (4th), ≥ 5.5 (6th) |
| MMS Euler/NS | coupled order | ≥ 3.5 |
| Low-Mach vortex / acoustic pulse | all-Mach Riemann | correct wave speeds, low dissipation |
| EDAC divergence | pressure-free accuracy | `‖∇·u‖ ∝ M_c` (measured ~1e-5 at `M_c=0.05`) |
| EDAC vs projection | cost crossover | same TGV accuracy, compare time-to-solution |
| Freestream preservation | geometric consistency | machine precision |
| Mass/momentum/energy conservation | flux form | ≤ 1e-12 relative per step |
| Static bubble | balanced force | spurious currents ≤ 1e-10 |
| TGV Re=1600 | DNS accuracy | match spectral reference within scheme error |
| Channel Re_τ=180/395 | wall-bounded DNS | match Moser et al. profiles |
| Sod shock tube | shock capturing | correct wave speeds, no oscillations |
| Dam break | VOF validation | match Martin & Moyce / Basilisk |
| AMR two-level MMS | interpolation/reflux order | ≥ 2nd, ideally ≥ 3rd |

---

## 6. Kokkos execution model and performance rules

### 6.1 Backends and build matrix

All three GPU backends are **buildable from day 1**, as decided. The honest
qualification: CUDA is the first performance-tuned target; HIP and SYCL are
kept fully compiling and unit-tested in CI, but kernel tuning on them waits
for hardware access.

| Backend | Day-1 status | CI | Tuning |
|---|---|---|---|
| Serial / OpenMP | full | every PR | continuous |
| CUDA | full | nightly (small GPU) | continuous |
| HIP | buildable + unit tests | weekly / cloud MI250 | after CUDA baseline |
| SYCL | buildable + unit tests | weekly / cloud PVC | after CUDA baseline |

```
CPU:   Serial, OpenMP
GPU:   CUDA (NVIDIA), HIP (AMD), SYCL (Intel)
```

CMake presets:

```bash
cmake --preset cpu-serial
cmake --preset cpu-openmp
cmake --preset cuda-a100      # -DKokkos_ARCH_AMPERE80=ON
cmake --preset cuda-h100      # -DKokkos_ARCH_HOPPER90=ON
cmake --preset cuda-local     # -DKokkos_ARCH_PASCAL61=ON (Quadro P600)
cmake --preset hip-mi250      # -DKokkos_ARCH_AMD_GFX90A=ON
cmake --preset sycl-pvc       # -DKokkos_ARCH_INTEL_PVC=ON
```

Pin **Kokkos 5.2.x** (or the current stable at kickoff) and record the exact
version in `CMakeLists.txt` and in every benchmark output. Kokkos 5 requires
C++20; that is our baseline anyway. Keep a **backend-agnostic numerics layer**;
any `#ifdef KOKKOS_ENABLE_*` outside the platform/halo/IO layer is a code-review
blocker.

### 6.2 Kernel patterns

- **Flat 1D `parallel_for`** over `ncell` with manual index arithmetic is the
  default for stencils. `MDRangePolicy` is fine for non-hot loops.
- `Kokkos::LayoutRight` with `i` fastest → contiguous loads/stores.
- **One thread per cell**, vectorized by the hardware across `i`.
- **TeamPolicy + scratch** only where a cell needs a temporary array or a
  block-local reduction (PLIC clipping, Riemann diagnostics).
- **Fuse aggressively:** one kernel per stage, not per variable. Reconstruct
  all variables in one kernel; compute all face fluxes in one kernel.
- **Avoid atomics in the state update.** Use atomics only for diagnostics, and
  make those deterministic if they feed regression tests.
- **No allocations inside the time loop.** Allocate workspaces at setup;
  reuse them.
- **Fence only at true boundaries** (before MPI, I/O, and host reads). Use a
  `Kokkos::fence()` audit in debug mode.
- **Index types:** `int` for local cell indices, `int64_t` for global IDs;
  never `size_t` in device inner loops (slower on GPU).
- **Avoid `DualView` in the hot path.** Use explicit `create_mirror_view_and_copy`
  only for I/O and configuration.

### 6.3 Memory traffic budget

For an explicit stage, the lower bound is:

```
bytes/step ≈ nvar · ncell · 8 · (reads + writes)
```

A 7-point 4th-order stencil touches ~2–3× the field per direction. This is a
**bandwidth-bound** code; the roofline target is 50–70% of STREAM bandwidth for
stencil kernels, lower for WENO/PLIC.

Optimizations, in order of expected payoff:

1. Fuse reconstruction + flux + divergence per direction (halves traffic).
2. Use `B=64` boxes to amortize ghosts.
3. Register-block over `i` (process 2–4 cells per thread) to reuse neighbors.
4. Cache-block the `k` direction; keep a sliding window of planes.
5. Avoid storing all three face-flux arrays if a fused sweep is possible.

### 6.4 Portability discipline

- `#ifdef KOKKOS_ENABLE_CUDA` is allowed **only** in:
  `halo/` (GPU-aware MPI query), `io/` (staging), `profiling/`, and
  `core/platform.hpp`. Never in numerics.
- Use `Kokkos::abort` for device-side fatal errors; host-side exceptions.
- Test every PR on at least Serial + OpenMP; GPU CI nightly (self-hosted or
  cloud runner).
- Keep a `KOKKOS_ENABLE_DEBUG` build with bounds checking and `Kokkos::fence`
  auditing.

### 6.5 Multi-GPU / multi-node

- **One MPI rank per GPU** (standard, simplest, best performing).
- `mpirun -np N` with `--bind-to` / `--map-by` tuned per cluster.
- GPU-aware MPI preferred; staged fallback is a supported mode.
- Overlap halo exchange with interior computation (see §4.6).
- For very large runs, consider `NVSHMEM`/`CUDA IPC` for neighbor exchange as
  a later optimization — **not** in v1.

---

## 7. Generic software design

### 7.1 Compile-time model interface

Static polymorphism only (no virtual calls in device code):

```cpp
struct Edac {
  static constexpr int nvar = 4;
  enum : int { U = 0, V = 1, W = 2, P = 3 };

  struct Primitive { double u, v, w, p; };

  KOKKOS_INLINE_FUNCTION
  static Primitive primitive(const double* q) { ... }

  KOKKOS_INLINE_FUNCTION
  static void flux(const Primitive& L, const Primitive& R, int dir,
                   double* F) { ... }   // central or Riemann
};

template <class Model, int Dim, class MemorySpace>
class Solver { ... };
```

- `Dim` is a compile-time template parameter (1, 2, 3) so 2D validation runs
  use the same code and index math.
- `MemorySpace` is the Kokkos memory space; `ExecSpace` is separate to allow
  host mirrors.
- Adding a model requires: variable list, primitive/conservative transforms,
  flux function, and (optionally) source terms. Nothing else.
- A runtime `model = "compressible" | "incompressible" | "vof"` string selects
  the equation set in `main`; `pressure.model = "edac" | "projection"` selects
  the coupler for the incompressible set. All device code remains compile-time
  specialized.
- Pressure coupling is a **separate policy** (`PressureCoupler`), so `edac` and
  `projection` reuse the same momentum, transport, and boundary-condition
  operators. This is what makes the "no-Poisson" experiment a config switch
  rather than a fork.

### 7.2 Field registry

```cpp
template <int NVAR, class MemorySpace>
class State {
  // named accessors resolved at compile time
  PatchData data;                 // [NVAR][nz+2G][ny+2G][nx+2G]
  std::array<std::string, NVAR> names;
  std::array<Centering, NVAR> centering;   // Cell, FaceX/Y/Z
  std::array<BCParams, NVAR*6> bc;
};
```

- Field centering is explicit, so the same `State` can hold cell-centered
  velocities (v1) and face-centered velocities (optional MAC variant) without
  changing operators.
- Metadata (name, centering, units, BC) is host-side only; kernels see raw
  views.

### 7.3 Configuration

TOML via `toml++` (header-only, MIT) — better than JSON for scientific input
(diff-friendly, comments, typed arrays). CLI overrides with `--set key=value`.

```toml
[grid]
mode = "uniform"          # "uniform" | "amr"
dim  = 3
n    = [512, 512, 512]
box  = [64, 64, 64]
periodic = [true, true, true]

[numerics]
order          = 4
reconstruction = "central4"   # central4 | central6 | weno5 | teno6
riemann        = "hllc_lowmach"  # none | hllc | hllc_lowmach | ausm_up | llf
time           = "ssprk3"
cfl            = 0.5
energy_conserving = true

[physics]
model = "compressible"   # compressible | incompressible | vof
gamma = 1.4
R     = 287.0
nu    = 1.0e-5

[pressure]
model      = "edac"      # edac | projection  (ignored for compressible)
mach_cutoff = 0.05       # EDAC pseudo-Mach M_c
damping_tau = 0.01       # EDAC tau
poisson_backend = "gmg"  # gmg | fft
tol        = 1.0e-10
max_iter   = 50

[amr]
enabled       = false
refine_ratio  = 2
max_level     = 2
tag           = "gradient"   # gradient | wavelet | vorticity | interface
tag_threshold = 0.05

[io]
format     = "hdf5"          # hdf5 | adios2 | vtk
plot_every = 100
checkpoint_every = 1000

[run]
t_end   = 20.0
max_step = 100000
```

Validation: unknown keys are errors; ranges are checked; the full resolved
config is written to the output directory.

### 7.4 I/O

- **Plotfiles:** HDF5 + XDMF for ParaView. Parallel HDF5 (MPI-IO) or ADIOS2
  (BP5) for scalable output. Isolate I/O behind `io::Writer`.
- **Checkpoint/restart:** full state, grid hierarchy, time, step, RNG state.
  Restart must reproduce the same trajectory on the same backend.
- **Staging:** GPU → host copy for I/O is acceptable (I/O is not the hot path),
  but keep it out of timed regions. Report I/O time separately.
- **Benchmark mode:** `--no-io`, `--no-diagnostics` for clean timing.

### 7.5 Testing pyramid

| Level | Tool | Scope |
|---|---|---|
| Unit | Catch2 v3 | index math, boxes, SFC, reconstruction coefficients, Riemann, EOS |
| Verification | custom + Catch2 | MMS, conservation, symmetry, convergence tables |
| Regression | Catch2 + golden files | small cases, norms with tolerance, CPU vs GPU |
| Performance | scripts + JSON | stencil microbench, TGV step time, scaling curves |

Definition of done for any feature: **unit test + verification case + CPU/GPU
run + docs + measured cost**.

### 7.6 Repository layout

```
saturnCFD/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/                    # toolchains, FindKokkos, options
├── external/                 # submodules or FetchContent: Kokkos, Catch2, toml++
├── apps/                     # tgv, channel, cavity, sod, rti, dam_break, bubble
├── src/
│   ├── core/                 # types, config, logging, timer, error, units
│   ├── mesh/                 # box, boxarray, patch, level, distribution, sfc
│   ├── halo/                 # exchange, interpolation, reflux, physical bc
│   ├── fields/               # state, field metadata, scratch
│   ├── numerics/             # reconstruction, riemann, operators, limiters
│   ├── physics/              # eos, transport, surface tension, ibm
│   ├── solvers/              # advection, all-mach (edac), projection, compressible, vof
│   ├── amr/                  # tagging, clustering, regrid, load balance
│   ├── io/                   # hdf5, adios2, vtk, checkpoint
│   └── drivers/              # time loop, subcycling, diagnostics
├── tests/
│   ├── unit/
│   ├── verification/         # mms_*, conservation_*
│   └── regression/
├── benchmarks/               # case files, scaling scripts, comparison data
│   ├── cases/
│   ├── scripts/              # run_sweep.py, plot_scaling.py, roofline.py
│   └── reference/            # published/spectral data
├── docs/
│   ├── design/               # grid.md, numerics.md, kokkos.md, amr.md
│   ├── validation/           # per-case reports
│   └── adr/                  # architecture decision records
└── PLAN.md
```

### 7.7 Coding standards

- C++20, `clang-format` (Google-ish, 100 cols), `clang-tidy` (bugprone,
  performance, modernize).
- Warnings-as-errors in CI; sanitizers (ASan/UBSan) on CPU builds.
- Every `Kokkos::parallel_*` has a unique, searchable label
  (`"advect_x"`, `"poisson_smooth_rb"`) for profiling.
- Naming: `snake_case` for functions/variables, `CamelCase` for types,
  `k_` for compile-time constants.
- No `using namespace` in headers; no raw `new/delete`.

---

## 8. Verification, validation, and benchmarking

### 8.1 Verification (code correctness)

Run on every PR:

1. **MMS convergence** for each model and each order; assert observed order.
2. **Conservation**: total mass/momentum/energy drift over 1000 steps.
3. **Symmetry**: rotated/reflected problems produce identical norms.
4. **Freestream preservation** on warped/AMR grids (uniform flow stays uniform).
5. **Static bubble**: spurious currents below threshold.
6. **Restart**: bitwise-identical trajectory on the same backend.

### 8.2 Validation (physics correctness)

| Case | Reference | Metrics |
|---|---|---|
| TGV Re=1600, 256³/512³ | Wang et al. spectral data | enstrophy/dissipation vs t |
| Channel Re_τ=180, 395 | Moser et al. | mean/log-law, Reynolds stresses |
| Lid-driven cavity Re=1000, 3200 | Ghia et al. | centerline profiles |
| Rayleigh–Taylor 3D | Ramaprabhu / Basilisk | front position, mixing width |
| Dam break 3D | Martin & Moyce; Basilisk | front position, height |
| Rising bubble | Hysing et al. | rise velocity, shape |
| Sod / double Mach | exact / published | wave speeds, oscillations |

### 8.3 Performance benchmarking

**Report per run:**

- Hardware, compiler, Kokkos version, backend, precision.
- Grid, scheme, order, `dt`, steps, wall time.
- **Cell-updates/s**, ns/cell/step.
- Achieved memory bandwidth vs STREAM; achieved FLOP/s vs peak.
- Pressure-coupling fraction (Poisson or EDAC update), halo fraction, I/O fraction.
- Number of steps and discrete divergence norm (essential for EDAC vs projection).
- Strong and weak scaling efficiency.

**Scaling protocol:**

- Strong: 512³ TGV on 1, 2, 4, 8, …, N GPUs; fixed `t_end`.
- Weak: 256³ per GPU; double the GPUs and the global grid.
- CPU: 1, 2, 4, 8, …, 1024 cores; report NUMA effects.
- **Pressure-strategy crossover:** run the same TGV with EDAC and with
  projection + GMG at each GPU count; plot `T_EDAC / T_proj` and divergence.
- Repeat each run 3×, report median and spread.
- Pin clocks / use `nvidia-smi -lgc` where permitted; otherwise report variance.

**Fair comparison with other codes:**

Compare at **fixed accuracy**, not fixed grid:

1. Choose TGV Re=1600; run each code at resolutions giving the same
   enstrophy error vs the spectral reference.
2. Report **time-to-solution** and **energy-to-solution**, not just CUPS.
3. Compare only codes with the same physics and comparable schemes:

| Code | Type | Fair for |
|---|---|---|
| CaNS | FFT, incompressible, GPU | uniform single-phase DNS |
| Basilisk | octree VOF, CPU | multiphase/AMR |
| OpenFOAM | unstructured FVM, CPU | general FVM baseline |
| AFiD | FD, GPU | wall-bounded DNS |
| incflo (AMReX) | AMR, GPU | AMR/IBM |
| NekRS | spectral element, GPU | high-order baseline |

Document differences in scheme order and grid; do not claim "faster" without
matching accuracy.

**Benchmark harness:**

- `benchmarks/scripts/run_sweep.py` → JSON results.
- `benchmarks/scripts/plot_scaling.py` → scaling and roofline plots.
- CI nightly job on a small GPU; manual large runs on the cluster.
- Store results in `benchmarks/results/<date>/` with the resolved config.

---

## 9. Roadmap

Assuming **one full-time developer**; adjust for your schedule. Each phase has
an exit criterion; do not start the next phase until it is met.

### Phase 0 — Foundation (1–2 weeks)

- [x] Create a **new git repo** in `saturnCFD/`.
- [x] Install CMake 3.31.6, Kokkos 5.2.2 (Serial + OpenMP), CUDA 12.2, and a
      Kokkos-CUDA build (sm_61).
- [x] CMake skeleton, presets, strict warnings; CI workflow (Kokkos build +
      full CTest) and clang-format configuration added.
- [x] `core/`: version and execution-space dispatch; config/logging pending.
- [x] Stencil + STREAM microbenchmarks on CPU and the Quadro P600; roofline
      measured (see `benchmarks/results/`).
- [x] Design captured in `PLAN.md`; splitting into `docs/design/` pending.

**Exit (met for CPU and local P600):** build, tests, and stencil roofline
measured; HIP/SYCL verification waits for hardware.

### Phase 1 — Mesh + scalar solver (2–3 weeks)

- [x] `Box`, `PatchField`, uniform single-rank decomposition
      (`BoxArray`/`Level` pending).
- [x] Field metadata and periodic ghost fill (physical BCs pending).
- [x] VTK snapshot output and binary checkpoint/restart (1–2 ranks, CPU/CUDA);
      HDF5/ADIOS2 backends are later.
- [x] 2nd-order scalar advection–diffusion + SSP-RK3.
- [x] MMS 2nd-order convergence; mass conservation on Serial, OpenMP, and
      CUDA (`docs/verification/`).
- [x] MPI halo exchange: face exchange and global reductions verified on 1–8
      CPU ranks and 2 CUDA ranks.
- [x] Halo bandwidth measured (`docs/verification/scalar_mms.md`); overlap and
      weak scaling still pending.
- [x] Overlap halo exchange with interior RHS (`begin`/`end`, boundary shell);
      implemented and correct but opt-in because it is not yet faster at the
      tested CPU/GPU sizes (`docs/verification/halo_overlap.md`).

**Exit:** 2nd-order MMS passes on 1–8 ranks at order 1.99–2.00; halo unit test
reports zero errors; exchange cost and overlap measured. Making overlap a win
requires GPU streams / GPU-aware MPI and larger exchange fractions.

### Phase 2 — High-order compressible (3–4 weeks)

- [x] WENO5-Z, TENO6, central 4/6 reconstruction
      (`src/numerics/reconstruction.hpp`; orders 5.06 and 5.99 measured,
      `docs/verification/reconstruction.md`).
- [x] Riemann solvers: HLLC, HLLC-low-Mach, AUSM⁺-up, LLF fallback
      (`src/numerics/riemann.hpp`; unit-tested, full Sod/MMS validation in a
      time-dependent solver is the next step).
- [ ] Compressible NS + ideal gas; Sod, double Mach, low-Mach vortex,
      acoustic pulse. 1D inviscid Euler Sod gate passing
      (`apps/euler1d.cpp`, `docs/verification/sod.md`); 3D compressible NS
      path (inviscid + 4th-order viscous fluxes) implemented with
      entropy/shear MMS at 4th order (`docs/verification/compressible_mms.md`).
      Production-style driver with TGV, acoustic pulse, low-Mach vortex, and
      the standard Woodward–Colella double-Mach setup with exact
      time-dependent inflow boundaries, VTK/checkpoint, and full 3D halo
      exchange (`docs/verification/compressible_driver.md`). Reference-data
      comparisons for double Mach and TGV, and GPU scaling, pending.
- [x] MMS 4th/6th order; energy conservation check (entropy-wave MMS at
      `central4`/WENO5-Z/`central6`; periodic inviscid mass and total-energy
      drift ≤ 1e-15 enforced by `--energy-tol`).
- [x] GPU performance baseline on the local P600: stencil 27.93 GB/s,
      STREAM 54.2 GB/s (`docs/performance/gpu_p600.md`); full serial and MPI
      CUDA test suites pass. Register/occupancy tuning and multi-GPU scaling
      remain.

**Current:** reconstruction, all four Riemann solvers, and the 1D Sod gate are
passing on CPU Serial/OpenMP. The 3D Navier–Stokes path has smooth MMS at
4th–6th order, exact conservation on periodic inviscid runs, the standard
double-Mach setup with exact time-dependent inflow boundaries, a low-Mach TGV
comparison at 64³ (peak 4.4% below the reference), and bitwise-identical
double-Mach results on 1/2/4 ranks. Next: TGV resolution convergence at
`M=0.1`, a GPU performance baseline, and CUDA validation.

**Exit:** 4th-order MMS ≥ 3.8; Sod correct; low-Mach vortex stable; strong
scaling to 8 GPUs.

### Phase 3 — Incompressible: projection + GMG, EDAC experiment (4–5 weeks)

- [ ] Projection + GMG production solver (RK3, Rhie–Chow).
- [x] Staggered MAC projection prototype with SSP-RK3, projection after every
      stage, and the periodic GMG solver; exact 2D Taylor–Green vortex and 3D
      ABC (Beltrami) flow at 2nd order
      (`docs/verification/incompressible.md`). The MAC grid avoids the
      checkerboard mode, so Rhie–Chow is not needed on this path.
- [x] Cell-centered geometric-multigrid Poisson solver with periodic,
      homogeneous-Neumann, and homogeneous-Dirichlet boundaries; 7-point
      Laplacian, red-black Gauss-Seidel, full-weighting restriction, trilinear
      prolongation, 2nd-order convergence on periodic and Neumann boxes
      (`docs/verification/poisson.md`).
- [x] FFT Poisson backend for periodic uniform grids (FFTW3, host-side);
      2nd-order MMS, zero mode projected, ~40x faster than MG at n=64
      (`docs/verification/poisson.md`).
- [x] Wall-bounded MAC channel (periodic x/z, no-slip y walls): Couette is an
      exact discrete steady state and plane Poiseuille converges at second
      order (`docs/verification/channel.md`). A fast FFT-x/z + tridiagonal-y
      Poisson solver is available via `--poisson fft`.
- [x] Lid-driven cavity with x/y no-slip walls and a moving lid; Re=100
      centreline profiles match Ghia et al. (RMS 7.6e-3 u, 9.9e-3 v),
      `max|div| ~ 3e-11` (`docs/verification/cavity.md`). Re=400 tables are
      included; a coarse 24² run gives RMS 0.074/0.072.
- [x] EDAC pressure evolution, damping `τ`, configurable pseudo-Mach `M_c`;
      C++ TGV prototype matches projection velocity error at `M_c=0.05` with
      ~12.6x more steps and `O(M_c)` divergence (`docs/verification/edac.md`).
- [ ] TGV Re=1600, turbulent channel Re_τ=180/395, cavity at higher Re on both
      paths. The 3D TGV is implemented; a 4th-order filter (`alpha=0.1`) plus
      the FFT pressure solver stabilizes n=64 (peak `-dK/dt/V = 1.142e-2` at
      `t=8.96`, 8.6% below the reference) but it remains a filtered exercise,
      not a DNS benchmark (`docs/verification/incompressible.md`).
- [x] Divergence-vs-`M_c` study; cost-crossover plot (EDAC vs projection).
      GPU P600 at N=64: EDAC (`M_c=0.05`) 1.53 s vs projection+FFT 0.23 s
      (6.7x slower); no crossover at this scale.
- [ ] Multi-GPU scaling; report steps, divergence, Poisson fraction.

**Current:** periodic GMG and FFT Poisson solvers, the staggered-MAC projection
solver, wall-bounded channel flow, the Ghia Re=100/400 cavity, and the EDAC
prototype are verified. The 3D TGV runs stably at n=64 with a 4th-order filter
and peaks within 9% of the reference. Next: turbulent channel, higher-Re
cavity, a dealiased 3D TGV, a GPU cost-crossover, and a distributed multigrid
solve.

**Exit:** projection + GMG validated against references and ≥ 70%
strong-scaling efficiency at 8 GPUs for 512³; EDAC measured against the
prototype, with a documented crossover only if the Poisson fraction is high
enough (Appendix C).

### Phase 4 — Multiphase VOF (4–6 weeks)

- [x] PLIC reconstruction + geometric flux (3D, split): 2D and 3D prototypes
      with polyhedron clipping, Strang splitting, step/sphere translation and
      rotation tests, round-off mass conservation
      (`docs/verification/vof.md`).
- [x] 2D PLIC prototype: Youngs normals, bisection alpha, Strang-split
      geometric fluxes; step/circle translation and solid-body rotation at
      L1 2.5e-3–7.7e-3 with round-off mass drift and bounded fractions
      (`docs/verification/vof.md`).
- [x] Bounded redistribution (clamp + capacity-weighted mass restore); 2D/3D
      tests conserve mass to ~1e-16 and restore `[0,1]`
      (`docs/verification/vof.md`).
- [x] Height-function curvature (2D and 3D): column/row heights and 5-point
      derivatives; sine interfaces give max errors 2.54e-07 (2D, n=128) and
      2.08e-03 (3D, n=48) (`docs/verification/vof.md`). Balanced-force CSF
      remains.
- [x] Variable-density projection `∇·(1/ρ ∇p)` with ρ(c) interpolation:
      static bubble at 1000:1 density ratio gives `|u|max=3.9e-13`, exact
      `sigma/r` pressure jump (`docs/verification/vof.md`). The full explicit
      step (advection, variable viscosity, gravity, CSF, projection, VOF) is
      exercised by static-bubble and uniform-advection tests.
- [ ] Dam break, RTI, rising bubble; comparison with Basilisk.
- [ ] Spurious-current test; interface sharpness metrics.
- [ ] Default pressure coupler: projection + GMG (EDAC-VOF is research).

**Current:** 2D and 3D PLIC prototypes are verified (translation, rotation,
mass conservation, boundedness), and 2D/3D height-function curvature is
verified on sine interfaces, bounded redistribution is in place, and balanced
CSF and a variable-density projection are validated on a static bubble, and a
level-set curvature estimator covers closed interfaces, and the full
variable-density momentum step is implemented. Next: wall boundaries and
dam-break/RTI/bubble validation.

**Exit:** dam break and bubble match references within a few %; mass drift
< 1e-8; static bubble currents < 1e-8.

### Phase 5 — AMR (4–6 weeks)

- [ ] Static two-level AMR: BoxArray, interpolation, refluxing.
- [ ] Tagging (gradient/wavelet/interface), Berger–Rigoutsos clustering.
- [ ] Regridding, SFC load balance, redistribution.
- [ ] Subcycling + composite GMG.
- [ ] AMR MMS (order at coarse–fine boundaries); AMR scaling.
- [ ] Adaptive TGV and dam break.

**Exit:** two-level MMS ≥ 2nd order; conservation across levels; AMR weak
scaling ≥ 70% at 32 GPUs.

### Phase 6 — Hardening + benchmarks (2–3 weeks)

- [ ] Profile and optimize the top 3 kernels; roofline report.
- [ ] Full benchmark matrix vs CaNS/Basilisk/OpenFOAM.
- [ ] Documentation, tutorials, input-deck reference.
- [ ] Release v0.1 (tag, license, DOI optional).

**Exit:** reproducible scaling plots; validation report; tagged release.

**Total:** ~6–8 months for a solid v1 with three backends. **MVP (Phase 0–3)
≈ 4 months.**

---

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| EDAC acoustic CFL costs `1/M_c` more steps | measured ~21× more steps at `M_c=0.05` | keep projection + GMG production; use EDAC only if measured Poisson fraction is extreme |
| EDAC divergence is `O(M_c)`, not `O(M_c²)` | accuracy/step trade-off worse than hoped | document; auto-calibrate `τ`; do not promise quadratic divergence |
| EDAC initial-pressure transient | `p₀=0` gave ~2e-2 error in TGV | one initial projection or spin-up; validate startup |
| Projection Poisson dominates/scales poorly | blocks multi-GPU scaling | GMG first, benchmark at 8–64 GPUs early; FFT fallback |
| GPU AMR halo/regrid overhead | kills AMR scaling | many boxes/rank, fused packing, overlap, device metadata |
| PLIC/WENO branch divergence | low GPU utilization | branchless select, separate smooth/shock kernels, profile |
| High-order AMR drops to 2nd order | accuracy claim invalid | high-order interpolation + reflux; MMS; document limits |
| Kokkos performance gaps vs native | 10–30% loss | backend-tuned launch bounds; `#ifdef` only in platform layer |
| Three backends triples CI/tuning | schedule slip | CUDA tuned first; HIP/SYCL build-only until hardware is available |
| Local GPU too small (P600, 2 GB) | can't measure scaling | develop on CPU; rent A100/H100 for scaling |
| CMake 3.16 locally | modern presets fail | install CMake ≥ 3.24 (Kitware apt or pip) |
| Scope creep (chemistry, RANS, GUI) | misses milestones | strict non-goals; MVP first |
| Verification debt | silent wrong results | MMS + regression gating from day 1 |

---

## 11. Immediate next steps (this week)

1. **Fix the repository:** `git init` inside `saturnCFD/`, add `.gitignore`,
   `README.md`, and this plan; keep the course repo at `/home/cfd103` untouched.
2. **Toolchain:** install CMake ≥ 3.24, Kokkos 5.2.x, Catch2, toml++; verify
   `cpu-serial`, `cpu-openmp`, and `cuda-local` (sm_61) builds. Add HIP/SYCL
   compile jobs to CI (no hardware needed for compile checks).
3. **Microbenchmark:** 7-point 3D stencil in Kokkos; measure GB/s and
   ns/cell/step on the i7 and the P600. This calibrates all later claims.
4. **EDAC de-risk:** completed in `experiments/edac_tgv/` (2D TGV, EDAC vs FFT
   projection). Result: EDAC matches accuracy at `M_c≈0.05` but costs ~21×
   more steps; projection + GMG stays the production default. Re-run the
   crossover on GPU hardware when available.
5. **Design docs:** split this plan into `docs/design/grid.md`,
   `numerics.md`, `kokkos.md`, `amr.md`, `pressure.md` with diagrams and
   equations.
6. **Skeleton:** CMake + config + logging + timers + a trivial MMS test.

---

## 12. Decisions and remaining questions

### 12.1 Locked

| Decision | Choice |
|---|---|
| First production solver | Compressible NS (density-based, high-order) |
| Incompressible strategy | Projection + GMG production; EDAC experimental extreme-scale |
| AMR | Uniform first; AMR-ready architecture; block AMR in Phase 5 |
| Backends | CUDA, HIP, SYCL buildable day 1; CUDA tuned first |
| Dependencies | Kokkos, MPI, HDF5, Catch2, toml++ |
| Geometry | Cartesian; IBM later |

### 12.2 Still open

1. **License:** Apache-2.0 (proposed), MIT, GPL-3.0, or private?
2. **Target cluster:** which HPC system/GPU model will host the scaling runs?
   This determines the CUDA/HIP/SYCL architecture flags and MPI launcher
   tuning. If none is fixed yet, we will benchmark on rented A100/H100.
3. **EDAC tuning responsibility:** do you want the EDAC constants (`M_c`, `τ`)
   exposed as user input (flexible, easy to misuse) or auto-calibrated from
   `U`, `h`, and `ν` (safer, less flexible)? Proposed: auto-calibrate with
   manual override.
4. **VOF timing:** keep VOF in Phase 4, or defer it until after the first
   paper/benchmark campaign? The compressible + EDAC paths alone are already a
   publishable code.

---

## Appendix A — Why 4th order, not 2nd or 6th?

- **2nd order** is robust but needs ~2× the cells per direction for the same
  accuracy as 4th order in smooth DNS; on GPUs the extra memory traffic
  usually costs more than the stencil.
- **4th order** is the sweet spot for DNS: significantly lower dissipation and
  dispersion error than 2nd order, still compact (5-point), GPU-friendly, and
  well understood on Cartesian AMR.
- **6th order** helps at very high Re / long integration times but increases
  the stencil to 7 points, ghost width to 3–4, and cost by ~30–50%. Make it a
  compile-time option, not the default.
- **WENO5/TENO6** are for shocks; they are more expensive and more diffusive
  than central 4th order in smooth regions. Use them only where a shock sensor
  or the user requests them.

## Appendix B — Expected performance envelope (targets, not promises)

| Kernel | Bound | Target |
|---|---|---|
| 7-point stencil (FP64) | memory | 60–80% of STREAM |
| WENO5 reconstruction | compute/memory | 25–45% of peak |
| HLLC / low-Mach flux | compute | 20–40% of peak |
| EDAC pressure update | memory | 60–80% of STREAM |
| PLIC VOF sweep | compute/branch | 15–35% of peak |
| GMG smoothing | memory | 50–70% of STREAM |
| Halo exchange (GPU-aware) | latency/bandwidth | < 15% of step at 64 GPUs |

These are goals to validate with the Phase 0 microbenchmark; replace them with
measured numbers before making external claims.

## Appendix C — Eliminating the Poisson solve: the honest math

This appendix exists because "compressible solver for incompressible flow
without a Poisson solve" is the project's central bet. It deserves a quantitative
sanity check before months of engineering.

### C.1 Why the incompressible limit is elliptic

`∇·u = 0` is a constraint on the *whole* velocity field. Taking the divergence
of the momentum equation gives

```
∇·(1/ρ ∇p) = ∇·( −∇·(u⊗u) + ν∇²u + f )
```

which is a global elliptic problem. No purely local update can enforce
`∇·u = 0` exactly; local schemes either relax it (artificial compressibility),
penalize it (`p = −λ∇·u`), or accept an `O(M²)` residual.

### C.2 What a compressible solver actually costs at low Mach

For a compressible solver, the acoustic CFL is

```
dt ≈ CFL · h / (|u| + c) = CFL · h / (U(1 + 1/M))
```

so at `M = 0.05` you take ~20× more steps than a projection solver at the same
`h`. Low-Mach Riemann fixes (HLLC-low-Mach, AUSM⁺-up) remove the `O(1/M)`
dissipation but **not** the stiffness. Preconditioning removes the stiffness
for steady problems; for unsteady DNS it requires dual time stepping, whose
inner iterations are again an elliptic solve. **A compressible solver does not
give you incompressible DNS for free.**

### C.3 EDAC cost model

Let:

- `f` = fraction of a projection step spent in the Poisson solve,
- `α` = cost of one EDAC step relative to the explicit part of a projection
  step (`α ≈ 0.5–0.8`; no V-cycle, but an extra pressure update),
- `M_c` = EDAC pseudo-Mach; steps scale as `1/M_c`.

Then

```
T_proj  ≈ N · t_expl / (1 − f)
T_EDAC  ≈ (N / M_c) · α · t_expl

T_EDAC / T_proj ≈ (α / M_c) · (1 − f)
```

**EDAC wins only when**

```
f > 1 − M_c / α
```

Examples with `α = 0.7`:

| `M_c` | EDAC wins if Poisson fraction `f` > |
|---|---|
| 0.05 | 93% |
| 0.10 | 86% |
| 0.20 | 71% |
| 0.30 | 57% |

The divergence does **not** enter this threshold directly; it sets whether the
EDAC result is accurate enough (measured `O(M_c)`; see C.6).

**Interpretation:** at 8 GPUs, a good GMG solve typically costs 20–50% of the
step, so projection wins. EDAC becomes attractive only when the global solve
dominates — which happens at **very large GPU counts** (coarse-grid latency,
all-to-all, load imbalance) or with a poor Poisson solver. This is exactly the
regime the project cares about, but the crossover must be **measured**, not
assumed.

### C.4 What we measured, and what remains

**Measured (2D TGV, single CPU, FFT projection):**

1. EDAC vs projection accuracy and step count at `N = 64, 128`.
2. Divergence and L2 error vs `M_c` and vs the damping parameter `τ`.
3. Sensitivity to the initial pressure (`p₀` exact vs zero).

**Still to measure (target hardware):**

1. `f` vs GPU count for projection + GMG on TGV 512³ (1 → 256 GPUs).
2. `T_EDAC / T_proj` on GPUs, including halo/communication effects.
3. The GPU count where the curves cross; report it explicitly.

If EDAC never wins on the target hardware, we keep it as a research path and
use projection + GMG for production — the architecture supports both, so this
is a data-driven decision, not a dead end.

### C.5 EDAC calibration (measured default)

```
c        = U_ref / M_c         (numerical pseudo-sound speed, NOT the
                                thermodynamic sound speed)
M_c      = 0.05 ... 0.1        (accuracy-dependent)
τ        = C_tau · h / c       (C_tau ≈ 0.01 measured on TGV)
ν_p      = c² τ
```

Use `U_ref = max(|u|_∞, sqrt(g L), ...)` from the initial condition. Do **not**
use the physical sound speed here: for a liquid it would be enormous and would
collapse `dt` for no accuracy benefit.

`C_tau` is flow- and resolution-dependent. Scan `C_tau ∈ [1e-3, 0.1]` during
set-up; values above ~0.5 over-damp and can destabilize the scheme. Initialize
the pressure from a projection (or a spin-up) rather than zero.

### C.6 Measured prototype results (2D TGV)

From `experiments/edac_tgv/README.md`, `ν = 0.01`, `t_end = 2`, `C_tau = 0.01`:

| Method | N | `M_c` | Steps | Rel. L2 error | `‖∇·u‖₂` | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| projection (FFT) | 64 | — | 40 | 3.2e-5 | 8.9e-16 | 0.12 s |
| EDAC | 64 | 0.05 | 855 | 3.6e-5 | 9.9e-6 | 1.5 s |
| projection (FFT) | 128 | — | 80 | 8.0e-6 | 1.5e-15 | 0.61 s |
| EDAC | 128 | 0.05 | 1710 | 1.3e-5 | 1.4e-5 | 6.5 s |

**Result:** EDAC reaches the projection's accuracy at `M_c ≈ 0.05`, but needs
21× more steps and ~11× more wall time on one CPU with an FFT Poisson solve.
The measured Poisson fraction is only ~24%, far below the 93% threshold
required at `M_c=0.05`. The data therefore confirms projection + GMG as the
production default; EDAC is a scaling bet for a regime this laptop cannot
reach.
