#!/usr/bin/env python3
"""2D Taylor-Green vortex: EDAC vs projection prototype.

Purpose
-------
De-risk the pressure strategy before writing the C++ solver. Both methods use
the *same* 2nd-order central momentum operators on a periodic cell-centered
grid. The projection solver uses an FFT Poisson solve (a best-case reference).
EDAC evolves a pseudo-pressure with local stencils and no global solve.

This is a numerical experiment, not production code. It answers:

  1. How small must M_c be for EDAC to match the exact TGV solution?
  2. How does the discrete divergence scale with M_c?
  3. How many extra steps does EDAC take compared with projection?
  4. What is the wall-time crossover on a single CPU?

Exact solution (periodic domain [0, 2*pi]^2):

    u =  sin(x) cos(y) exp(-2 nu t)
    v = -cos(x) sin(y) exp(-2 nu t)
    p =  (cos(2x) + cos(2y)) / 4 * exp(-4 nu t)

Usage:
    python3 edac_tgv.py --N 128 --nu 0.01 --tend 2 --mc 0.05
    python3 edac_tgv.py --sweep --N 64 128 256 --mc 0.2 0.1 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Exact solution and grid
# ---------------------------------------------------------------------------

def exact_tgv(x, y, t, nu):
    decay = math.exp(-2.0 * nu * t)
    u = np.sin(x) * np.cos(y) * decay
    v = -np.cos(x) * np.sin(y) * decay
    p = 0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * math.exp(-4.0 * nu * t)
    return u, v, p


def make_grid(n, length=2.0 * math.pi):
    h = length / n
    x = (np.arange(n) + 0.5) * h
    return h, np.meshgrid(x, x, indexing="ij")


def ddx(f, h):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * h)


def ddy(f, h):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * h)


def laplacian(f, h):
    return (
        np.roll(f, -1, axis=0)
        + np.roll(f, 1, axis=0)
        + np.roll(f, -1, axis=1)
        + np.roll(f, 1, axis=1)
        - 4.0 * f
    ) / (h * h)


def momentum_rhs(u, v, nu, h):
    """Skew-symmetric advection + viscous diffusion; no pressure."""
    dudx = ddx(u, h)
    dudy = ddy(u, h)
    dvdx = ddx(v, h)
    dvdy = ddy(v, h)

    adv_u = 0.5 * (ddx(u * u, h) + u * dudx) + 0.5 * (ddy(u * v, h) + v * dudy)
    adv_v = 0.5 * (ddx(u * v, h) + u * dvdx) + 0.5 * (ddy(v * v, h) + v * dvdy)

    return -adv_u + nu * laplacian(u, h), -adv_v + nu * laplacian(v, h)


# ---------------------------------------------------------------------------
# EDAC
# ---------------------------------------------------------------------------

@dataclass
class EdacParams:
    nu: float
    mach_c: float
    tau_factor: float = 0.1
    cfl: float = 0.5


def edac_rhs(u, v, p, prm: EdacParams, h):
    ru, rv = momentum_rhs(u, v, prm.nu, h)
    ru -= ddx(p, h)
    rv -= ddy(p, h)

    c = 1.0 / prm.mach_c
    c2 = c * c
    tau = prm.tau_factor * h / c
    nu_p = c2 * tau

    div = ddx(u, h) + ddy(v, h)
    rp = -c2 * div + nu_p * laplacian(p, h)
    return ru, rv, rp


def edac_dt(u, v, prm: EdacParams, h):
    c = 1.0 / prm.mach_c
    tau = prm.tau_factor * h / c
    nu_p = c * c * tau
    umax = max(np.max(np.abs(u)), np.max(np.abs(v)))
    dt_adv = prm.cfl * h / (umax + c)
    dt_visc = 0.5 * h * h / prm.nu
    dt_p = 0.5 * h * h / nu_p
    return min(dt_adv, dt_visc, dt_p)


def rk3(state, rhs, dt):
    k1 = rhs(state)
    y1 = tuple(s + dt * k for s, k in zip(state, k1))
    k2 = rhs(y1)
    y2 = tuple(
        0.75 * s + 0.25 * (y1i + dt * k2i)
        for s, y1i, k2i in zip(state, y1, k2)
    )
    k3 = rhs(y2)
    return tuple(
        (1.0 / 3.0) * s + (2.0 / 3.0) * (y2i + dt * k3i)
        for s, y2i, k3i in zip(state, y2, k3)
    )


def solve_edac(n, prm: EdacParams, tend, p0="exact"):
    h, (x, y) = make_grid(n)
    u, v, p = exact_tgv(x, y, 0.0, prm.nu)
    if p0 == "zero":
        p = np.zeros_like(p)
    p -= p.mean()

    t = 0.0
    steps = 0
    dt_sum = 0.0
    start = time.perf_counter()

    while t < tend - 1.0e-15:
        dt = min(edac_dt(u, v, prm, h), tend - t)
        u, v, p = rk3(
            (u, v, p),
            lambda state: edac_rhs(state[0], state[1], state[2], prm, h),
            dt,
        )
        p -= p.mean()
        t += dt
        steps += 1
        dt_sum += dt

    seconds = time.perf_counter() - start
    return finalize(u, v, x, y, h, tend, prm.nu, steps, dt_sum, seconds,
                   method="edac", mach_c=prm.mach_c)


# ---------------------------------------------------------------------------
# Projection reference (FFT Poisson solve)
# ---------------------------------------------------------------------------

def project_fft(u, v, dt, h):
    n = u.shape[0]
    k = 2.0 * math.pi * np.fft.fftfreq(n, d=h)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    denom = kx * kx + ky * ky

    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    div_hat = 1j * kx * u_hat + 1j * ky * v_hat

    p_hat = np.zeros_like(div_hat)
    mask = denom > 1.0e-14
    p_hat[mask] = -div_hat[mask] / (dt * denom[mask])

    u_hat -= dt * 1j * kx * p_hat
    v_hat -= dt * 1j * ky * p_hat
    return np.real(np.fft.ifft2(u_hat)), np.real(np.fft.ifft2(v_hat))


def solve_projection(n, nu, cfl, tend):
    h, (x, y) = make_grid(n)
    u, v, _ = exact_tgv(x, y, 0.0, nu)

    t = 0.0
    steps = 0
    dt_sum = 0.0
    start = time.perf_counter()

    while t < tend - 1.0e-15:
        umax = max(np.max(np.abs(u)), np.max(np.abs(v)))
        dt = min(cfl * h / max(umax, 1.0e-12), 0.5 * h * h / nu, tend - t)

        # RK3 with projection after each stage.
        k1u, k1v = momentum_rhs(u, v, nu, h)
        u1, v1 = project_fft(u + dt * k1u, v + dt * k1v, dt, h)

        k2u, k2v = momentum_rhs(u1, v1, nu, h)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2u)
        v2 = 0.75 * v + 0.25 * (v1 + dt * k2v)
        u2, v2 = project_fft(u2, v2, dt, h)

        k3u, k3v = momentum_rhs(u2, v2, nu, h)
        u3 = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3u)
        v3 = (1.0 / 3.0) * v + (2.0 / 3.0) * (v2 + dt * k3v)
        u, v = project_fft(u3, v3, dt, h)

        t += dt
        steps += 1
        dt_sum += dt

    seconds = time.perf_counter() - start
    return finalize(u, v, x, y, h, tend, nu, steps, dt_sum, seconds,
                   method="projection", mach_c=0.0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def finalize(u, v, x, y, h, tend, nu, steps, dt_sum, seconds, method, mach_c):
    ue, ve, _ = exact_tgv(x, y, tend, nu)
    err = np.sqrt(np.mean((u - ue) ** 2 + (v - ve) ** 2))
    scale = np.sqrt(np.mean(ue * ue + ve * ve))
    linf = max(np.max(np.abs(u - ue)), np.max(np.abs(v - ve)))

    div = ddx(u, h) + ddy(v, h)
    div_l2 = np.sqrt(np.mean(div * div))
    div_inf = np.max(np.abs(div))

    return {
        "method": method,
        "mach_c": mach_c,
        "n": u.shape[0],
        "nu": nu,
        "tend": tend,
        "steps": steps,
        "dt_mean": dt_sum / steps,
        "seconds": seconds,
        "l2_error": float(err),
        "l2_relative": float(err / scale),
        "linf_error": float(linf),
        "div_l2": float(div_l2),
        "div_inf": float(div_inf),
        "us_per_cell_step": seconds / (u.shape[0] * u.shape[1] * steps) * 1.0e6,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_row(result):
    print(
        f"{result['method']:>10s}  N={result['n']:>4d}  "
        f"Mc={result['mach_c']:>5.3f}  "
        f"steps={result['steps']:>6d}  "
        f"dt={result['dt_mean']:.2e}  "
        f"L2rel={result['l2_relative']:.3e}  "
        f"divL2={result['div_l2']:.3e}  "
        f"time={result['seconds']:.3f}s"
    )


def make_plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edac = [r for r in results if r["method"] == "edac"]
    proj = [r for r in results if r["method"] == "projection"]
    if not edac:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    n_values = sorted({r["n"] for r in edac})
    for n in n_values:
        rows = sorted((r for r in edac if r["n"] == n), key=lambda r: r["mach_c"])
        mc = [r["mach_c"] for r in rows]
        axes[0].loglog(mc, [r["l2_relative"] for r in rows], "o-", label=f"N={n}")
        axes[1].loglog(mc, [r["div_l2"] for r in rows], "o-", label=f"N={n}")

    for n in n_values:
        rows = [r for r in edac if r["n"] == n]
        if rows:
            axes[2].loglog(
                [r["mach_c"] for r in rows],
                [r["steps"] for r in rows],
                "o-",
                label=f"EDAC N={n}",
            )
    for n in sorted({r["n"] for r in proj}):
        row = next(r for r in proj if r["n"] == n)
        axes[2].axhline(row["steps"], linestyle="--", linewidth=1.0,
                        label=f"projection N={n}")

    axes[0].set_xlabel("M_c")
    axes[0].set_ylabel("relative L2 velocity error")
    axes[1].set_xlabel("M_c")
    axes[1].set_ylabel("discrete divergence L2")
    axes[2].set_xlabel("M_c")
    axes[2].set_ylabel("number of steps")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--tend", type=float, default=2.0)
    parser.add_argument("--mc", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1,
                        help="pressure damping time factor: tau = factor*h/c")
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--p0", choices=["exact", "zero"], default="exact")
    parser.add_argument("--method", choices=["edac", "projection", "both"],
                        default="both")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--n-list", type=int, nargs="+", default=None,
                        help="resolutions for a sweep")
    parser.add_argument("--mc-list", type=float, nargs="+", default=None,
                        help="M_c values for a sweep")
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--plot", type=str, default="")
    args = parser.parse_args()

    if args.n_list:
        n_values = args.n_list
    elif args.sweep:
        n_values = [64, 128, 256]
    else:
        n_values = [args.n]

    if args.mc_list:
        mc_values = args.mc_list
    elif args.sweep:
        mc_values = [0.2, 0.1, 0.05]
    else:
        mc_values = [args.mc]

    results = []
    for n in n_values:
        if args.method in ("projection", "both"):
            results.append(solve_projection(n, args.nu, args.cfl, args.tend))
            print_row(results[-1])
        for mc in mc_values:
            prm = EdacParams(nu=args.nu, mach_c=mc, tau_factor=args.tau,
                             cfl=args.cfl)
            results.append(solve_edac(n, prm, args.tend, p0=args.p0))
            print_row(results[-1])

    if args.json:
        path = Path(args.json)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if args.plot:
        make_plot(results, args.plot)


if __name__ == "__main__":
    main()
