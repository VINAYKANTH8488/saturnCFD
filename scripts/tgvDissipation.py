#!/usr/bin/env python3
"""Compute the kinetic-energy dissipation rate from a driver history CSV.

`saturn_compressible_sim --history FILE --diag-every N` writes

    t,step,mass,momentum,kinetic,internal,total_energy,max_mach

This script differentiates the kinetic energy in time with second-order
central differences (one-sided at the ends) and writes t,K,epsilon where
epsilon = -dK/dt. Use --volume to report per-volume quantities (for the
Taylor-Green box the volume is (2*pi)^3).

Example:
    python3 scripts/tgvDissipation.py tgv.csv --volume 248.0502134
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


def read_history(path: str) -> tuple[list[float], list[float]]:
    times: list[float] = []
    kinetic: list[float] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        missing = {"t", "kinetic"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            times.append(float(row["t"]))
            kinetic.append(float(row["kinetic"]))
    if len(times) < 2:
        raise ValueError(f"{path}: need at least two samples")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{path}: times are not strictly increasing")
    return times, kinetic


def dissipation(times: list[float], kinetic: list[float]) -> list[float]:
    eps = [0.0] * len(times)
    for i in range(len(times)):
        if i == 0:
            dt = times[1] - times[0]
            eps[i] = -(kinetic[1] - kinetic[0]) / dt
        elif i == len(times) - 1:
            dt = times[-1] - times[-2]
            eps[i] = -(kinetic[-1] - kinetic[-2]) / dt
        else:
            dt = times[i + 1] - times[i - 1]
            eps[i] = -(kinetic[i + 1] - kinetic[i - 1]) / dt
    return eps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", help="CSV written by --history")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="divide K and epsilon by this volume")
    parser.add_argument("--out", help="write t,K,epsilon to this CSV")
    args = parser.parse_args()

    if not math.isfinite(args.volume) or args.volume <= 0.0:
        parser.error("--volume must be positive")

    times, kinetic = read_history(args.history)
    eps = dissipation(times, kinetic)
    k_norm = [k / args.volume for k in kinetic]
    e_norm = [e / args.volume for e in eps]

    peak = max(range(len(e_norm)), key=e_norm.__getitem__)
    print(f"samples: {len(times)}")
    print(f"K(0) = {k_norm[0]:.8e}   K(end) = {k_norm[-1]:.8e}")
    print(f"peak -dK/dt = {e_norm[peak]:.8e} at t = {times[peak]:.4f} "
          f"(K = {k_norm[peak]:.8e})")

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "kinetic", "dissipation"])
            for t, k, e in zip(times, k_norm, e_norm):
                writer.writerow([f"{t:.12g}", f"{k:.12e}", f"{e:.12e}"])
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
