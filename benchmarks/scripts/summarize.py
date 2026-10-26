#!/usr/bin/env python3
"""Aggregate saturnCFD microbenchmark JSON files into a Markdown/JSON summary.

Usage:
    python3 benchmarks/scripts/summarize.py benchmarks/results/runs/*.json
    python3 benchmarks/scripts/summarize.py --json out.json runs/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict


def load(paths):
    records = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
            record["_file"] = path
            records.append(record)
    return records


def label(record):
    if "kernel" in record:
        return f"stream:{record['kernel']}"
    return "stencil7"


def group_key(record):
    grid = (
        f"{record['nx']}x{record['ny']}x{record['nz']}"
        if "nx" in record
        else str(record.get("n"))
    )
    return (
        label(record),
        record.get("space", "?"),
        record.get("backend", "?"),
        grid,
    )


def metrics(record):
    if "kernel" in record:
        return record["gbps"], "GB/s"
    return record["ns_per_cell_iter"], "ns/cell/iter"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSON files or glob patterns")
    parser.add_argument("--json", help="write the aggregate as JSON")
    parser.add_argument("--markdown", help="write the aggregate as Markdown")
    args = parser.parse_args()

    records = load(args.paths)
    if not records:
        raise SystemExit("no input files matched")

    groups = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)

    rows = []
    for key, group in sorted(groups.items()):
        values = [metrics(record)[0] for record in group]
        unit = metrics(group[0])[1]
        rows.append(
            {
                "benchmark": key[0],
                "space": key[1],
                "backend": key[2],
                "grid": key[3],
                "concurrency": group[0].get("concurrency"),
                "unit": unit,
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "runs": len(values),
            }
        )

    markdown = ["# saturnCFD microbenchmark summary", ""]
    markdown.append(
        "| benchmark | space | backend | grid | threads | median | min | max | runs |"
    )
    markdown.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        markdown.append(
            "| {benchmark} | {space} | {backend} | {grid} | {concurrency} | "
            "{median:.4g} {unit} | {min:.4g} | {max:.4g} | {runs} |".format(**row)
        )

    # Stencil efficiency relative to the copy benchmark on the same space.
    stream = {
        (row["space"]): row
        for row in rows
        if row["benchmark"] == "stream:copy"
    }
    stencil_bandwidth = defaultdict(list)
    for record in records:
        if "kernel" not in record:
            stencil_bandwidth[record.get("space", "?")].append(
                record["gbps_effective"]
            )
    # Prefer the 256^3 reference grid when it is present; otherwise average all.
    reference = defaultdict(list)
    for record in records:
        if ("kernel" not in record and record.get("nx") == 256
                and record.get("ny") == 256 and record.get("nz") == 256):
            reference[record.get("space", "?")].append(
                record["gbps_effective"]
            )
    if reference:
        stencil_bandwidth = reference
    if stream and stencil_bandwidth:
        markdown += ["", "## Stencil efficiency vs STREAM copy", ""]
        markdown.append("| space | stencil GB/s (median) | copy GB/s (median) | ratio |")
        markdown.append("|---|---:|---:|---:|")
        for space, values in sorted(stencil_bandwidth.items()):
            stencil_gbps = statistics.median(values)
            copy_gbps = stream[space]["median"]
            markdown.append(
                f"| {space} | {stencil_gbps:.3f} | {copy_gbps:.3f} | "
                f"{stencil_gbps / copy_gbps * 100:.1f}% |"
            )

    text = "\n".join(markdown) + "\n"
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
