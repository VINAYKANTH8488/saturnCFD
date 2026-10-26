# Benchmarks

Reproducible microbenchmarks for saturnCFD.

## Microbenchmarks

| Executable | Purpose |
|---|---|
| `saturn_stencil` | 7-point 3D stencil, FP64; the dominant explicit-FVM kernel |
| `saturn_stream` | STREAM-style `copy` and `triad`; machine bandwidth roofline |

Both accept `--space serial|openmp|cuda|hip|sycl|default`, `--iters`,
`--warmup`, `--json PATH`, and `--verbose`. `--kokkos-*` options are passed to
Kokkos.

## Recording a run

```bash
source scripts/env.sh
export OMP_PROC_BIND=spread OMP_PLACES=threads

./build/cpu-openmp/apps/saturn_stencil \
  --space openmp --nx 256 --ny 256 --nz 256 --iters 50 \
  --kokkos-num-threads=4 \
  --json benchmarks/results/runs/stencil_openmp4_256_run1.json

./build/cpu-openmp/apps/saturn_stream \
  --space openmp --kernel copy --n 33554432 --iters 30 \
  --kokkos-num-threads=4 \
  --json benchmarks/results/runs/stream_copy_openmp4_run1.json
```

## Aggregating

```bash
python3 benchmarks/scripts/summarize.py \
  'benchmarks/results/runs/*.json' \
  --markdown benchmarks/results/summary.md \
  --json benchmarks/results/summary.json
```

The summarizer reports the median/min/max over repeated runs and the stencil's
effective bandwidth as a fraction of STREAM `copy` on the same execution space.

## Metrics

- **Stencil:** `ns/cell/iter` (lower is better) and `GB/s effective`
  (one read + one write per cell). `GB/s naive` (8 accesses per cell) is an
  upper bound on load traffic, not a machine-bandwidth measurement.
- **STREAM:** `GB/s` using 16 bytes/element for `copy` and 24 bytes/element for
  `triad`.

## Fair comparison rules

- Same grid, same precision (FP64), same iteration count, same thread/GPU
  count.
- Report the median of at least three runs.
- State the compiler, Kokkos version, backend, and hardware with every result.
- Compare stencil bandwidth to STREAM on the *same* backend, not to a
  datasheet number.
