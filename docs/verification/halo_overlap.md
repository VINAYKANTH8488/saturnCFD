# Halo overlap

**Date:** 2026-10-26
**Status:** implemented and correct; **opt-in** (`--overlap`) because it is not
yet a win on the tested CPU/GPU sizes.

## What is overlapped

`FaceExchange` now exposes:

```cpp
exchange.begin(q);   // pack + post Isend/Irecv, do not wait
// compute interior RHS (one cell away from every local face)
exchange.end(q);     // Waitall + copy back + unpack ghosts
// compute the boundary shell (six faces)
```

The scalar solver uses a **split RHS**:

- `interior_box = box.shrink(1)` is safe while MPI ghosts are in flight.
- after `end`, `scalar_advection_diffusion_rhs_shell` computes the six face
  slabs, partitioned so every shell cell is visited exactly once (x faces all
  tangential cells, y faces excluding x edges, z faces excluding x/y edges).

Both paths produce identical results; the overlap path is covered by
`saturn.scalar_mms_overlap_mpi` in CTest.

## Why it is opt-in

Overlap only helps when the exchange is longer than the **interior** RHS:

```
T_overlap  = max(T_exchange, T_interior_rhs) + T_shell_rhs
T_serial   = T_exchange + T_interior_rhs + T_shell_rhs
```

So overlap wins only if `T_exchange > T_interior_rhs`; otherwise the extra
shell launch is pure overhead. Measured (1 Kokkos thread per rank):

| Case | Serialized | `--overlap` | Result |
|---|---:|---:|---|
| CPU, 4 ranks, 256³, 8 steps | 1.324 s | 1.420 s | 7% slower |
| CPU, 8 ranks, 256³, 8 steps | 1.304 s | 1.441 s | 11% slower |
| P600, 2 ranks, 128³, 36 steps | 1.202 s | 1.240 s | 3% slower |
| P600, 1 rank, 128³ | 1.123 s | 1.120 s | no MPI; no-op |

At these sizes the exchange is a few percent of the RHS, so there is nothing
to hide. At much larger GPU counts the exchange fraction grows (latency,
global communication, host staging), which is where this path should be
re-evaluated.

## What would make it win

1. **GPU streams**: pack on a separate stream from the interior kernel, and
   start the exchange before the RHS instead of after a blocking pack.
2. **GPU-aware MPI**: remove the host staging copy that currently serializes
   most of the GPU exchange.
3. **Larger exchange/compute ratio**: benchmark at 64–256 GPUs, where the
   Poisson/coarse communication and halo latency dominate.
4. **Fuse pack into the boundary RHS**: avoid reading boundary cells twice.

Until then, default is the serialized `exchange()` path; use `--overlap` to
test the overlapped path explicitly.

## Reproduce

```bash
# serialized
/usr/bin/mpiexec -n 4 ./build/cpu-openmp/apps/saturn_scalar_mms \
  --space openmp --n 256 --nu 0.01 --tend 0.02 --kokkos-num-threads=1

# overlapped
/usr/bin/mpiexec -n 4 ./build/cpu-openmp/apps/saturn_scalar_mms \
  --space openmp --n 256 --nu 0.01 --tend 0.02 --overlap \
  --kokkos-num-threads=1
```
