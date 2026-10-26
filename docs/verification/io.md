# Verification: output and checkpointing

**Date:** 2026-10-26
**Status:** VTK snapshot and binary checkpoint/restart pass on 1 and 2 ranks,
CPU and CUDA.
**Code:** `src/io/vtk_writer.hpp`, `src/io/checkpoint.hpp`,
`tests/test_io.cpp`.

## VTK snapshots

`write_vti` writes a cell-centered scalar field as VTK ImageData:

- single rank: `<base>.vti`
- multiple ranks: `<base>.rankNNNN.vti` plus `<base>.pvti` (rank 0)

The `.pvti` collection references each rank's piece with its global extent, so
ParaView opens the parallel dataset directly. The writer is ASCII for
portability; binary/appended data and HDF5/ADIOS2 are later performance work.

The scalar MMS app exposes it as:

```bash
mpiexec -n 2 ./build/cpu-openmp/apps/saturn_scalar_mms \
  --n 32 --nu 0.01 --tend 0.05 --output /tmp/mms_out \
  --kokkos-num-threads=1
# -> /tmp/mms_out.pvti, /tmp/mms_out.rank0000.vti, ...
```

## Binary checkpoint

Each rank writes `<base>.rankNNNN.bin`:

```
CheckpointHeader (magic, version, rank/size, nvar/ng,
                  local/global extents, step, time)
raw float64 field, [v][k][j][i], x fastest
```

`read_checkpoint` validates the magic, version, rank/size, field metadata, and
extents before reading. The test writes step 7 at t=1.25, zeroes a second field,
reads it back, and checks every interior cell. Max error is 0 on 1 and 2 ranks
(CPU and CUDA), and the global extents are preserved.

The header is POD and assumes the same endianness/ABI on write and read; a
portable format (HDF5, ADIOS2) is the production path.

## Tests

```bash
ctest --test-dir build/cpu-openmp --output-on-failure
```

CTest includes `saturn.io` (1 rank) and `saturn.io_mpi` (2 ranks), alongside the
halo, scalar MMS, and MPI tests.

## Remaining I/O work

- HDF5 (parallel MPI-IO) or ADIOS2 backend behind the same interface.
- Binary/appended VTK data for large snapshots.
- Time-series `.pvd` collection.
- Checkpoint restart in the production driver (the low-level read/write is
  verified).
