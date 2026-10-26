# saturnCFD microbenchmark summary

| benchmark | space | backend | grid | threads | median | min | max | runs |
|---|---|---|---|---:|---:|---:|---:|---:|
| stencil7 | openmp | OpenMP | 256x256x256 | 4 | 0.9846 ns/cell/iter | 0.9497 | 0.9931 | 3 |
| stencil7 | serial | Serial | 256x256x256 | 1 | 2.66 ns/cell/iter | 2.589 | 2.697 | 3 |
| stream:copy | openmp | OpenMP | 33554432 | 4 | 19.56 GB/s | 19.34 | 20.01 | 3 |
| stream:copy | serial | Serial | 33554432 | 1 | 18.84 GB/s | 18.22 | 19.44 | 3 |
| stream:triad | openmp | OpenMP | 33554432 | 4 | 21.98 GB/s | 21.73 | 22.38 | 3 |
| stream:triad | serial | Serial | 33554432 | 1 | 20.71 GB/s | 20.64 | 21.36 | 3 |

## Stencil efficiency vs STREAM copy

| space | stencil GB/s (median) | copy GB/s (median) | ratio |
|---|---:|---:|---:|
| openmp | 16.251 | 19.562 | 83.1% |
| serial | 6.015 | 18.835 | 31.9% |
