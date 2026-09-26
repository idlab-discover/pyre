# GPU row-chunk × tree kernel

Replaces the one-row-per-thread GPU kernel. Design (see `src/pyre/_gpu_kernel.mojo`):

- **Packed depth-first nodes.** 8 bytes per float32 node (threshold/leaf, plus a
  word holding default-distant bit, operator, 13-bit feature and 16-bit distant
  offset); 16 bytes for float64 or oversized float32 forests. The adjacent child is
  at `node + 1`, the distant child at `node + offset`, and a leaf has offset 0.
- **Normalized splits.** GT/GE exchange children, and LE becomes LT on
  `nextafter(t, +inf)`, so almost every forest runs the single-compare kernel.
  NaN thresholds keep their always-false routing. EQ and `LE +inf` use the
  general kernel.
- **Float64 integer compares.** Thresholds are stored as order-preserving int64
  keys. The RTX 5090's FP64 compare rate is 1/64 of FP32, and this was the float64
  bottleneck. The results are IEEE-identical, including ±0 and NaN.
- **Work split.** A block owns 2^k rows. Its threads split across trees and
  accumulate in registers. Partials are reduced through shared memory: one thread
  per row when there are ≤16 partials, otherwise a warp per row.
- **Launch policy.** Fitted to a 48-cell sweep on an RTX 5090, it lands within
  4% geo-mean of each cell's optimum (worst 1.24×). It uses up to 128 threads per
  block, and grows rows per block while the block's rows stay ≤512 bytes and the
  grid has ≥8 blocks per SM. Small grids widen to up to 1024 threads. Staging
  rows in shared memory was measured slower, because it shrinks the L1 that
  caches nodes, so it was removed.

GPU tree summation is no longer in model order. Results match CPU within
tolerance (float32 1e-5 / float64 1e-12 relative in the tests), not bitwise.

## Results (RTX 5090, resident completed-call latency, µs)

Exploratory single-process screen from the ignored `this-vs-nvforest/gpu-explore`
harness (nvForest 26.6.0, depth-first layout). The comparison uses the same models,
inputs and synchronization per call. Every cell passes whole-output agreement and
GTIL checks. Numbers are medians of 3 samples on a shared desktop.

| Model | Rows | Old Pyre | New Pyre | nvForest auto | nvForest best chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (64 trees, d6) | 1 | 142.4 | 7.4 | 15.8 | 8.1 |
| base | 65,536 | 181.8 | 31.3 | 74.5 | 44.8 |
| trees-1024 | 1 | 2363 | 9.1 | 161.4 | 12.9 |
| trees-1024 | 65,536 | 3157 | 512 | 1005 | 666 |
| depth-10 | 65,536 | 309.5 | 84.5 | 104.4 | 88.8 |
| features-512 | 65,536 | 765.9 | 86.5 | 438.4 | 132.4 |
| float64 | 65,536 | 326.0 | 101.0 | 130.9 | 120.9 |
| catboost-64-depth6 (f64) | 65,536 | 316.2 | 76.6 | 134.9 | 120.7 |
| balsa catboost (f64, 4 trees) | 65,536 | 17.8 | 10.0 | 49.0 | 49.4 |

In the full 18-model × {1, 33, 4096, 65536} screen before the final
narrow-reduction change, Pyre was faster than nvForest auto in 70/72 cells. The two
exceptions were the 4-tree float64 Balsa models at 65,536 rows, at 1.01–1.02×.
The narrow reduction then brought those to 10 µs against nvForest's 49 µs. The
remaining cells slower than nvForest's best probed chunk are at noise level
(trees-1 at 4096 rows, 1.14×; depth-10 at 4096, 1.07×). These are exploratory
screens, not the five-process release protocol in `docs/gpu-benchmarking.md`.

## Validation

`pixi run test-gpu` runs 13 tests, including new ones covering:
- every launch configuration against CPU on random mixed-operator forests
  (both dtypes);
- the wide float32 layout;
- ±inf, ±MAX, ±tiny, ±0 and NaN thresholds for every operator in both dtypes.

`pixi run check` passes. On the test binary, compute-sanitizer memcheck reports
0 kernel errors with `--report-api-errors no`; MAX's allocator produces benign
`cuMemCreate` API errors. Racecheck reports 0 hazards. initcheck and synccheck
were not completed.
