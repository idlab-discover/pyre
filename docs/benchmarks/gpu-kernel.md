# GPU row-chunk × tree kernel

## Multi-chain traversal, transposed staging and launch plans (2026-11-09)

Changes since 9d1f729 (see `src/pyre/_gpu_kernel.mojo` and `src/pyre/gpu.mojo`):

- **Chains per thread.** A thread walks 1, 2 or 4 independent (row, tree) chains
  in lockstep (`rows_per_thread` × `trees_per_thread`), so several dependent node
  loads are in flight per thread. Finished chains step by zero.
- **Transposed staging, both precisions.** Staged rows sit in shared memory as
  `stage[feature * (chunk + 1) + row]`. Lanes on distinct rows hit distinct banks,
  and the odd stride keeps the coalesced fill conflict-free. The earlier row-major
  staging put a whole warp on one bank at the root (32 features → stride 32 words),
  which is why staging had measured slower. Stages above 48 KiB opt in to the
  device's larger dynamic shared memory.
- **Launch plans from the device.** `GpuPlan` replaces the RTX 5090-fitted policy.
  The default derives from discovered SM count, warp size, thread/block limits and
  shared memory, and from the forest's trees, mean depth, node bytes and row width.
  `tune(batch)` measures candidates on real rows (rows per block × threads × staging
  jointly, then chain shapes) and keeps the best per power-of-two bucket.
  `use_plan` pins one. The warp reduction follows the target's `WARP_SIZE`.
- **Pipelined host arrays.** NDArray overloads of the four prediction methods
  overlap slab uploads with compute and downloads across two cached copy streams.

Method: RTX 5090 desktop with an active display, native completed calls, resident
input and reused output. Each cell is the median of three samples of at least
20 ms, from one process; the old and new kernels ran back to back. All cells pass
whole-batch agreement with nvForest and GTIL checks on 257 rows (`explore.py`), and
every plan in `test_launch_configurations` matches CPU. Hardware counters were
unavailable (`RmProfilingAdminOnly=1`), so plans were chosen by timing alone.
Run-to-run drift on this shared desktop reached about 10%; compare within a row.

Completed-call latency (µs). The nvForest best column is the best of depth/breadth
layout × chunk {auto, 1, 8, 32}.

| Model | Rows | Pyre 9d1f729 | Pyre default | Pyre tuned | nvForest auto | nvForest best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base (64 trees, d6) | 65,536 | 31.9 | 24.5 | 24.5 | 74.2 | 44.5 |
| trees-1 | 65,536 | 14.8 | 11.9 | 9.4 | 19.3 | 19.3 |
| trees-256 | 65,536 | 138.2 | 82.9 | 79.3 | 262.7 | 181.5 |
| trees-1024 | 65,536 | 507.4 | 286.4 | 287.9 | 1005.7 | 664.8 |
| depth-10 | 65,536 | 81.4 | 61.2 | 45.0 | 104.2 | 90.4 |
| features-512 | 65,536 | 88.8 | 91.9 | 85.3 | 438.3 | 132.3 |
| float64 | 65,536 | 53.7 | 44.7 | 44.7 | 131.4 | 119.5 |
| catboost-64-depth6 (f64) | 65,536 | 49.7 | 40.5 | 39.6 | 134.5 | 117.1 |
| trees-1024 | 4,096 | 60.6 | 36.1 | 36.5 | 208.1 | 59.9 |
| depth-10 | 4,096 | 15.5 | 16.1 | 12.8 | 26.6 | 14.6 |
| features-512 | 4,096 | 12.2 | 13.1 | 11.8 | 29.1 | 12.3 |

At 262,144 rows (Pyre only): trees-1024 1904 → 994 µs (default), depth-10
319 → 180 → 156 (tuned), trees-1 37.4 → 23.4 → 17.2, features-128
146 → 139 → 111.

In the full 18-model × {1, 33, 4096, 65536} nvForest screen, Pyre's default plans
were faster than nvForest auto in 71/72 cells. The exception was trees-1 at one
row (7.42 vs 7.36 µs), which is launch latency. Against nvForest's best probed
setting on the eight heavy models above at 4,096 and 65,536 rows, default plans
won 13/16 cells and tuned plans 15/16. The remaining losses and the trees-1 tie
are at most 10% and 0.1-1.5 µs. Below a few thousand rows every engine sits on
the ~7 µs launch-and-synchronize floor.

Tuning took 22-560 ms per bucket in these runs, and 1.2 s for trees-1024 at
262,144 rows. The default plan is within a few percent of tuned for most
models. The largest remaining gaps are deep trees (depth-10, 1.36×) and wide
or single-tree forests.

Host-array prediction at 262,144 rows (host input → host output, best of 7):

| Model | upload + predict + to_host | pipelined |
| --- | ---: | ---: |
| trees-1024 | 3160 µs | 2414 µs |
| trees-256 | 2467 | 2271 |
| float64 | 4829 | 4589 |
| base | 2283 | 2227 |
| features-512 | 37,398 | 37,524 |

Uploads from pageable host memory are bound by host memory bandwidth (~15-16 GB/s
here; pinned DMA reaches 28 GB/s, but copying into pinned memory first runs at
the same ~15 GB/s). Pipelining hides kernels and downloads, so its gain grows
with kernel time.

Experiments that measured slower and were not kept: summing leaf values after
the walk instead of on its last pass (2-4% slower, one more dependent load),
and `not (v >= t)` NaN routing (2-4% slower, a divergent branch).

## Earlier: replacing the one-row-per-thread kernel

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

`pixi run test-gpu` runs 14 tests, including new ones covering:
- every launch configuration against CPU on random mixed-operator forests
  (both dtypes);
- the wide float32 layout;
- ±inf, ±MAX, ±tiny, ±0 and NaN thresholds for every operator in both dtypes;
- oversized batches rejected before allocation or launch.

The [2026-09-26 consolidation review](../branch-cleanup.md) validates the combined
CPU/GPU candidate. All four Compute Sanitizer tools pass with device and host
allocation pools disabled and API error reporting enabled. This supersedes the
initial partial memcheck/racecheck run with API errors suppressed.
