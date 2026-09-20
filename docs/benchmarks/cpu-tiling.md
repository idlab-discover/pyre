# CPU row/tree tiling default

CPU inference uses 256-row tiles and consecutive groups of four trees within each
worker's existing row partition. This default prioritizes larger prediction batches
and deeper/larger forests, accepting measured overhead on tiny calls. It does not
change precision, tree accumulation order, SIMD dispatch or worker selection.

A worker owns one scratch allocation per prediction, reused across its tiles.
At SIMD widths up to 256, payload is 1 KiB for FP32 or 2 KiB for FP64 per worker;
wider explicit widths require one vector per tile. Each row's accumulator carries
through all tree groups. Averaging, the FP64 base-score addition, and the output
transform run after the last group. Tile/group sizes are internal constants.

## Evidence and tradeoff

The private experiment compared frozen revision
`b3f155126e29694c833c94b6719577eb334a7788` with runtime-configured tiling on a Ryzen 9
5950X pinned to CPUs 8–15. The shortlist was 256 rows / four trees. Confirmation
covered ten workloads, explicit and automatic SIMD, and worker requests 1/4/8/auto,
with five fresh processes per arm and alternating order. Timings included output
and scratch costs and excluded model/input setup.

For explicit SIMD-8 serial execution, median paired throughput ratios were 1.149x
for the large forest and 1.337x for the deep forest. Across tiny-call configurations,
ratios were 0.792–0.888x. Wide-input, automatic SIMD and threaded results were mixed;
this is a workload-priority decision, not a claim of universal improvement.
Separate counter profiles showed approximately 96%/94% fewer L2 data misses per
retired instruction in the large/deep cases. Profiles used multiplexed counters;
they explain the likely cache benefit rather than establish statistical confidence.

The initial experiment recommended retaining baseline because of those controls.
Adoption explicitly accepts that tradeoff because expected use emphasizes larger,
deeper forests and more samples. No size-based fallback or public tuning API is
introduced. The adopted implementation fixes the experimental settings as constants.

## Validation and reproduction

The experiment passed the saved-oracle suites and 279,936 bitwise output-array
comparisons against the frozen baseline. The committed regression test covers
cancellation-sensitive tree-group boundaries, row-tile boundaries, worker partitions,
offset views, both precisions, averaging, and FP64 base scores. Run `pixi run check`
for the production build/package and CPU suites. The adopted code passed all 29
CPU tests and 23,328 additional bitwise array comparisons against the frozen baseline.

A final five-process-per-arm confirmation of the fixed-constant build found median
paired speedups of 1.205x large / 1.301x deep for serial SIMD-8, and 1.146x / 1.169x
for automatic SIMD with automatic workers. These 80 selected-workload samples
confirm the intended use cases without removing the broader tradeoffs above.

Probability benchmarks now honor the requested explicit SIMD width. Use
`pixi run benchmark --width 8` for explicit width and `--width 0` for automatic
dispatch; do not mix those measurement contracts. The original private snapshots,
commands, raw timing/counter samples, hashes and adoption checks are retained locally
under `docs/private/benchmarks/cpu-tiling/` and remain outside version control.
