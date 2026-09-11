**SIMD width and LT specialization**

All four `Forest` prediction methods accept a compile-time width:

```mojo
var y = forest.predict[16](x, num_threads=1)
var margins = forest.predict_margin[16](x, num_threads=1)
# For binary classifiers:
var probabilities = classifier.predict_proba[16](x, num_threads=1)
var labels = classifier.predict_class[16](x, threshold=0.5, num_threads=1)
```

Width must be a positive power of two. Existing calls keep their target-native
`simd_width_of[dtype]()` default. Width 16 is opt-in: a logical group may span
multiple registers, and the best width depends on the CPU, model and batch.
CPU accelerator adapters continue to use the default width. Output allocation,
thread selection and execution-context behavior are unchanged.

During the existing construction/validation walk, Pyre records whether every
non-leaf numerical split is LT. Empty sets of splits (leaf-only forests) are
eligible. Prediction checks that immutable flag once and calls either
`_predict_impl[..., True]` or `_predict_impl[..., False]`. The flag is a runtime
property of the loaded model; the kernel's `lt_only` parameter is compile-time.
`comptime if lt_only` removes operator loads and general operator comparisons
from scalar and vector traversal. The generic kernel continues to handle all
supported operators, including mixtures within and across trees. Missing-value
routing, masked gathers, summation order and transformations are shared.

This follows Mojo's [compile-time parameterization](https://mojolang.org/docs/manual/parameters/)
and [compile-time conditionals](https://mojolang.org/docs/manual/metaprogramming/comptime-evaluation/):
only the selected constant branch is materialized in each specialization. No
checkpoint is evaluated at compile time and no runtime compilation is needed.
The compiler may emit both kernels for a used width/dtype/output combination;
this trades some code size for less work in the inner loop.

`pixi run check` builds the example and package and runs the native, parallel and
specialization suites. The focused `pixi run test-specialization` suite checks
widths 1/4/16 for both precisions and all four public operations against fixed-tree
expectations. It includes late non-LT nodes/trees, ties, NaNs, infinities, empty
batches, tails, averaging/bias and sigmoid alpha. Existing suites cover every
operator, Treelite reference outputs, zero-feature forests, offset views, deep
lane divergence, validation errors and concurrent callers.

**Measured implementation results (11 September 2026)**

Serial native prediction latency on a Ryzen 9 5950X, CPUs 8–15. Each result is
three pyperf processes × five values, two warmups, calibrated ≥0.1-second loops.
Both implementations allocate/free an output per iteration; model loading,
startup and correctness dumps are excluded. Same model/input bytes as the pinned
comparison; baseline Pyre is `f317662`.

| Case | Baseline µs | Current default µs | Current width 16 µs | Default reduction | Width-16 reduction |
|---|---:|---:|---:|---:|---:|
| small-f32 | 48.27 | 40.25 | 33.71 | 16.6% | 30.2% |
| batch-f32 | 5652.91 | 4924.39 | 4064.42 | 12.9% | 28.1% |
| batch-f64 | 8419.16 | 7015.89 | 4509.61 | 16.7% | 46.4% |
| trained-xgboost | 188.10 | 159.36 | 141.77 | 15.3% | 24.6% |

These four performance cases are LT-only regression forests. Both production
workers passed 28 exact output comparisons across the original one/eight-thread
cases; all 23 tests in `pixi run check` passed. The fallback and classifier paths
are correctness-tested, but these timings do not measure their performance.

Pyperf reports sample-stability warnings on this shared workstation. Raw standard
deviations and warnings are retained locally, and no sub-percent, cross-CPU or
threaded speedup claim is made. Wider vectors remain an explicit choice.

Raw measurements, reproducible commands, hashes and a full comparison including
nvForest are in `this-vs-nvforest/implementation-1-2/README.md`. That directory is
ignored by Git; this document retains the implementation summary. Disassembly
contains both `lt_only=true` and `lt_only=false` versions for each tested dtype;
the true variants contain fewer vector comparison instructions.
