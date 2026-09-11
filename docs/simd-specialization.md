**SIMD width selection**

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

The width is a Mojo [compile-time parameter](https://mojolang.org/docs/manual/parameters/),
so callers select a concrete kernel without a per-node runtime width check.

`pixi run test-specialization` checks widths 1/4/16 against fixed-tree
expectations for float32/float64 regression, margins, probabilities and labels,
including mixed operators, NaNs, infinities, ties, empty batches and tails.
The task is included in `pixi run check`.
