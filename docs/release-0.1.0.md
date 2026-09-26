# Pyre 0.1.0

First release of Pyre: native Mojo CPU and GPU inference for Balsa tree ensembles,
with NuMojo arrays and no Python/libtreelite inference dependency.

- Float32/float64 numerical regression and binary classification, missing values,
  margins, probabilities, and labels.
- Structure-aware CPU dispatch for constant, compact symmetric, and general
  forests, plus bounded parallel traversal for general trees.
- Packed resident GPU inference with explicit transfers and reusable outputs.
- Explicit CPU sigmoid underflow handling preserves extreme probabilities and
  threshold-zero labels across target-specific math implementations.
- Pinned Pyre/Balsa library bundle, checksums, dependency lock, and an isolated
  consumer example; no sibling checkout required.

The measured comparison found 2.49× / 2.07× / 1.71× geometric-mean speedups over
nvForest's default chunk setting on CPU (one/eight threads) and GPU respectively.
Against the best measured chunk per case, the figures are 2.43× / 1.59× / 1.05×.
See [methodology and limitations](https://github.com/idlab-discover/pyre/blob/v0.1.0/docs/benchmarks/comparison-0.1.0.md): these compare
native Mojo and Python API calls with resident inputs, not isolated kernels.

Requires Linux x86-64, Mojo 1.0.0, MAX core 26.5.0, and NuMojo 0.10.0.
The bundle pins Balsa revision `737874f`; it does not target Balsa 0.2's
packed-default loader. GPU execution was validated on RTX 5090.
Categorical/vector/multiclass models are unsupported. Compact CPU paths remain
serial; GPU calls are synchronous and require explicit transfers.

Future work will improve larger-batch and larger-forest inference, including
parallel compact CPU execution. Saturation tests and broader hardware coverage
will guide scalable defaults for concurrent library users.

See [installation](https://github.com/idlab-discover/pyre/blob/v0.1.0/docs/distribution.md) and the [user guide](https://github.com/idlab-discover/pyre/blob/v0.1.0/docs/usage.md).
