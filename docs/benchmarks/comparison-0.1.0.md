# Pyre 0.1.0 comparison snapshot

Measured 2026-09-26 on an AMD Ryzen 9 5950X (16 cores / 32 logical CPUs)
and NVIDIA GeForce RTX 5090. The measurement run contains 52,275 intervals:
17,425 successful process attempts and 1,025 explicit unsupported native-GPU
attempts, with no failed, duplicate, or missing planned attempts.

The corpus contains 41 numerical scalar regression models: eight each from
scikit-learn random forest, CatBoost SymmetricTree, XGBoost depthwise, and
XGBoost lossguide, plus nine LightGBM leafwise models. Requested tree counts
were 16/64/256 and depth limits 4/8/12; conservative model-size bounds excluded
four combinations. Actual tree structure, rather than requested depth alone,
describes the resulting models. Inputs are deterministic held-out numerical
data, with 32 features and batches of 1/32/256/4,096/65,536 rows.

Each cell has five independent processes, each with three calibrated intervals
of at least 100 ms. A full-output GTIL correctness gate precedes timing. Latency
is elapsed time divided by loop count; summaries take the median within each
process, then the median across processes. Aggregate speedup is the geometric
mean of matched competitor/Pyre latency ratios, equally weighting model/batch
cases. CPU comparisons match requested thread limits. Win counts use a strict
ratio greater than one and do not establish statistical significance.

## Results

| Execution | Speedup vs nvForest default chunk | Faster cases | Speedup vs best measured chunk | Faster cases |
| --- | ---: | ---: | ---: | ---: |
| CPU, 1 thread | 2.49× | 182/205 | 2.43× | 181/205 |
| CPU, 8 threads | 2.07× | 189/205 | 1.59× | 142/205 |
| GPU | 1.71× | 195/205 | 1.05× | 108/205 |

The best measured chunk is chosen retrospectively for each case from automatic,
1, 8, and 32, with depth-first layout and alignment zero fixed. This is a limited
settings probe, not exhaustive nvForest tuning. Pyre uses production dispatch.

Against their original CPU predictors, Pyre wins every measured scikit-learn
and LightGBM case and every framework/model at batches of 1 and 32 rows.
At 65,536 rows, native XGBoost is about 1.3–1.4× faster in aggregate. Native
CatBoost benefits from parallel scaling that Pyre's serial compact path lacks:
at eight threads and 65,536 rows, CatBoost is 1.88× faster in aggregate.
Some larger GPU forests also favor nvForest: a 256-tree, depth-8 random forest
at 65,536 rows takes 1.32 ms in Pyre versus 0.54 ms with nvForest's default chunk.

## Interpretation and provenance

Models and inputs are resident before timing. Each call allocates output,
executes prediction, completes, and disposes the output. Model loading,
preparation, uploads, downloads, warmup, calibration, and process startup are
excluded. Pyre uses native Mojo calls; references use Python APIs, including
binding and conversion overhead. These results cannot isolate kernel speed,
per-request tail latency, or end-to-end GPU transfer costs.

The runs were serial across scenarios, and CPU limits were one or eight
workers. This is not a saturation or concurrent-serving study. No native
framework GPU implementation was measured. The corpus does not establish
performance on classifiers, categorical models, real application datasets,
or other CPU/GPU architectures.

Versions: Mojo 1.0.0, MAX core 26.5.0, NuMojo 0.10.0, nvForest 26.6.0,
scikit-learn 1.9.1, LightGBM 4.7.0, CatBoost 1.2.10, XGBoost 3.4.1,
Treelite 4.6.1. GPU driver: 615.71.09.

The collector is preserved separately on `feat/comparison-matrix`, commit
`47da87d`; it is not part of the 0.1.0 library release. Local raw evidence lives
at `build/comparison-20260926-172901/`, run
`eeedeea0-9827-4000-9781-d3833c053e9e`. Raw measurements are not distributed with
this summary. The collector commit adds tooling to runtime revision `856f875`;
release preparation changes packaging and documentation, not inference kernels.
