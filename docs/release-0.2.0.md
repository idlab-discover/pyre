# Pyre 0.2.0

Native Mojo CPU and GPU inference for Balsa tree ensembles. Target: Linux
x86-64, Mojo/compiler 1.2.0, MAX core 26.7.0, NuMojo 0.12.0 and Balsa 0.2.1.
This is the first Pyre release distributed as a Conda package, `mojo-pyre`
(the conda-forge name `pyre` belongs to an unrelated project). It is still
imported as `pyre`.

## Distribution

| Library version | Mojo compiler | MAX core | NuMojo | Balsa | Distribution |
|---|---|---|---|---|---|
| 0.1.0 | 1.0.0 | 26.5.0 | 0.10.0 | `737874f` (bundled) | GitHub tarball |
| 0.2.0 build 0 | 1.2.0 | 26.7.0 | 0.12.0 | 0.2.1 (package) | modular-community (pending review) |

The 0.1.0 tarball bundled a Balsa build. From 0.2.0 on, Balsa is an ordinary
package dependency, and the tarball and `pixi run release` are gone. Each
compiler migration gets a new library version and tag. A build number
increment is reserved for recipe-only rebuilds of the same source.

## Changes

- **Packed node layout shared by CPU and GPU.** CPU traversal walks packed
  depth-first nodes with overlapped tree walks. Balsa's model is released after
  CPU packing.
- **CPU scheduling.** Tasks are sized by estimated work, and symmetric paths now
  run in parallel. See [CPU planning](cpu-planning.md) and
  [symmetric inference](cpu-symmetric.md).
- **GPU kernel.** Each thread walks several (row, tree) chains, rows are staged
  transposed in shared memory for both precisions, and node indices are 32-bit.
  Launch plans come from device attributes. `tune(batch)` measures candidates
  on real rows, and `use_plan` pins one. NDArray inputs pipeline uploads,
  compute and downloads across copy streams. Measurements are in
  [GPU kernel results](benchmarks/gpu-kernel.md).
- **Mojo 1.2.0.** GPU primitives and the async runtime moved under `max.*`,
  `InlineArray` became `Array`, and `parallelism_level` takes an optional
  device context. The CPU executor uses a host-target device context, so
  CPU-only builds (`--target-accelerator=none`) keep working.
- **Balsa 0.2.** Balsa 0.2 loads packed models by default, and Pyre still takes
  an editable `Model`. Load with `load_editable`, `load_auto_editable` or
  `decode_editable`.

## Known limitations

- Under Mojo 1.2, NuMojo 0.12's `AcceleratorNDArray` cannot be compiled with
  `--target-accelerator=none`, so `pyre.accelerator` needs a build with an
  accelerator target. `Forest` with `NDArray` works in CPU-only builds.
- GPU execution was validated on an RTX 5090. Hosted CI runs on CPU only.
- Categorical, vector-leaf and multiclass models remain unsupported.

## Verification

The CPU suites pass, along with the CPU-only build, Python tooling tests and the
packaged consumer outside the source tree. The GPU transfer and inference
suites (18 tests) pass on an RTX 5090. The Conda package passes its isolated
installation test, which compiles a CPU-only consumer with no source include
path. It also builds with modular-community's rattler-build 0.30.
