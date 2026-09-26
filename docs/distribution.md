# Distribution and CI

Pyre 0.1.0 targets Linux x86-64 with Mojo 1.0.0, MAX core 26.5.0, and
NuMojo 0.10.0. A compiled Mojo package must be consumed with the matching
compiler. CPU use does not require a GPU; GPU use is validated on RTX 5090.

## Release bundle

Download `pyre-0.1.0-linux-64.tar.gz` and `SHA256SUMS` from the
[GitHub release](https://github.com/idlab-discover/pyre/releases/tag/v0.1.0).
Then, from the download directory:

```sh
sha256sum -c SHA256SUMS
tar -xzf pyre-0.1.0-linux-64.tar.gz
cd pyre-0.1.0-linux-64
pixi install --locked
pixi run consumer
```

The bundle includes `lib/pyre.mojoc` and `lib/balsa.mojoc`, a standalone consumer,
the dependency lock, notices/licenses, and a manifest of source and library
hashes. It does not bundle the Mojo/MAX/NuMojo runtime; Pixi installs those
pinned dependencies. It does not require a sibling Balsa or Pyre source tree.
This is a GitHub library bundle, not a published Conda-channel package.

For your own program, use this environment and point `-I` at the extracted
`lib` directory:

```sh
pixi run mojo run -I /absolute/path/to/pyre-0.1.0-linux-64/lib /path/to/app.mojo
```

Use `--target-accelerator=none` for an explicitly CPU-only consumer. The bundled
`consumer.mojo` builds a small forest in memory and verifies float32/float64
prediction, so no checkpoint fixtures are needed to verify installation.
For GPU use, follow the [GPU example](../examples/predict_gpu.mojo) and
[storage/lifetime contracts](usage.md#native-gpu-inference).

## Balsa compatibility

Pyre 0.1.0 uses Balsa's editable `Model` API at immutable revision
`737874f3eae9d95a0517397130f445b303d28cc6`. The bundled Balsa is deliberately
pinned to that API; do not replace it with Balsa 0.2's packed-default loader.
Balsa owns file loading; Pyre validates supported inference semantics at
construction. There is no implicit precision conversion or device migration.

Source builds run `tools/prepare_balsa.py`, which fetches the exact revision
from the public Git repository. An existing matching snapshot is reused.
For offline preparation, pass `--repository /path/to/balsa` to that script;
the repository must contain the pinned commit. No working-tree files are read.

## Build and verification

```sh
pixi install --locked
pixi run check
pixi run test-gpu  # On a supported GPU host
pixi run release
```

`package` builds both libraries. `test-consumer` copies only those packages
and a consumer to a temporary directory outside the repository, then compiles
and executes without source include paths. `release` repeats that check and
creates the bundle and checksums under `build/release/`.

GitHub Actions follows Balsa's pinned-action, locked-environment setup. CPU
checks run on Ubuntu for main/feature/release branches, pull requests, and tags.
CI also builds the release bundle, installs its lock in an isolated directory,
and runs its consumer. A version tag publishes the checked bundle and notes
as a GitHub release after those checks pass. The tag must match the manifest
version. GPU execution is checked locally on a supported GPU; hosted CPU CI
makes no GPU-validation claim.
