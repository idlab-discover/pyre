# Packaging and release

Pyre 0.2.0 targets Linux x86-64 with Mojo 1.2.0, MAX core 26.7.0, NuMojo
0.12.0 and Balsa 0.2.1. A compiled Mojo package must be consumed with the
matching compiler. CPU use does not require a GPU; GPU use is validated on an
RTX 5090.

## Installing the package

The Conda package is **`mojo-pyre`**. The name `pyre` on conda-forge belongs to
an unrelated project. Use the GitHub release asset directly in a Linux x86-64 Pixi workspace. Pixi resolves Balsa and NuMojo from modular-community.

```toml
[workspace]
name = "pyre-example"
channels = ["https://conda.modular.com/max", "conda-forge", "https://repo.prefix.dev/modular-community"]
platforms = ["linux-64"]

[dependencies]
mojo-pyre = { url = "https://github.com/idlab-discover/pyre/releases/download/v0.2.0/mojo-pyre-0.2.0-hb0f4dca_0.conda" }
gcc_linux-64 = "*"
```

It installs `lib/mojo/pyre.mojoc`, where Mojo discovers it automatically, and
depends on the matching `balsa`, `numojo`, `max-core` and `mojo-compiler`
packages. A program then needs no `-I` flags:

```sh
pixi run mojo run app.mojo
pixi run mojo run --target-accelerator=none app.mojo  # explicitly CPU-only
```

The second command builds an explicitly CPU-only consumer. Under Mojo 1.2,
NuMojo 0.12's `AcceleratorNDArray` cannot be compiled for that target, so
`pyre.accelerator` needs a build with an accelerator target. `Forest` with
`NDArray` does not. For GPU use, follow the
[GPU example](../examples/predict_gpu.mojo) and the
[storage/lifetime contracts](usage.md#native-gpu-inference).

## Build and validate

Use the checked-in lockfile and Pixi 0.80.0, matching CI:

```sh
pixi install --locked -e default -e packaging
pixi run --locked check
pixi run --locked test-gpu   # on a supported GPU host
pixi run --locked -e packaging conda-build
```

Source builds run `tools/prepare_balsa.py`, which fetches the pinned Balsa
revision into `build/balsa-reference`, the same revision the `balsa ==0.2.1`
package is built from. For offline preparation, pass
`--repository /path/to/balsa`; the repository must contain the pinned commit.
Balsa 0.2 loads packed models by default. Pyre takes an editable `Model`, so it
loads with `load_editable`, `load_auto_editable` or `decode_editable`.

`test-consumer` copies only the precompiled packages and
`examples/consumer.mojo` to a temporary directory and compiles a CPU-only
consumer with no source include path. The Conda recipe runs the same consumer
against the installed `mojo-pyre`, `balsa` and `numojo` packages. Build outputs
are under `build/conda/linux-64/`.

## Versions and build numbers

A Conda artifact is identified by name, version and build string. This recipe
has no variant keys, so the build string is always `hb0f4dca_<build number>`
and the build number alone separates artifacts of one version. The
modular-community CI builds with `--skip-existing=all`: an unchanged
name/version/build is skipped, not rebuilt.

- New source, a new compiler pin or a new dependency pin: new library version
  and tag, build number 0.
- Same tagged source, recipe-only fix (metadata, tests, license files):
  same version, build number + 1.

## Release sequence

1. Verify the pinned Balsa release is available from the community channel.
2. Merge the release commit to `main` after both GitHub CI jobs pass. The
   workspace and recipe versions must equal the release version, with build 0.
   `tools/prepare_balsa.py` must pin the commit of Balsa's release tag.
3. Tag that merge commit with an annotated `v<version>` tag and push the tag.
   The Release workflow checks that the tag is on `main` and matches the
   versions, build number 0 and `docs/release-<version>.md`. It runs the full
   CI, checks the artifact's `SOURCE_COMMIT` and `SHA256SUMS`, and creates a
   draft GitHub release with the notes, `.conda` file, checksum file and
   source-commit record.
4. Review the draft release and publish it.
For a future community-channel submission, add `recipes/mojo-pyre/recipe.yaml` in a fork of `modular/modular-community`.
   Replace `source.path`/`source.filter` with the Git repository and the tag's
   full commit as `rev`. Run the **Validate community recipe** workflow (manual
   dispatch) with the fork and commit. The workflow builds the exact recipe with upstream's
   Rattler-Build version, variants and channel order and installs it into a
   clean workspace. Then open the PR with the upstream checklist. The package
   reaches `modular-community` only after a maintainer merges it and the upload
   job runs.

Do not move an existing release tag.

## CI behavior

CI runs on branch pushes, pull requests and manual dispatch. CPU tests run on
Ubuntu in three groups. The package job builds the example, runs the Python
tooling tests and the packaged consumer, and builds the Conda package. It resolves Balsa 0.2.1 from modular-community. It uploads the Conda artifact with SHA256 checksums and the
exact checkout commit. Actions are pinned and environments locked, and the
repository permission is read-only. GPU execution is checked locally on a
supported GPU; hosted CI makes no GPU-validation claim.

`v*` tag pushes run the Release workflow instead. It calls CI and releases the
artifact that CI tested. Only its draft-release job has write permission. It
does not publish the release, overwrite an existing one, or upload to a Conda
channel.
