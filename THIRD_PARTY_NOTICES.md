# Upstream references

Pyre's inference behavior and porting work reference NVIDIA RAPIDS nvForest,
commit `f4e27dbf27ca9ddf3a502133aaaf2a193aa69293`, licensed Apache-2.0.
The row traversal, missing-value routing, aggregation and output conventions
were studied from that implementation. The Mojo implementation uses Balsa's
explicit comparison operators and SIMD row batches rather than nvForest's
packed-node representation.

Reference: <https://github.com/rapidsai/nvforest>.

Balsa (Apache-2.0) is a runtime dependency, installed as the `balsa` Conda
package. Its model fields implement the Treelite v4 format; the Balsa package
carries its own license and Treelite attribution. Source builds fetch the same
pinned Balsa revision from `https://github.com/idlab-discover/balsa.git` into
`build/balsa-reference`. Checkpoint fixtures are owned by Balsa. Pyre records
Treelite 4.6.1 prediction expectations and source hashes without duplicating
checkpoint files.

NuMojo 0.12.0 is a dependency, licensed Apache-2.0 with LLVM Exceptions.
Reference: <https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo>.
Pyre's accelerator adapter accesses NuMojo storage internals through borrowing
accessors to avoid the copied-storage pointer issue observed in NuMojo 0.10.0.
NuMojo and Balsa source checkouts were not modified.
