# Upstream references

Pyre's inference behavior and porting work reference NVIDIA RAPIDS nvForest,
commit `f4e27dbf27ca9ddf3a502133aaaf2a193aa69293`, licensed Apache-2.0.
The row traversal, missing-value routing, aggregation and output conventions
were studied from that implementation. The Mojo implementation uses Balsa's
explicit comparison operators and SIMD row batches rather than nvForest's
packed-node representation.

Reference: <https://github.com/rapidsai/nvforest>.

Balsa at sibling `../balsa/src` is a source dependency, tested at commit
`b3a3f1c`. Its model fields implement the Treelite v4 format. Refer to Balsa's
`THIRD_PARTY_NOTICES.md` for its upstream attribution. Checkpoint fixtures are owned by Balsa. Pyre records Treelite 4.6.1 prediction
expectations and source hashes without duplicating checkpoint files.

NuMojo 0.10.0 is a dependency, licensed Apache-2.0 with LLVM Exceptions.
Reference: <https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo>.
Pyre's accelerator adapter accesses NuMojo storage internals through borrowing
accessors to avoid the copied-storage pointer issue observed in that version.
NuMojo and Balsa source checkouts were not modified.
