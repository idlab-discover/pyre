# Treelite attribution

Balsa's checkpoint field layout and serialization behavior are adapted
from the Treelite v4 specification and implementation:

- Project: https://github.com/dmlc/treelite
- Source reference: `a2cd458e2140052a2234402835bd02815d11458e`
- License: Apache License 2.0; reproduced in [LICENSES/Treelite.txt](LICENSES/Treelite.txt).
- Serializer notices: Copyright (c) 2021–2023 by Contributors; author Hyunsu Cho.

The Balsa implementation translates the format into Mojo, adds owned storage
and bounds/structural validation, and preserves unknown extension records.
The Python oracle uses Treelite 4.6.1 to produce and check synthetic fixtures.
Detailed source links are recorded in [the MVP plan](docs/mvp-plan.md).

This notice preserves upstream attribution; it does not select a license for
the original Balsa contributions.
