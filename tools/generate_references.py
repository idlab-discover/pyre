"""Generate Pyre prediction oracles from Balsa's existing checkpoints.

Offline only: Treelite evaluates models; Balsa owns checkpoint generation.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import treelite

ROOT = Path(__file__).resolve().parents[1]
BALSA = ROOT / "build/balsa-reference"
SOURCE = BALSA / "tests/fixtures"
assert treelite.__version__ == "4.6.1"
lines = [
    '"""Generated Treelite GTIL predictions; see tools/generate_references.py."""',
    "from acceptance import check_dense_reference",
    "from std.math import nan, inf",
    "",
    "def check_references[on_gpu: Bool = False]() raises:",
]
manifest = {
    "treelite": treelite.__version__,
    "numpy": np.__version__,
    "balsa_commit": (BALSA / ".reference-commit").read_text().strip(),
    "fixtures": [],
}
cases = [(f"{dtype}_{kind}.tl", False) for dtype in ("float32", "float64")
         for kind in ("leaf", "op2_missing0", "deep", "sigmoid")]
cases += [(f"frameworks/{name}.tl", True) for name in (
    "xgboost_regression", "catboost_numeric_regression",
    "sklearn_gradient_binary", "sklearn_hist_missing_regression",
)]
for name, trained in cases:
    path = SOURCE / name
    model = treelite.Model.deserialize(path)
    metadata = json.loads(model.dump_as_json())
    dtype = metadata["threshold_type"]
    features = metadata["num_feature"]
    if trained:
        x = ((np.arange(33 * features).reshape(33, features) % 13 - 6) * 0.25).astype(dtype)
        x[::5, 1] = np.nan
        # Exact root thresholds and their neighbors, plus normal divergent rows.
        borders = []
        for tree in metadata["trees"]:
            node = tree["nodes"][0]
            value = np.array(node["threshold"], dtype=dtype)
            for threshold in [np.nextafter(value, -np.inf, dtype=dtype), value,
                              np.nextafter(value, np.inf, dtype=dtype)]:
                row = np.zeros(features, dtype=dtype)
                row[node["split_feature_id"]] = threshold
                borders.append(row)
        x = np.concatenate([x, np.array(borders, dtype=dtype)])
    else:
        x = np.array([-np.inf, -1, -0.0, 0, 0.25, 0.5, 1, np.inf, np.nan], dtype=dtype).reshape(-1, 1)
    margins = treelite.gtil.predict(model, x, pred_margin=True).reshape(-1)
    predictions = treelite.gtil.predict(model, x).reshape(-1)

    def literal(value):
        if np.isnan(value):
            return f"nan[DType.{dtype}]()"
        if np.isinf(value):
            return ("-" if value < 0 else "") + f"inf[DType.{dtype}]()"
        return repr(float(value))

    def values(array):
        return "[" + ", ".join(literal(v) for v in array.reshape(-1)) + "]"

    tolerance = (1e-6 if dtype == "float32" else 1e-12) if trained else 0.0
    lines += [f'    check_dense_reference[DType.{dtype}, on_gpu]("{name}",',
              f"        {values(x)},",
              f"        {values(margins)}, {values(predictions)}, {tolerance})"]
    manifest["fixtures"].append({"balsa_path": "tests/fixtures/" + name,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "rows": len(x), "tolerance": tolerance})
(ROOT / "tests/reference_data.mojo").write_text("\n".join(lines) + "\n")
(ROOT / "tests/fixtures/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
subprocess.run(["pixi", "run", "mojo", "format", "tests/reference_data.mojo"], cwd=ROOT, check=True)
