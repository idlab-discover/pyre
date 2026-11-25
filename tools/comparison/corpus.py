"""Train ordinary numerical regressors and preserve identical comparison inputs."""
from __future__ import annotations

import importlib.metadata
import hashlib
import itertools
import json
from pathlib import Path

from .schema import VERSION, identity, inspect_structure, parquet, sha, package_version


def catboost_model(data, features):
    """Export numeric oblivious trees as equivalent LE splits, without rounding.

    CatBoost's bit d is set by x > border. The right edge of an LE split sets
    that same bit. NaN routing follows the exported feature treatment.
    """
    from treelite.model_builder import Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation
    info = data["features_info"]
    if any(info.get(k) for k in ("categorical_features", "ctrs", "text_features", "embedding_features")):
        raise ValueError("CatBoost comparison supports numeric symmetric trees only")
    trees = data["oblivious_trees"]
    scale, bias = data["scale_and_bias"]
    if len(bias) != 1:
        raise ValueError("Expected scalar regression")
    by_feature = {f["feature_index"]: f for f in info["float_features"]}
    builder = ModelBuilder(threshold_type="float64", leaf_output_type="float64",
        metadata=Metadata(num_feature=features, task_type="kRegressor", average_tree_output=False,
                          num_target=1, num_class=[1], leaf_vector_shape=(1, 1)),
        tree_annotation=TreeAnnotation(num_tree=len(trees), target_id=[0] * len(trees), class_id=[0] * len(trees)),
        postprocessor=PostProcessorFunc(name="identity"), base_scores=bias)
    for tree in trees:
        splits = tree.get("splits") or []
        if len(tree["leaf_values"]) != 1 << len(splits) or any(s["split_type"] != "FloatFeature" for s in splits):
            raise ValueError("Unsupported CatBoost tree")
        builder.start_tree()

        def visit(node, depth, leaf):
            builder.start_node(node)
            if depth == len(splits):
                builder.leaf(scale * tree["leaf_values"][leaf])
            else:
                split = splits[depth]
                feature = by_feature[split["float_feature_index"]]
                treatment = feature["nan_value_treatment"]
                if treatment not in {"AsIs", "AsFalse", "AsTrue"}:
                    raise ValueError(f"Unknown NaN treatment {treatment}")
                builder.numerical_test(feature_id=feature["flat_feature_index"], threshold=split["border"],
                    default_left=treatment != "AsTrue", opname="<=",
                    left_child_key=2 * node + 1, right_child_key=2 * node + 2)
            builder.end_node()
            if depth < len(splits):
                visit(2 * node + 1, depth + 1, leaf)
                visit(2 * node + 2, depth + 1, leaf | (1 << depth))

        visit(0, 0, 0)
        builder.end_tree()
    return builder.commit()


def train(family, trees, depth, x, y, seed, path, training_threads=1):
    import treelite.frontend
    import treelite.sklearn
    if family == "sklearn_random_forest":
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        params = dict(n_estimators=trees, max_depth=depth, n_jobs=training_threads, random_state=seed)
        estimator = RandomForestRegressor(**params).fit(x, y)
        joblib.dump(estimator, path / "native.joblib")
        return treelite.sklearn.import_model(estimator), estimator.predict, "scikit-learn", "native.joblib", params
    if family.startswith("xgboost_"):
        import xgboost as xgb
        params = dict(n_estimators=trees, max_depth=depth, n_jobs=training_threads, random_state=seed,
                      tree_method="hist", learning_rate=0.05, grow_policy=family.removeprefix("xgboost_"),
                      max_leaves=127 if family == "xgboost_lossguide" else 0,
                      objective="reg:squarederror")
        estimator = xgb.XGBRegressor(**params).fit(x, y)
        estimator.save_model(path / "native.ubj")
        return treelite.frontend.from_xgboost(estimator.get_booster()), estimator.predict, "xgboost", "native.ubj", params
    if family == "lightgbm_leafwise":
        import lightgbm as lgb
        params = dict(objective="regression", max_depth=depth, num_leaves=min(31, 1 << depth),
                      seed=seed, num_threads=training_threads, verbosity=-1, deterministic=True, force_col_wise=True)
        estimator = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=trees)
        estimator.save_model(path / "native.txt")
        return treelite.frontend.from_lightgbm(estimator), estimator.predict, "lightgbm", "native.txt", params | {"num_boost_round": trees}
    if family == "catboost_symmetric":
        import catboost
        params = dict(iterations=trees, depth=depth, grow_policy="SymmetricTree", thread_count=training_threads,
                      random_seed=seed, verbose=False, allow_writing_files=False, loss_function="RMSE")
        estimator = catboost.CatBoostRegressor(**params).fit(x, y)
        estimator.save_model(path / "native.cbm")
        estimator.save_model(path / "native.json", format="json")
        converted = catboost_model(json.loads((path / "native.json").read_text()), x.shape[1])
        return converted, estimator.predict, "catboost", "native.cbm", params
    raise ValueError(f"Unknown family {family}")


def conversion_probe(meta, x):
    """Include split ties and adjacent representable input values."""
    import numpy as np
    # Trainers in this corpus consume float32 features, including trainers with
    # float64 exported thresholds. Probe adjacent *input* values before promotion.
    dtype = "float32"
    rows = [r.copy() for r in x[:64].astype(dtype)]
    for tree in meta["trees"][:8]:
        for node in tree["nodes"]:
            if "threshold" not in node:
                continue
            v = np.array(node["threshold"], dtype=dtype)
            for value in [np.nextafter(v, -np.inf, dtype=dtype), v,
                          np.nextafter(v, np.inf, dtype=dtype)]:
                row = np.zeros(x.shape[1], dtype=dtype)
                row[node["split_feature_id"]] = value
                rows.append(row)
            break
    return np.asarray(rows, dtype=meta["threshold_type"])


def prepare(matrix, output):
    import numpy as np
    import treelite.gtil
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if max(matrix["batch_rows"]) * matrix["features"] * 8 > matrix["max_input_bytes"]:
        raise ValueError("Input matrix exceeds byte budget")
    rng = np.random.default_rng(matrix["seed"])
    x = rng.normal(size=(matrix["training_rows"], matrix["features"])).astype("float32")
    y = (2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] ** 2 + np.sin(x[:, 3])
         + 0.1 * x[:, 4:].sum(axis=1) + rng.normal(0, 0.1, len(x))).astype("float64")
    # Held-out rows are shared by every training family; batches are prefixes.
    test = rng.normal(size=(max(matrix["batch_rows"]), matrix["features"])).astype("float32")
    np.savez(output / "training.npz", x=x, y=y)
    packages = {"numpy", "treelite"}
    for family in matrix["families"]:
        packages.add({"sklearn_random_forest": "scikit-learn", "catboost_symmetric": "catboost",
                      "lightgbm_leafwise": "lightgbm", "xgboost_depthwise": "xgboost-cpu",
                      "xgboost_lossguide": "xgboost-cpu"}[family])
    versions = {p: package_version(p) for p in sorted(packages)}
    manifest = {"schema_version": VERSION, "matrix": matrix, "versions": versions,
                "source_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("schema.py"), Path(__file__).with_name("requirements.lock")]},
                "training_task": "Nonlinear regression on four primary features, linear contributions from remaining features, Gaussian noise sigma=0.1",
                "training_sha256": identity({"x": hashlib.sha256(x.tobytes()).hexdigest(),
                                             "y": hashlib.sha256(y.tobytes()).hexdigest()}),
                "models": [], "excluded": []}
    for family, count, depth in itertools.product(matrix["families"], matrix["tree_counts"], matrix["depth_limits"]):
        requested = dict(family=family, requested_tree_count=count, requested_depth_limit=depth)
        # Bound worst-case construction before training/expanding a checkpoint.
        bound = (1 << (depth + 1)) - 1
        if family != "catboost_symmetric":
            bound = min(bound, 2 * matrix["training_rows"] - 1)
        if family == "lightgbm_leafwise":
            bound = min(bound, 61)
        if family == "xgboost_lossguide":
            bound = min(bound, 253)
        if count * bound > matrix["max_model_nodes"]:
            manifest["excluded"].append(requested | {"reason": "conservative node budget", "upper_bound_nodes": count * bound})
            continue
        name = f"{family}-t{count}-d{depth}"
        path = output / name
        path.mkdir()
        model, predict, framework, native, params = train(family, count, depth, x, y, matrix["seed"], path, matrix.get("training_threads", 1))
        meta = json.loads(model.dump_as_json())
        shape = inspect_structure(meta)
        if shape["node_count"] > matrix["max_model_nodes"]:
            raise ValueError(f"{name}: actual node budget exceeded")
        model.serialize(path / "model.tl")
        dtype = meta["threshold_type"]
        tolerance = 2e-5 if dtype == "float32" else 1e-12
        probe = conversion_probe(meta, test)
        oracle = treelite.gtil.predict(model, probe, nthread=1).reshape(-1)
        original = np.asarray(predict(probe)).reshape(-1)
        # Some native APIs quantize input to float32 internally; record a separate
        # conversion tolerance rather than pretending the precision is identical.
        conversion_tolerance = 2e-5 if framework in {"scikit-learn", "xgboost"} else 1e-12
        np.testing.assert_allclose(oracle, original, atol=conversion_tolerance, rtol=conversion_tolerance)
        model_hash = sha(path / "model.tl")
        model_id = identity(requested | {"model_sha256": model_hash, "training_sha256": manifest["training_sha256"]})
        item = requested | shape | dict(model_id=model_id, model_name=name, framework=framework, model_dtype=dtype,
            features=meta["num_feature"], task="regression", training_parameters=params,
            native_path=f"{name}/{native}", native_sha256=sha(path / native),
            model_path=f"{name}/model.tl", model_sha256=sha(path / "model.tl"),
            tolerance=tolerance, native_tolerance=conversion_tolerance,
            conversion_probe_rows=len(probe), conversion_max_abs_error=float(np.max(np.abs(oracle-original))), inputs=[])
        for rows in matrix["batch_rows"]:
            samples = test[:rows].astype(dtype)
            expected = treelite.gtil.predict(model, samples, nthread=1).reshape(-1).astype("float64")
            if not np.all(np.isfinite(expected)):
                raise ValueError("Nonfinite oracle")
            inp, ref = path / f"x-{rows}.bin", path / f"oracle-{rows}.bin"
            samples.tofile(inp)
            expected.tofile(ref)
            item["inputs"].append(dict(batch_rows=rows, input_path=str(inp.relative_to(output)),
                                       input_sha256=sha(inp), oracle_path=str(ref.relative_to(output)), oracle_sha256=sha(ref)))
        manifest["models"].append(item)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print("prepared", name, flush=True)
    if not manifest["models"]:
        raise ValueError("Matrix produced no models within budget")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    fields = ["model_id", "model_name", "framework", "family", "model_dtype", "features", "requested_tree_count",
              "requested_depth_limit", "tree_count", "max_depth", "node_count", "tree_type", "symmetric_tree_count",
              "model_sha256", "native_sha256", "conversion_probe_rows", "conversion_max_abs_error"]
    model_rows = []
    for model in manifest["models"]:
        row = {k: model[k] for k in fields}
        row.update(schema_version=VERSION, training_sha256=manifest["training_sha256"],
            training_parameters_json=json.dumps(model["training_parameters"], sort_keys=True),
            nodes_per_tree=[t["nodes"] for t in model["trees"]],
            depth_per_tree=[t["max_depth"] for t in model["trees"]],
            min_leaf_depth_per_tree=[t["min_leaf_depth"] for t in model["trees"]],
            mean_leaf_depth_per_tree=[sum(t["leaf_depths"]) / t["leaves"] for t in model["trees"]],
            symmetric_per_tree=[t["symmetric"] for t in model["trees"]])
        model_rows.append(row)
    parquet(output / "models.parquet", model_rows, "models")
