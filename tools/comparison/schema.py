"""Versioned matrix and raw measurement contracts; no ranking or aggregation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

VERSION = 1
FAMILIES = {"sklearn_random_forest", "xgboost_depthwise", "xgboost_lossguide",
            "catboost_symmetric", "lightgbm_leafwise"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_matrix(path):
    m = json.loads(Path(path).read_text())
    return validate_matrix(m)


def validate_matrix(m):
    if m.get("schema_version") != VERSION:
        raise ValueError("Unsupported matrix schema_version")
    allowed = {"schema_version", "purpose", "seed", "families", "tree_counts", "depth_limits",
               "batch_rows", "features", "training_rows", "max_model_nodes",
               "max_input_bytes", "devices", "engines", "cpu_threads", "nvforest_chunks",
               "pyre_gpu_row_groups", "processes", "samples", "min_seconds",
               "worker_timeout_seconds"}
    allowed |= {"training_threads", "native_input"}
    if set(m) - allowed or allowed - {"training_threads", "native_input"} - set(m):
        raise ValueError(f"Matrix fields mismatch: {set(m) ^ allowed}")
    if m["purpose"] not in {"validation", "measurement"}:
        raise ValueError("Unknown matrix purpose")
    if type(m["seed"]) is not int or not 0 <= m["seed"] < 2**32:
        raise ValueError("seed must be a uint32 integer")
    for field, values in [("families", FAMILIES), ("devices", {"cpu", "gpu"}),
                          ("engines", {"pyre", "nvforest", "native"})]:
        if not m[field] or len(set(m[field])) != len(m[field]) or not set(m[field]) <= values:
            raise ValueError(f"Invalid {field}")
    for field in ["tree_counts", "depth_limits", "batch_rows", "cpu_threads"]:
        if not m[field] or any(type(v) is not int or v < 1 for v in m[field]):
            raise ValueError(f"{field} requires positive integers")
        if len(set(m[field])) != len(m[field]):
            raise ValueError(f"Duplicate {field}")
    for field in ["features", "training_rows", "max_model_nodes", "max_input_bytes", "processes", "samples"]:
        if type(m[field]) is not int or m[field] < 1:
            raise ValueError(f"Invalid {field}")
    if m["features"] < 4 or max(m["depth_limits"]) > 16 or m["max_model_nodes"] > 1000000:
        raise ValueError("Require >=4 features, depth <=16, and <=1000000 model nodes")
    for field in ["min_seconds", "worker_timeout_seconds"]:
        if not 0 < m[field] < 86400:
            raise ValueError(f"Invalid {field}")
    for field, maximum in [("nvforest_chunks", 32), ("pyre_gpu_row_groups", 32)]:
        if not m[field] or len(set(m[field])) != len(m[field]):
            raise ValueError(f"Invalid {field}")
        for value in m[field]:
            if value is not None and (type(value) is not int or value < 1 or value > maximum or value & (value - 1)):
                raise ValueError(f"{field} requires null or powers of two <= {maximum}")
    if type(m.get("training_threads", 1)) is not int or m.get("training_threads", 1) < 1:
        raise ValueError("training_threads must be positive")
    if m.get("native_input", "prepared") not in {"prepared", "ndarray"}:
        raise ValueError("native_input must be prepared or ndarray")
    if "gpu" in m["devices"] and "native" in m["engines"]:
        if not set(m["families"]) <= {"xgboost_depthwise", "xgboost_lossguide"}:
            raise ValueError("Resident native GPU comparisons support XGBoost only; use engines pyre/nvforest for other families")
        if m.get("native_input", "prepared") != "prepared":
            raise ValueError("Native GPU comparisons require prepared device inputs")
    return m


def parquet(path, rows, kind):
    """Explicit nullable types also give empty tables stable schemas."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    common = [("schema_version", pa.int32()), ("run_id", pa.string()), ("run_purpose", pa.string()),
              ("cell_id", pa.string()), ("model_id", pa.string()), ("engine", pa.string()),
              ("device", pa.string()), ("batch_rows", pa.int64()), ("cpu_threads", pa.int32()),
              ("nvforest_chunk", pa.int32()), ("pyre_gpu_row_group", pa.int32()),
              ("repetition", pa.int32())]
    if kind == "measurements":
        fields = common + [("sample", pa.int32()), ("loops", pa.int64()),
            ("elapsed_ns", pa.int64()), ("api_layer", pa.string()), ("phase", pa.string()),
            ("output_dtype", pa.string()), ("route", pa.string()),
            ("effective_row_group", pa.int32()), ("max_abs_error", pa.float64()),
            ("max_scaled_error", pa.float64()), ("model_sha256", pa.string()),
            ("input_sha256", pa.string()), ("oracle_sha256", pa.string()),
            ("engine_version", pa.string()), ("engine_settings_json", pa.string()), ("gate_tolerance", pa.float64()),
            ("requested_tree_count", pa.int32()),
            ("requested_depth_limit", pa.int32()), ("tree_count", pa.int32()),
            ("max_depth", pa.int32()), ("node_count", pa.int64()),
            ("tree_type", pa.string()), ("framework", pa.string()),
            ("family", pa.string()), ("model_dtype", pa.string()), ("features", pa.int32())]
    elif kind == "attempts":
        fields = common + [("status", pa.string()), ("reason", pa.string()),
                           ("request_path", pa.string()), ("stdout_path", pa.string()),
                           ("stderr_path", pa.string())]
    elif kind == "models":
        fields = [("schema_version", pa.int32()), ("model_id", pa.string()), ("model_name", pa.string()),
            ("framework", pa.string()), ("family", pa.string()), ("model_dtype", pa.string()),
            ("features", pa.int32()), ("requested_tree_count", pa.int32()), ("requested_depth_limit", pa.int32()),
            ("tree_count", pa.int32()), ("max_depth", pa.int32()), ("node_count", pa.int64()),
            ("tree_type", pa.string()), ("symmetric_tree_count", pa.int32()),
            ("nodes_per_tree", pa.list_(pa.int32())), ("depth_per_tree", pa.list_(pa.int32())),
            ("min_leaf_depth_per_tree", pa.list_(pa.int32())), ("mean_leaf_depth_per_tree", pa.list_(pa.float64())),
            ("symmetric_per_tree", pa.list_(pa.bool_())), ("training_parameters_json", pa.string()),
            ("model_sha256", pa.string()), ("native_sha256", pa.string()), ("training_sha256", pa.string()),
            ("conversion_probe_rows", pa.int32()), ("conversion_max_abs_error", pa.float64())]
    else:
        raise ValueError(f"Unknown table {kind}")
    schema = pa.schema(fields, metadata={b"pyre.comparison.schema_version": b"1",
                                        b"pyre.comparison.table": kind.encode()})
    unknown = set().union(*(set(row) for row in rows)) - set(schema.names)
    if unknown:
        raise ValueError(f"Unknown output fields: {unknown}")
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary = Path(str(path) + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def inspect_structure(meta):
    """Describe actual topology, independently of trainer name and node order."""
    trees = []
    for tree in meta["trees"]:
        nodes = {n["node_id"]: n for n in tree["nodes"]}
        if len(nodes) != len(tree["nodes"]):
            raise ValueError("Duplicate node IDs")
        seen, levels, leaves = set(), {}, []
        stack = [(0, 0)]
        symmetric = True
        while stack:
            index, depth = stack.pop()
            if index in seen:
                raise ValueError("Repeated node in tree")
            seen.add(index)
            node = nodes[index]
            if "leaf_value" in node:
                if not isinstance(node["leaf_value"], (int, float)):
                    raise ValueError("Only scalar leaves are supported")
                leaves.append(depth)
            else:
                if node.get("node_type") != "numerical_test_node":
                    raise ValueError("Only numerical splits are supported")
                rule = (node["split_feature_id"], node["threshold"], node["comparison_op"], node["default_left"])
                if depth in levels and levels[depth] != rule:
                    symmetric = False
                levels[depth] = rule
                stack += [(node["left_child"], depth + 1), (node["right_child"], depth + 1)]
        if len(seen) != len(nodes) or not leaves:
            raise ValueError("Unreachable nodes or no leaves")
        symmetric &= min(leaves) == max(leaves)
        trees.append({"nodes": len(nodes), "max_depth": max(leaves),
                      "min_leaf_depth": min(leaves), "leaves": len(leaves),
                      "symmetric": symmetric, "leaf_depths": leaves})
    if not trees:
        raise ValueError("Empty forest")
    symmetrical = sum(t["symmetric"] for t in trees)
    return {"tree_count": len(trees), "node_count": sum(t["nodes"] for t in trees),
            "max_depth": max(t["max_depth"] for t in trees),
            "tree_type": "symmetric" if symmetrical == len(trees) else "asymmetric" if not symmetrical else "mixed",
            "symmetric_tree_count": symmetrical, "trees": trees}


def package_version(name):
    import importlib.metadata
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if name == "xgboost-cpu":
            return importlib.metadata.version("xgboost")
        raise
