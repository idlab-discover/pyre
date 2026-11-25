"""Contracts that prevent silent comparison-axis or evidence corruption."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import array
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from comparison.schema import inspect_structure, parquet, read_matrix, validate_matrix
from comparison.run import cells, parse_native, validate_samples, verify_corpus


def split(index, left, right, feature=0):
    return dict(node_id=index, node_type="numerical_test_node", split_feature_id=feature,
                threshold=0.5, comparison_op="<=", default_left=True, left_child=left, right_child=right)


def leaf(index):
    return dict(node_id=index, leaf_value=1.0)


class ComparisonContracts(unittest.TestCase):
    def test_topology_is_independent_of_numbering_and_node_order(self):
        nodes = [leaf(13), split(9, 12, 13, 1), split(0, 3, 9), leaf(5),
                 split(3, 5, 6, 1), leaf(12), leaf(6)]
        result = inspect_structure({"trees": [{"nodes": nodes}]})
        self.assertEqual((result["tree_type"], result["max_depth"], result["node_count"]), ("symmetric", 2, 7))
        nodes[1]["split_feature_id"] = 2
        self.assertEqual(inspect_structure({"trees": [{"nodes": nodes}]})["tree_type"], "asymmetric")

    def test_uneven_depth_is_not_symmetric(self):
        nodes = [split(0, 1, 2), leaf(1), split(2, 3, 4), leaf(3), leaf(4)]
        self.assertEqual(inspect_structure({"trees": [{"nodes": nodes}]})["tree_type"], "asymmetric")

    def test_missing_direction_is_part_of_symmetry(self):
        nodes = [split(0, 1, 2), split(1, 3, 4), split(2, 5, 6), *map(leaf, range(3, 7))]
        nodes[2]["default_left"] = False
        self.assertEqual(inspect_structure({"trees": [{"nodes": nodes}]})["tree_type"], "asymmetric")

    def test_bad_topology_is_rejected(self):
        for nodes in [[split(0, 0, 1), leaf(1)], [leaf(0), leaf(9)], [leaf(0), leaf(0)]]:
            with self.assertRaises(ValueError):
                inspect_structure({"trees": [{"nodes": nodes}]})

    def test_matrix_rejects_silent_typos_and_invalid_axes(self):
        matrix = read_matrix(ROOT / "benchmarks/cpu-smoke.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.json"
            for field, value in [("devices", ["tpu"]), ("samples", 0), ("batch_rows", [1, 1]),
                                 ("pyre_gpu_row_groups", [3]), ("nvforest_chunks", []), ("typo", True)]:
                path.write_text(json.dumps(matrix | {field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    read_matrix(path)

    def test_device_controls_do_not_form_meaningless_cross_products(self):
        matrix = read_matrix(ROOT / "benchmarks/cpu-smoke.json")
        matrix.update(devices=["cpu", "gpu"], cpu_threads=[1, 8], nvforest_chunks=[None, 8], pyre_gpu_row_groups=[None, 4])
        manifest = {"models": [{"model_id": "a", "inputs": [{"batch_rows": 33}]}]}
        plan = [cell for cell, _, _ in cells(manifest, matrix)]
        self.assertEqual(len(plan), 13)  # CPU 2*(1+2+1), GPU 2+2+1.
        self.assertEqual(len({c["cell_id"] for c in plan}), len(plan))
        for cell in plan:
            if cell["device"] == "gpu":
                self.assertIsNone(cell["cpu_threads"])
            if cell["engine"] != "nvforest":
                self.assertIsNone(cell["nvforest_chunk"])

    def test_gpu_presets_and_native_support(self):
        smoke = read_matrix(ROOT / "benchmarks/gpu-smoke.json")
        heavy = read_matrix(ROOT / "benchmarks/gpu-heavy.json")
        self.assertEqual(smoke["devices"], ["gpu"])
        self.assertGreater(max(heavy["batch_rows"]), 262144)
        self.assertGreater(max(heavy["tree_counts"]), 1024)
        self.assertLessEqual(max(heavy["batch_rows"]) * heavy["features"] * 8, heavy["max_input_bytes"])
        with self.assertRaisesRegex(ValueError, "XGBoost only"):
            validate_matrix(smoke | {"engines": ["pyre", "native"]})
        xgb = smoke | {"engines": ["pyre", "native"], "families": ["xgboost_depthwise"]}
        validate_matrix(xgb)
        with self.assertRaisesRegex(ValueError, "prepared"):
            validate_matrix(xgb | {"native_input": "ndarray"})

    def test_gpu_protocol_preserves_launch_plan(self):
        with self.assertRaisesRegex(ValueError, "launch plan"):
            parse_native("gate 0 0\nmeta -1 8\nsample 0 2 1000\n", 1)
        row = parse_native("gate 0 0\nmeta -1 8\nplan 3 128 2 1 0\nsample 0 2 1000\n", 1)[0]
        self.assertEqual(row["route"], "gpu_packed")
        self.assertEqual(row["effective_row_group"], 8)
        self.assertEqual(row["gpu_plan"], dict(chunk_shift=3, threads=128,
                                             rows_per_thread=2, trees_per_thread=1, staged=0))

    def test_native_protocol_requires_gate_and_complete_intervals(self):
        text = "gate 0.000001 0.0000001\nmeta 2 0\nsample 0 128 10000000\n"
        row = parse_native(text, 1)[0]
        self.assertEqual(row["route"], "symmetric_bins")
        for bad in [text.replace("gate 0.000001 0.0000001\n", ""),
                    text.replace("128", "0"), text.replace("0.000001", "nan"), text + "junk\n"]:
            with self.assertRaises(ValueError):
                parse_native(bad, 1)
        with self.assertRaises(ValueError):
            parse_native(text, 2)

    def test_duplicate_samples_rejected(self):
        row = dict(sample=0, loops=1, elapsed_ns=1, max_abs_error=0., max_scaled_error=0.)
        with self.assertRaises(ValueError):
            validate_samples([row, row], 2)

    def test_tampered_corpus_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "Changed model"):
                verify_corpus(root, {"models": [{"model_id": "a", "model_path": "model", "model_sha256": "wrong"}]})

    @unittest.skipUnless((ROOT / "build/comparison-worker-current").exists(), "build comparison worker first")
    def test_native_worker_gates_before_timing(self):
        self.check_worker_gate("comparison-worker-current", "cpu")

    @unittest.skipUnless((ROOT / "build/comparison-gpu-worker-current").exists(), "build GPU comparison worker first")
    def test_gpu_worker_gates_before_timing(self):
        self.check_worker_gate("comparison-gpu-worker-current", "gpu")

    @unittest.skipUnless((ROOT / "build/comparison-worker-current").exists(), "build comparison worker first")
    def test_inputs_can_exceed_checkpoint_byte_limit(self):
        rows = 16 * 1024 * 1024 + 1  # Just over 64 MiB of float32 input.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with (tmp / "x.bin").open("wb") as stream:
                stream.truncate(rows * 4)
            with (tmp / "oracle.bin").open("wb") as stream:
                block = array.array("d", [-1.] * (1024 * 1024)).tobytes()
                for _ in range(16):
                    stream.write(block)
                stream.write(array.array("d", [-1.]).tobytes())
            result = subprocess.run([str(ROOT / "build/comparison-worker-current"), "cpu",
                str(ROOT / "build/balsa-reference/tests/fixtures/float32_op2_missing0.tl"),
                "float32", str(tmp / "x.bin"), str(rows), "1", str(tmp / "oracle.bin"),
                "1", "1", "0.001", "0.00002", "0"], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(parse_native(result.stdout, 1)), 1)

    def check_worker_gate(self, binary, device):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "x.bin").write_bytes(array.array("f", [0., .5, 1.]).tobytes())
            reference = tmp / "oracle.bin"
            command = [str(ROOT / "build" / binary), device,
                str(ROOT / "build/balsa-reference/tests/fixtures/float32_op2_missing0.tl"),
                "float32", str(tmp / "x.bin"), "3", "1", str(reference), "1", "1", "0.001", "0.00002", "0"]
            reference.write_bytes(array.array("d", [-1., 3., 3.]).tobytes())
            valid = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(len(parse_native(valid.stdout, 1)), 1)
            reference.write_bytes(array.array("d", [99., 3., 3.]).tobytes())
            invalid = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertNotIn("sample ", invalid.stdout)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "comparison environment required")
    def test_parquet_empty_schema_and_nullable_controls(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measurements.parquet"
            parquet(path, [], "measurements")
            schema = pq.read_schema(path)
            self.assertEqual(str(schema.field("elapsed_ns").type), "int64")
            parquet(path, [{"schema_version": 1, "elapsed_ns": 100, "loops": 2, "nvforest_chunk": None}], "measurements")
            self.assertEqual(pq.read_table(path).to_pylist()[0]["nvforest_chunk"], None)
            with self.assertRaises(ValueError):
                parquet(path, [{"accidental_rank": 1}], "measurements")


class ComparisonCLI(unittest.TestCase):
    def test_selected_libraries_and_axes_are_effective(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/compare.py"), "plan",
            "--libraries", "xgboost", "nvforest", "--trees", "256", "--depths", "10",
            "--rows", "65536", "--threads", "16"], capture_output=True, text=True, check=True)
        matrix = json.loads(result.stdout)
        self.assertEqual(matrix["families"], ["xgboost_depthwise", "xgboost_lossguide"])
        self.assertEqual(set(matrix["engines"]), {"pyre", "native", "nvforest"})
        self.assertEqual(matrix["tree_counts"], [256])
        self.assertEqual(matrix["cpu_threads"], [16])

    def test_invalid_override_is_rejected(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/compare.py"), "plan",
            "--threads", "0"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integers", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "comparison environment required")
    def test_report_rejects_changed_sources(self):
        from comparison.report import summarize
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "changed-sources.json").write_text('["src/pyre/forest.mojo"]')
            with self.assertRaisesRegex(ValueError, "Sources changed"):
                summarize(tmp)


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"), "comparison environment required")
class ComparisonReport(unittest.TestCase):
    def test_gpu_and_cpu_pairs_remain_separate_with_multiple_gpu_settings(self):
        from comparison.report import summarize
        matrix = read_matrix(ROOT / "benchmarks/gpu-smoke.json")
        matrix.update(devices=["cpu", "gpu"], batch_rows=[33], processes=1, samples=1)
        manifest = {"models": [{"model_id": "m", "inputs": [{"batch_rows": 33}]}], "matrix": matrix}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "provenance.json").write_text(json.dumps({"matrix": matrix}))
            (path / "corpus-manifest.json").write_text(json.dumps(manifest))
            attempts, measurements = [], []
            for cell, _, _ in cells(manifest, matrix):
                attempts.append(cell | {"repetition": 0, "status": "ok"})
                elapsed = 100 if cell["engine"] == "pyre" else 200 if cell["device"] == "cpu" else 400
                measurements.append(cell | dict(repetition=0, sample=0, loops=1, elapsed_ns=elapsed,
                    max_scaled_error=0., gate_tolerance=1e-12, family="xgboost_depthwise",
                    model_sha256="m", input_sha256="i", oracle_sha256="o", phase="resident_allocate_completed"))
            parquet(path / "attempts.parquet", attempts, "attempts")
            parquet(path / "measurements.parquet", measurements, "measurements")
            _, _, _, pairs = summarize(path)
            self.assertEqual(list(pairs[pairs.device.eq("cpu")].speedup), [2., 2.])
            self.assertEqual(list(pairs[pairs.device.eq("gpu")].speedup), [4.] * 4)
            self.assertEqual(set(pairs.configuration_pyre), {"CPU 1 threads", "GPU automatic", "GPU 8 rows/block"})
            attempts[0]["status"] = "failed"
            parquet(path / "attempts.parquet", attempts, "attempts")
            with self.assertRaisesRegex(ValueError, "unsuccessful"):
                summarize(path)

    def test_matched_ratio_and_missing_attempt_rejection(self):
        from comparison.report import summarize
        matrix = read_matrix(ROOT / "benchmarks/cpu-smoke.json")
        matrix.update(engines=["pyre", "native"], batch_rows=[33], processes=2, samples=1)
        model = {"model_id": "model-a", "inputs": [{"batch_rows": 33}]}
        manifest = {"models": [model], "matrix": matrix}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "provenance.json").write_text(json.dumps({"matrix": matrix}))
            (path / "corpus-manifest.json").write_text(json.dumps(manifest))
            attempts, measurements = [], []
            for cell, _, _ in cells(manifest, matrix):
                for rep in range(2):
                    attempts.append(cell | {"repetition": rep, "status": "ok"})
                    measurements.append(cell | {"repetition": rep, "sample": 0, "loops": 2,
                        "elapsed_ns": 200 if cell["engine"] == "pyre" else 400,
                        "max_scaled_error": 0., "gate_tolerance": 1e-12,
                        "family": "xgboost_depthwise", "model_sha256": "m",
                        "input_sha256": "i", "oracle_sha256": "o", "phase": "resident_allocate_completed"})
            parquet(path / "attempts.parquet", attempts, "attempts")
            parquet(path / "measurements.parquet", measurements, "measurements")
            _, _, stats, pairs = summarize(path)
            self.assertEqual(list(pairs.speedup), [2.0])
            self.assertEqual(list(stats["count"]), [2, 2])
            parquet(path / "attempts.parquet", attempts[:-1], "attempts")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                summarize(path)


if __name__ == "__main__":
    unittest.main()
