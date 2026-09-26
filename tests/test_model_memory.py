"""Accounting checks against pinned native models and their field schema.

Build model-memory-worker first; run with python -m unittest discover -s tests
-p test_model_memory.py.
"""

import re
import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from model_memory import parse_worker


class ModelMemoryTests(unittest.TestCase):
    def worker(self, fixture):
        return subprocess.run(
            [str(ROOT / "build/model-memory-worker"),
             str(ROOT / "build/balsa-reference/tests/fixtures" / fixture)],
            text=True, capture_output=True,
        )

    def test_leaf_payload_both_precisions(self):
        for dtype, node_bytes in (("float32", 23), ("float64", 31)):
            with self.subTest(dtype=dtype):
                result = self.worker(f"{dtype}_leaf.tl")
                self.assertEqual(result.returncode, 0, result.stderr)
                report = parse_worker(result.stdout)
                self.assertEqual(report["nodes"], 1)
                self.assertEqual(report["dtype"], dtype)
                self.assertEqual(report["groups"]["inference"]["logical_bytes"], node_bytes + 8)
                self.assertEqual(report["groups"]["unsupported_semantics"]["logical_bytes"], 33)
                # A packed constant tree retains one typed leaf, two split and
                # leaf offsets, and two border offsets for its one feature.
                self.assertEqual(report["groups"]["packed_cpu"]["logical_bytes"],
                                 (4 if dtype == "float32" else 8) + 48)

    def test_all_owned_list_fields_are_visited(self):
        # Fail on upstream schema additions rather than silently undercounting.
        schema = (ROOT / "build/balsa-reference/src/balsa/model.mojo").read_text()
        worker = (ROOT / "tools/model_memory_worker.mojo").read_text()
        for name, prefix in (("Model", "model"), ("Tree", "tree")):
            body = schema.split(f"struct {name}[")[1].split("    def __init__")[0]
            fields = re.findall(r"var (\w+): List\[", body)
            for field in fields:
                self.assertIn(f'"{prefix}.{field}"', worker)
        extension = schema.split("struct Extension(")[1].split("struct Tree[")[0]
        self.assertEqual(re.findall(r"var (\w+): List\[", extension), ["name", "payload"])

    def test_unsupported_model_fails(self):
        result = self.worker("frameworks/lightgbm_categorical.tl")
        self.assertNotEqual(result.returncode, 0)
        with self.assertRaises(ValueError):
            parse_worker(result.stdout)

    def test_truncation_fails(self):
        output = self.worker("float32_leaf.tl").stdout
        with self.assertRaises(ValueError):
            parse_worker("\n".join(output.splitlines()[:-1]))

    def test_capacity_cannot_be_smaller_than_payload(self):
        with self.assertRaises(ValueError):
            parse_worker("tree.cleft\tinference\t8\t4\n")


if __name__ == "__main__":
    unittest.main()
