"""Checks that malformed timing output cannot silently become benchmark results."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('benchmark_gpu', Path(__file__).resolve().parents[1]/'tools/benchmark_gpu.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class WorkerRecords(unittest.TestCase):
    def test_uses_calibrated_loops(self):
        record, = harness.parse_worker('reuse 0.125 250 1 2 3 4 5 6 7 8', ('reuse',))
        self.assertEqual(record['seconds_per_call'], 0.0005)
        self.assertEqual(record['cold_ns']['decode'], 3)

    def test_rejects_invalid_or_incomplete_results(self):
        for raw in ('', 'reuse nan 2 1 2 3 4 5 6 7 8', 'reuse 0.1 0 1 2 3 4 5 6 7 8',
                    'reuse -1 2 1 2 3 4 5 6 7 8', 'device 1 2 1 2 3 4 5 6 7 8',
                    'reuse 1 2 1 2 3 4 5 6 7 8\nreuse 1 2 1 2 3 4 5 6 7 8'):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                harness.parse_worker(raw, ('reuse',))

    def test_single_process_does_not_claim_repeatability(self):
        result, = harness.summaries([dict(case='x', rows=33, phase='reuse', seconds_per_call=0.01)])
        self.assertIsNone(result['cv'])
        self.assertEqual(result['processes'], 1)


if __name__ == '__main__':
    unittest.main()
