"""Counter exports must identify exactly one Pyre prediction kernel."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from profile_gpu_counters import parse_metrics

HEADER = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
ROW = '"0","pyre__gpu_kernel_predict_kernel_float32","launch__registers_per_thread","register/thread","40"\n'


class CounterRecords(unittest.TestCase):
    def test_retains_units_and_identity_with_preamble(self):
        record, = parse_metrics('==PROF== Importing\n'+HEADER+ROW)
        self.assertEqual(record['Metric Unit'], 'register/thread')
        self.assertEqual(record['Metric Value'], '40')
        self.assertIn('predict_kernel', record['Kernel Name'])

    def test_wide_raw_export(self):
        raw = ('"ID","Kernel Name","Block Size","Grid Size","launch__registers_per_thread"\n'
               '"","","","","register/thread"\n'
               '"0","pyre__gpu_kernel_predict_kerne6A","(128, 1, 1)","(512, 1, 1)","40"\n')
        record, = parse_metrics(raw)
        self.assertEqual(record['Metric Value'], '40')
        self.assertEqual(record['Grid Size'], '(512, 1, 1)')
        with self.assertRaises(ValueError):
            parse_metrics(raw+raw.splitlines()[-1]+'\n')

    def test_rejects_missing_wrong_or_multiple_kernels(self):
        for raw in ('', HEADER, HEADER+ROW.replace('pyre__gpu_kernel_predict_kernel', 'other'),
                    HEADER+ROW+ROW.replace('"0"', '"1"'),
                    HEADER+ROW.replace('"40"', '""')):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_metrics(raw)


if __name__ == '__main__':
    unittest.main()
