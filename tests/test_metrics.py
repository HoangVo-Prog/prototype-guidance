import os
import sys
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if torch is not None:
    from utils.metrics import Evaluator


@unittest.skipUnless(torch is not None, 'Torch is required for metrics tests.')
class MetricsTests(unittest.TestCase):
    def test_host_only_selection_row_detection_excludes_host_t2i_variants(self):
        self.assertTrue(Evaluator._is_host_only_selection_row('host-t2i'))
        self.assertTrue(Evaluator._is_host_only_selection_row('host_only-t2i'))
        self.assertTrue(Evaluator._is_host_only_selection_row('host-t2i out'))
        self.assertTrue(Evaluator._is_host_only_selection_row('host_t2i_out'))
        self.assertTrue(Evaluator._is_host_only_selection_row('host(1.00)+prototype(0.00)-t2i'))

    def test_host_only_selection_row_detection_keeps_pas_and_nonzero_fusion(self):
        self.assertFalse(Evaluator._is_host_only_selection_row('pas-t2i'))
        self.assertFalse(Evaluator._is_host_only_selection_row('prototype-t2i'))
        self.assertFalse(Evaluator._is_host_only_selection_row('host+prototype(0.25)-t2i'))

    def test_pas_selection_row_detection(self):
        self.assertTrue(Evaluator._is_pas_selection_row('pas-t2i'))
        self.assertTrue(Evaluator._is_pas_selection_row('pas_t2i'))
        self.assertFalse(Evaluator._is_pas_selection_row('host-t2i'))
        self.assertFalse(Evaluator._is_pas_selection_row('prototype-t2i'))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
