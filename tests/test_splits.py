"""Split-integrity tests: grouped splits isolate leakage; naive splits do not.

Uses synthetic groups only (no data needed). If the grouped splitter ever lets one
source group appear on both sides, the pipeline's contamination guard has failed.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.pain.improve_mil import grouped_strat_split


class TestSplitIntegrity(unittest.TestCase):
    def _data(self):
        rng = np.random.RandomState(0)
        groups = np.repeat([f'src{i}' for i in range(40)], 5)  # 40 sources x 5 crops
        y = (rng.rand(200) < 0.3).astype(float)
        return y, groups

    def test_grouped_split_isolates_groups(self):
        y, groups = self._data()
        tr, va = grouped_strat_split(y, groups)
        self.assertEqual(set(groups[tr]) & set(groups[va]), set())
        self.assertEqual(len(tr) + len(va), len(y))

    def test_grouped_split_is_deterministic(self):
        y, groups = self._data()
        a = grouped_strat_split(y, groups)
        b = grouped_strat_split(y, groups)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_grouped_split_keeps_both_classes(self):
        y, groups = self._data()
        tr, va = grouped_strat_split(y, groups)
        self.assertEqual(set(np.unique(y[tr])), {0.0, 1.0})
        self.assertEqual(set(np.unique(y[va])), {0.0, 1.0})

    def test_naive_random_split_leaks_groups(self):
        # documents WHY grouping matters: random frame split strands groups across sides
        y, groups = self._data()
        rng = np.random.RandomState(42)
        va = rng.choice(len(y), 40, replace=False)
        leaked = len(set(groups[va]) & set(groups[np.setdiff1d(np.arange(len(y)), va)]))
        self.assertGreater(leaked, 0)


if __name__ == '__main__':
    unittest.main()
