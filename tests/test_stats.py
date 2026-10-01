"""Tests for imbalance-aware and cluster-aware evaluation helpers."""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.eval.stats import (classification_summary, cluster_bootstrap_auc,
                            paired_cluster_mean_ci, binomial_miss_upper_bound)
from src.cusum.cusum import calibrate_threshold, cusum_statistic, inject_feature_shift


class TestStats(unittest.TestCase):
    def test_classification_reports_majority_separately(self):
        y = np.array([0] * 8 + [1] * 2)
        pred = np.zeros_like(y)
        out = classification_summary(y, pred)
        self.assertEqual(out['accuracy'], 0.8)
        self.assertEqual(out['majority_accuracy'], 0.8)
        self.assertEqual(out['uniform_chance'], 0.5)
        self.assertLess(out['balanced_accuracy'], out['accuracy'])

    def test_cluster_auc_ci_and_pair_ci(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        c = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'])
        ci = cluster_bootstrap_auc(y, s, c, n_resamples=100, seed=0)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])
        dci = paired_cluster_mean_ci(s, s - 0.1, c, n_resamples=100, seed=0)
        self.assertLessEqual(dci[0], dci[1])
        self.assertTrue(dci[0] > 0)

    def test_zero_miss_bound_is_not_zero(self):
        self.assertGreater(binomial_miss_upper_bound(0, 5), 0.4)
        self.assertEqual(binomial_miss_upper_bound(5, 5), 1.0)


class TestCusumProtocol(unittest.TestCase):
    def test_feature_injection_precedes_rescoring(self):
        x = np.zeros((10, 4))
        y = inject_feature_shift(x, 5, [1, 3], 2.0)
        self.assertTrue(np.all(y[:5] == 0))
        self.assertTrue(np.all(y[5:, [1, 3]] == 2.0))
        self.assertEqual(float(np.max(cusum_statistic(y[:, 1]))), 7.5)

    def test_threshold_is_calibrated_from_cusum_statistic(self):
        streams = [np.zeros(20), np.r_[np.zeros(19), 1.0]]
        tau, met, summary, maxima = calibrate_threshold(streams, 0.01)
        self.assertTrue(met)
        self.assertEqual(summary['stream_far'], 0.0)
        self.assertEqual(len(maxima), 2)
        self.assertGreaterEqual(tau, max(maxima))


if __name__ == '__main__':
    unittest.main()
