"""Data-free synthetic smoke test: exercises the baseline + CUSUM machinery end to end
on random animals. This tests plumbing (fit -> score -> detect -> alarm), NOT science:
synthetic Gaussians are guaranteed to favor the per-cow model by construction."""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.baseline.gaussian import PerCowBaseline
from src.cusum.cusum import cusum_hits, cusum_statistic
from src.front_end.geometry import roi_features


def fake_animals(n_cows=6, n_frames=40, seed=0):
    rng = np.random.RandomState(seed)
    rows, K = [], []
    for c in range(n_cows):
        center = rng.randn(13, 2) * 50 + 500
        for f in range(n_frames):
            K.append(center + rng.randn(13, 2) * 8)
            rows.append({'cow': f'cow{c}', 'area': 40000.0})
    df = pd.DataFrame(rows)
    return df, np.array(K)


class TestSmoke(unittest.TestCase):
    def test_baseline_separates_injected_shift(self):
        from sklearn.metrics import roc_auc_score
        df, K = fake_animals()
        b = PerCowBaseline().fit(df, K)
        X = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])
        Xinj = X.copy()
        Xinj[:, [3, 4]] += 3.0 * b.sd_g[[3, 4]]
        s_h, _ = b.deviation(X, df.cow.values)
        s_i, _ = b.deviation(Xinj, df.cow.values)
        y = np.r_[np.zeros(len(s_h)), np.ones(len(s_i))]
        self.assertGreater(roc_auc_score(y, np.r_[s_h, s_i]), 0.7)

    def test_coldstart_path_is_calibrated(self):
        df, K = fake_animals()
        b = PerCowBaseline().fit(df, K)
        X = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])
        s, conf = b.deviation(X[:5], ['new_cow'] * 5)
        self.assertTrue(np.all(conf == 0))
        self.assertTrue(np.isfinite(s).all())
        # same scale as known-cow scores (both z-scored)
        s_known, _ = b.deviation(X[:5], df.cow.values[:5])
        self.assertLess(abs(s.mean() - s_known.mean()), 5.0)

    def test_cusum_fires_on_sustained_shift(self):
        x = np.r_[np.zeros(100), np.full(100, 3.0)]
        hits = cusum_hits(x, 5.0)
        self.assertTrue(any(h >= 100 for h in hits))
        self.assertEqual(cusum_hits(np.zeros(200), 5.0), [])
        self.assertEqual(len(cusum_statistic(x)), 200)


if __name__ == '__main__':
    unittest.main()
