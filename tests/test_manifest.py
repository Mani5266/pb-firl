"""Loader count checks + split integrity (no identity leakage across splits)."""
import os
import unittest
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'runs', 'features_cache')


class TestManifest(unittest.TestCase):
    def test_counts(self):
        rgb = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
        thm = pd.read_parquet(os.path.join(CACHE, 'manifest_thermal.parquet'))
        self.assertEqual(len(rgb), 1890)
        self.assertEqual(len(thm), 2611)
        self.assertTrue(bool(rgb.exists.all()) and bool(thm.exists.all()))

    def test_no_identity_leak(self):
        rgb = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
        got = {s: set(rgb[rgb.split == s].cow) for s in ['train', 'val', 'test']}
        self.assertEqual(got['train'] & got['val'], set())
        self.assertEqual(got['train'] & got['test'], set())
        self.assertEqual(got['val'] & got['test'], set())

    def test_keypoints_shape(self):
        import numpy as np
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from src.front_end.geometry import as_kpts
        rgb = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
        k = as_kpts(rgb.kpts.iloc[0])
        self.assertEqual(k.shape, (13, 2))
        self.assertTrue(np.isfinite(k).all())


if __name__ == '__main__':
    unittest.main()
