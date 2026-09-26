"""Math/eval-behaviour unit tests (no torch, no data needed except manifests for none)."""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.front_end.geometry import umeyama, align, oks, oks_map, box_area, as_kpts
from src.cusum.cusum import cusum_hits


class TestGeometry(unittest.TestCase):
    def test_rotation_recovery(self):
        rng = np.random.RandomState(0)
        src = rng.randn(13, 2)
        th, s, t = np.deg2rad(30), 1.5, np.array([3.0, -2.0])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        dst = s * (src @ R.T) + t
        self.assertLess(np.abs(align(src, dst) - dst).max(), 1e-6)

    def test_box_area(self):
        self.assertEqual(box_area(np.array([10., 20., 30., 40.])), 400.0)
        self.assertNotEqual(box_area(np.array([10., 20., 30., 40.])), 30. * 40.)  # old x2*y2 bug

    def test_oks_perfect(self):
        g = np.random.RandomState(1).randn(13, 2) * 100
        self.assertAlmostEqual(oks(g, g, np.ones(13, int), 50000.0), 1.0)

    def test_oks_no_vis(self):
        g = np.zeros((13, 2))
        self.assertEqual(oks(g + 5, g, np.zeros(13, int), 100.0), 0.0)

    def test_success_rate(self):
        rec = [(0.9, 1.0), (0.6, 0.5), (0.2, 0.9)]
        m = oks_map(rec)
        self.assertAlmostEqual(m['success@0.50'], 2/3)
        self.assertAlmostEqual(m['success@0.75'], 1/3)
        self.assertNotIn('mAP', m)  # must not masquerade as COCO AP

    def test_as_kpts_ragged(self):
        k = [np.array([1., 2.]) for _ in range(13)]
        self.assertEqual(as_kpts(k).shape, (13, 2))


class TestCusum(unittest.TestCase):
    def test_silent_on_flat(self):
        self.assertEqual(cusum_hits(np.zeros(200), 5.0), [])

    def test_detects_shift(self):
        x = np.r_[np.zeros(100), np.full(100, 3.0)]
        hits = cusum_hits(x, 5.0)
        self.assertTrue(any(h >= 100 for h in hits))
        self.assertLess(min(h for h in hits if h >= 100) - 100, 20)


if __name__ == '__main__':
    unittest.main()
