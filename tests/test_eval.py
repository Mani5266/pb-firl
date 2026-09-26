"""Math/eval behaviour tests: procrustes recovery, OKS, box area, CUSUM, MIL shapes."""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.front_end.geometry import umeyama, align, oks, oks_map, box_area, canonical_template
from src.cusum.cusum import cusum_hits


class TestGeometry(unittest.TestCase):
    def test_umeyama_recovers_known_transform(self):
        rng = np.random.RandomState(0)
        src = rng.rand(13, 2) * 100
        th = 0.7
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        dst = 2.5 * (src @ R.T) + np.array([30.0, -10.0])
        s, Rr, t = umeyama(src, dst)
        self.assertAlmostEqual(s, 2.5, places=6)
        self.assertTrue(np.allclose(Rr, R, atol=1e-6))
        self.assertTrue(np.allclose(t, [30.0, -10.0], atol=1e-6))
        self.assertTrue(np.allclose(align(src, dst), dst, atol=1e-6))

    def test_oks_perfect_and_formula(self):
        g = np.array([[0.0, 0.0], [10.0, 0.0]])
        self.assertAlmostEqual(oks(g, g, np.array([1, 1]), 100.0), 1.0)
        p = g + np.array([[1.0, 0.0], [0.0, 0.0]])  # d=1 on first kpt
        expect = (np.exp(-1.0 / (2 * 100.0 * 0.01)) + 1.0) / 2
        self.assertAlmostEqual(oks(p, g, np.array([1, 1]), 100.0), expect)
        self.assertEqual(oks(p, g, np.array([0, 0]), 100.0), 0.0)

    def test_box_area_not_origin_product(self):
        self.assertEqual(box_area([10.0, 20.0, 30.0, 50.0]), 20.0 * 30.0)
        self.assertNotEqual(box_area([10.0, 20.0, 30.0, 50.0]), 30.0 * 50.0)

    def test_oks_map_is_success_rate(self):
        rec = [(0.9, 0.9), (0.4, 0.1), (0.0, 0.0)]
        m = oks_map(rec)
        self.assertAlmostEqual(m['success@0.50'], 1 / 3)
        self.assertAlmostEqual(m['success@0.75'], 1 / 3)
        self.assertAlmostEqual(m['mean_success'], 8/30)  # 0.9 clears t in 0.5..0.85


class TestCusum(unittest.TestCase):
    def test_no_alarm_on_zeros_high_tau(self):
        self.assertEqual(cusum_hits(np.zeros(500), 6.0), [])

    def test_alarm_on_sustained_shift(self):
        x = np.r_[np.zeros(100), np.full(100, 2.0)]
        hits = cusum_hits(x, 1.0)
        self.assertTrue(any(h >= 100 for h in hits))
        self.assertTrue(all(h >= 100 for h in hits))


class TestMIL(unittest.TestCase):
    def test_shapes_and_attention_sums(self):
        import torch
        from src.pain.train_sheep import AttMIL
        m = AttMIL(d=32).eval()
        h = torch.randn(4, 25, 2048)
        with torch.no_grad():
            logit, attn = m(h)
        self.assertEqual(tuple(logit.shape), (4,))
        self.assertEqual(tuple(attn.shape), (4, 25))
        self.assertTrue(torch.allclose(attn.sum(1), torch.ones(4), atol=1e-5))

    def test_focal_finite_and_hard_weighted(self):
        from src.pain.improve_mil import FocalLoss
        import torch
        lf = FocalLoss(alpha=0.25, gamma=2.0)
        hard = lf(torch.tensor([0.0]), torch.tensor([1.0]))
        easy = lf(torch.tensor([4.0]), torch.tensor([1.0]))
        self.assertTrue(torch.isfinite(hard) and torch.isfinite(easy))
        self.assertGreater(float(hard), float(easy))


if __name__ == '__main__':
    unittest.main()
