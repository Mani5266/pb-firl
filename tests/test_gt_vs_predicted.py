"""Positive-control test: GT shift-3.0 must score AUROC > 0.9.
Catches the per-group standardisation bug that forces AUROC to 0.5."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

SEED = 42
NREF, NEVL, NDIM = 20, 20, 8


def md(loc, inv, Z):
    D = Z - loc
    return np.sqrt((D @ inv * D).sum(1))


def run_battery(Z, rng):
    ix = np.arange(len(Z))
    rng.shuffle(ix)
    ref, evl = ix[:NREF], ix[NREF:NREF + NEVL]
    mu, sd = Z[ref].mean(0), Z[ref].std(0) + 1e-9
    Zs = (Z - mu) / sd
    Zr, Ze = Zs[ref], Zs[evl]
    dim = rng.choice(Z.shape[1], min(NDIM, Z.shape[1]), replace=False)
    Zi = Ze.copy()
    Zi[:, dim] += 3.0
    lab = np.r_[np.zeros(len(Ze)), np.ones(len(Zi))]
    lw_s = LedoitWolf().fit(Zr)
    inv_s = lw_s.precision_
    d, di = md(lw_s.location_, inv_s, Ze), md(lw_s.location_, inv_s, Zi)
    z_mu, z_sd = d.mean(), d.std() + 1e-9
    return roc_auc_score(lab, np.r_[(d - z_mu) / z_sd, (di - z_mu) / z_sd])


def test_gt_shift3_auroc_above_09():
    rng = np.random.RandomState(SEED)
    Z = rng.randn(60, 12)
    auc = run_battery(Z, np.random.RandomState(SEED))
    assert auc > 0.9, f'GT shift-3.0 AUROC={auc:.3f} not > 0.9 — scoring bug?'


def test_shared_scale_preserves_signal():
    rng = np.random.RandomState(SEED)
    Z = rng.randn(60, 12)
    ref, evl = Z[:20], Z[20:40]
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    Zs = (Z - mu) / sd
    Zr, Ze = Zs[:20], Zs[20:40]
    Zi = Ze.copy()
    Zi[:, :4] += 3.0
    lab = np.r_[np.zeros(20), np.ones(20)]
    lw = LedoitWolf().fit(Zr)
    inv = lw.precision_
    d, di = md(lw.location_, inv, Ze), md(lw.location_, inv, Zi)
    z_mu, z_sd = d.mean(), d.std() + 1e-9
    auc = roc_auc_score(lab, np.r_[(d - z_mu) / z_sd, (di - z_mu) / z_sd])
    assert auc > 0.9, f'shared-scale AUROC={auc:.3f} not > 0.9'


def test_per_group_standardisation_kills_signal():
    """Demonstrates the bug: per-group standardisation forces AUROC ~0.5."""
    rng = np.random.RandomState(SEED)
    Z = rng.randn(60, 12)
    ref, evl = Z[:20], Z[20:40]
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    Zs = (Z - mu) / sd
    Zr, Ze = Zs[:20], Zs[20:40]
    Zi = Ze.copy()
    Zi[:, :4] += 3.0
    lab = np.r_[np.zeros(20), np.ones(20)]
    lw = LedoitWolf().fit(Zr)
    inv = lw.precision_
    d, di = md(lw.location_, inv, Ze), md(lw.location_, inv, Zi)
    auc = roc_auc_score(lab, np.r_[(d - d.mean()) / (d.std() + 1e-9),
                                    (di - di.mean()) / (di.std() + 1e-9)])
    assert auc < 0.6, f'per-group standardisation should kill signal, got {auc:.3f}'
