"""Permuted-identity null for the own-vs-pop gap (folder-map sensitivity).
Same pipeline as falsify D_full/A_pop, but cow labels shuffled: if the gap depends on
the (unverified) folder=identity mapping, it should vanish under permutation.
50 shuffles, shift 1.5, frozen dims/budgets."""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42
NREF, NEVL, NDIM, SHIFT, NSHUF = 20, 20, 8, 1.5, 50


def md(loc, inv, Z):
    D = Z - loc
    return np.sqrt((D @ inv * D).sum(1))


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_recow.parquet'))
    E = np.load(os.path.join(CACHE, 'dairy_emb.npz'))['E']
    y0 = df.cow.values
    counts = {c: (y0 == c).sum() for c in np.unique(y0)}
    ordered = sorted(counts, key=counts.get, reverse=True)
    pool_cows = ordered[:30]
    pca = PCA(n_components=64, random_state=SEED).fit(E[np.isin(y0, pool_cows)])
    P = pca.transform(E)
    eval_cows = [c for c in ordered[30:] if counts[c] >= NREF + NEVL][:60]
    dim = rng.choice(64, NDIM, replace=False)
    null_gaps = []
    for s in range(NSHUF):
        perm = rng.permutation(len(y0))
        y = y0[perm]  # shuffle labels, keep features fixed
        gaps = []
        for c in np.unique(y):
            if c in pool_cows:
                continue
            ix = np.where(y == c)[0]
            if len(ix) < NREF + NEVL or c not in eval_cows:
                continue
            take = ix[:NREF + NEVL] if len(ix) >= NREF + NEVL else ix
            if len(take) < NREF + 5:
                continue
            ref, evl = take[:NREF], take[NREF:NREF + NEVL]
            others = np.setdiff1d(np.where(~np.isin(y, [c] + pool_cows))[0], take)
            if len(others) < 100:
                continue
            mu_g = P[others].mean(0)
            sd_g = P[others].std(0) + 1e-9
            Z0 = (P[others] - mu_g) / sd_g
            Zr, Ze = (P[ref] - mu_g) / sd_g, (P[evl] - mu_g) / sd_g
            Zi = Ze.copy()
            Zi[:, dim] += SHIFT
            lab = np.r_[np.zeros(len(Ze)), np.ones(len(Zi))]
            lw_p = LedoitWolf().fit(Z0)
            lw_s = LedoitWolf().fit(Zr)
            a = roc_auc_score(lab, np.r_[md(lw_p.location_, lw_p.precision_, Ze),
                                         md(lw_p.location_, lw_p.precision_, Zi)])
            d, di = md(lw_s.location_, lw_s.precision_, Ze), md(lw_s.location_, lw_s.precision_, Zi)
            d_auc = roc_auc_score(lab, np.r_[(d - d.mean()) / (d.std() + 1e-9),
                                           (di - d.mean()) / (d.std() + 1e-9)])
            gaps.append(d_auc - a)
        null_gaps.append(float(np.mean(gaps)))
    null_gaps = np.array(null_gaps)
    res = {'n_shuffles': NSHUF, 'null_mean': round(float(null_gaps.mean()), 4),
           'null_ci95': [round(float(np.percentile(null_gaps, 2.5)), 4),
                         round(float(np.percentile(null_gaps, 97.5)), 4)],
           'observed_gap': 0.0198,
           'p_perm': round(float((null_gaps >= 0.0198).mean()), 4)}
    print(res)
    import json
    json.dump(res, open(os.path.join(CACHE, 'permute.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
