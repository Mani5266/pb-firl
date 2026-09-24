"""Phase 6: identity-decodability audit. Cow-ID probe (LDA 5-fold) on frozen
representations: raw ROI-12D, procrustes coords, per-cow-z ROI, 1-D z-deviation.
Requirement: pain features near chance (0.20) while retaining pain signal (Phase 3/4)."""
import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.front_end.geometry import as_kpts, align, roi_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')


def main():
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    K = np.stack([as_kpts(k) for k in df.kpts.values])
    y = df.cow.values
    chance = 1 / len(np.unique(y))
    R = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])
    tpl = np.load(os.path.join(CACHE, 'template_13.npy'))
    P = np.stack([align(k, tpl) for k in K]).reshape(len(K), -1)
    Z = np.stack([(r - R[y == c].mean(0)) / (R[y == c].std(0) + 1e-9) for r, c in zip(R, y)])
    # 1-D pain scalar: per-cow z-scored Mahalanobis distance (healthy-only fit)
    mu_g, sd_g = R.mean(0), R.std(0) + 1e-9
    G = (R - mu_g) / sd_g
    dev = np.zeros(len(G))
    for c in np.unique(y):
        m = y == c
        lw = LedoitWolf().fit(G[m])
        inv = np.linalg.inv(lw.covariance_)
        d = np.sqrt(((G[m] - lw.location_) @ inv * (G[m] - lw.location_)).sum(1))
        dev[m] = (d - d.mean()) / (d.std() + 1e-9)
    reps = {'raw ROI-12D': R, 'procrustes 26D': P,
            'per-cow-z ROI-12D': Z, 'z-deviation scalar': dev.reshape(-1, 1)}
    res = {}
    for name, X in reps.items():
        acc = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=5).mean()
        res[name] = round(float(acc), 4)
        print(f'{name}: cow-ID acc={acc:.3f} (chance={chance:.2f})')
    plt.figure(figsize=(7, 4))
    plt.bar(list(res), list(res.values()))
    plt.axhline(chance, color='red', ls='--', label=f'chance ({chance:.2f})')
    plt.ylabel('cow-identity LDA accuracy (5-fold)')
    plt.title('Identity decodability of frozen representations')
    plt.xticks(rotation=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'identity_decodability.png'), dpi=100)
    import json
    json.dump({'acc': res, 'chance': chance, 'n_cows': len(np.unique(y)), 'n': len(df)},
              open(os.path.join(CACHE, 'identity_audit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
