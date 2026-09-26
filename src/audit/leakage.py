"""Phase 6: identity-decodability audit (leakage-safe, review fix).
Per-cow reference/probe split; stats/baselines fit on reference only; probes (LDA + 5NN)
trained on reference, scored on probe. Includes nonlinear probe."""
import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.front_end.geometry import as_kpts, align, roi_features, canonical_template

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED = 42


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    K = np.stack([as_kpts(k) for k in df.kpts.values])
    y = df.cow.values
    cows = np.unique(y)
    chance = 1 / len(cows)
    R = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])
    # reference/probe split per cow
    ref_idx, probe_idx = [], []
    for c in cows:
        ix = np.where(y == c)[0].copy()
        rng.shuffle(ix)
        h = len(ix) // 2
        ref_idx.extend(ix[:h])
        probe_idx.extend(ix[h:])
    ref_idx, probe_idx = np.array(ref_idx), np.array(probe_idx)
    yr, yp = y[ref_idx], y[probe_idx]
    # template fit on reference only
    tpl = canonical_template(K[ref_idx])
    Pr = np.stack([align(k, tpl) for k in K[ref_idx]]).reshape(len(ref_idx), -1)
    Pp = np.stack([align(k, tpl) for k in K[probe_idx]]).reshape(len(probe_idx), -1)
    Rr, Rp = R[ref_idx], R[probe_idx]
    rmean = {c: Rr[yr == c].mean(0) for c in cows}
    rstd = {c: Rr[yr == c].std(0) + 1e-9 for c in cows}
    Zr = np.stack([(r - rmean[c]) / rstd[c] for r, c in zip(Rr, yr)])
    Zp = np.stack([(r - rmean[c]) / rstd[c] for r, c in zip(Rp, yp)])
    # z-deviation scalar: global scaler + per-cow LW fit on reference only
    mu_g, sd_g = Rr.mean(0), Rr.std(0) + 1e-9
    Gr = (Rr - mu_g) / sd_g
    lws = {}
    for c in cows:
        lws[c] = LedoitWolf().fit(Gr[yr == c])
    dev = np.zeros(len(R))
    G = (R - mu_g) / sd_g
    for c in cows:
        m = (y == c)
        inv = np.linalg.inv(lws[c].covariance_)
        d = np.sqrt(((G[m] - lws[c].location_) @ inv * (G[m] - lws[c].location_)).sum(1))
        dref = d[np.isin(np.where(m)[0], ref_idx)]
        dev[m] = (d - dref.mean()) / (dref.std() + 1e-9)
    reps = {'raw ROI-12D': (Rr, Rp), 'procrustes 26D': (Pr, Pp),
            'per-cow-z ROI-12D': (Zr, Zp), 'z-deviation scalar': (dev[ref_idx].reshape(-1, 1),
                                                                 dev[probe_idx].reshape(-1, 1))}
    probes = {'LDA': LinearDiscriminantAnalysis(), '5NN': KNeighborsClassifier(5)}
    res = {}
    for pname, clf in probes.items():
        for rname, (Xtr, Xte) in reps.items():
            clf.fit(Xtr, yr)
            a = float(clf.score(Xte, yp))
            res[f'{pname}_{rname}'] = round(a, 4)
            print(f'{pname} {rname}: {a:.3f} (chance={chance:.2f})')
    plt.figure(figsize=(9, 4))
    plt.bar(list(res), list(res.values()))
    plt.axhline(chance, color='red', ls='--', label=f'chance ({chance:.2f})')
    plt.ylabel('cow-identity accuracy (train ref, score probe)')
    plt.title('Identity decodability of frozen representations (leakage-safe)')
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'identity_decodability.png'), dpi=100)
    import json
    json.dump({'acc': res, 'chance': chance, 'n_cows': len(cows),
               'ref_n': len(ref_idx), 'probe_n': len(probe_idx),
               'protocol': 'per-cow ref/probe split; stats on ref; train ref score probe'},
              open(os.path.join(CACHE, 'identity_audit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
