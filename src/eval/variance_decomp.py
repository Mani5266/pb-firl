"""Phase 2 FIRST EXPERIMENT: between-cow vs within-cow variance (GT kpts, RGB).
Leakage-safe (review fix): per-cow stats fit on reference half, all metrics on probe half;
probe classifier trained on reference, scored on probe."""
import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


def scatter_share(X, y):
    mu = X.mean(0)
    st = ((X - mu) ** 2).sum()
    sb = sum(len(X[y == c]) * ((X[y == c].mean(0) - mu) ** 2).sum() for c in np.unique(y))
    return float(sb / (st + 1e-12))


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    K = np.stack([as_kpts(k) for k in df.kpts.values])
    y = df.cow.values
    cows = np.unique(y)
    n_cows = len(cows)
    print('n=', len(df), 'cows=', sorted(cows))
    ref_idx, probe_idx = [], []
    for c in cows:
        ix = np.where(y == c)[0].copy()
        rng.shuffle(ix)
        h = len(ix) // 2
        ref_idx.extend(ix[:h])
        probe_idx.extend(ix[h:])
    ref_idx, probe_idx = np.array(ref_idx), np.array(probe_idx)
    R = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])
    tpl = canonical_template(K[ref_idx])
    P = np.stack([align(k, tpl) for k in K])
    rmean = {c: R[ref_idx][y[ref_idx] == c].mean(0) for c in cows}
    rstd = {c: R[ref_idx][y[ref_idx] == c].std(0) + 1e-9 for c in cows}
    Z = np.stack([(r - rmean[c]) / rstd[c] for r, c in zip(R, y)])
    yp = y[probe_idx]
    reps = {'raw-centered 26D': (K - K.mean(1, keepdims=True)).reshape(len(K), -1),
            'procrustes 26D': P.reshape(len(K), -1),
            'ROI-12D raw': R,
            'ROI-12D per-cow-z (ref-fit)': Z}
    rows = []
    for name, X in reps.items():
        b = scatter_share(X[probe_idx], yp)
        clf = LinearDiscriminantAnalysis().fit(X[ref_idx], y[ref_idx])
        acc = float(clf.score(X[probe_idx], yp))
        rows.append((name, b, 1 - b, acc))
        print(f'{name}: between={b:.3f} within={1-b:.3f} cow-LDA ref->probe={acc:.3f}')
    names = [r[0] for r in rows]
    x = np.arange(len(names))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, [r[1] for r in rows], label='between-cow')
    plt.bar(x, [r[2] for r in rows], bottom=[r[1] for r in rows], label='within-cow')
    plt.xticks(x, names, rotation=12)
    plt.ylabel('share of total scatter (probe half)')
    plt.title(f'Facial-geometry variance, ref-fit stats (5 cows, chance={1/n_cows:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'variance_bars.png'), dpi=100)
    with open(os.path.join(ROOT, 'reports', 'variance_decomposition.md'), 'w') as f:
        f.write('# Variance decomposition (Phase 2, GT keypoints, RGB, 5 cows x 1890 imgs)\n\n')
        f.write(f'Protocol: per-cow reference/probe split; stats fit on reference, metrics on probe; '
                f'LDA trained on reference, scored on probe. Chance cow accuracy: {1/n_cows:.2f}.\n\n')
        f.write('| representation | between | within | cow-LDA acc |\n|---|---|---|---|\n')
        for n, b, w, a in rows:
            f.write(f'| {n} | {b:.3f} | {w:.3f} | {a:.3f} |\n')
        f.write('\nInterpretation: Procrustes alignment concentrates identity (pose nuisance removed); '
                'per-cow z-scoring with reference-fit stats cuts linear cow-decodability on unseen '
                'frames. Identity labels assume folders = 02_13 sequences (see data/README).\n')
    import json
    json.dump({n: {'between': b, 'within': w, 'cow_lda': a, 'n_cows': n_cows, 'n': len(df)}
               for n, b, w, a in rows},
              open(os.path.join(CACHE, 'variance.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
