"""Phase 5: one-sided upper CUSUM on per-cow z-scored deviation streams.
Streams = frame-ordered RGB frames per cow (frame_id order assumed temporal).
Calibrate threshold on healthy streams for target FAR; validate on injected shifts."""
import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.front_end.geometry import as_kpts, roi_features
from src.baseline.gaussian import INJECT_DIMS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED, KREF = 42, 0.5


def cusum_hits(x, tau, k=KREF):
    s, hits = 0.0, []
    for i, v in enumerate(x):
        s = max(0.0, s + (v - k))
        if s > tau:
            hits.append(i)
            s = 0.0
    return hits


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    K = np.stack([as_kpts(k) for k in df.kpts.values])
    X = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])
    mu_g, sd_g = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu_g) / sd_g
    streams = {}
    for c in df.cow.unique():  # frame order as stream order
        idx = np.where(df.cow.values == c)[0]
        order = np.argsort(df.file.values[idx])  # zero-padded frame names = temporal order
        Zc = Z[idx[order]]
        h = len(Zc) // 2
        lw = LedoitWolf().fit(Zc[:h])
        inv = np.linalg.inv(lw.covariance_)
        md = lambda ZZ: np.sqrt(((ZZ - lw.location_) @ inv * (ZZ - lw.location_)).sum(1))
        d = md(Zc[h:])
        streams[c] = (d - d.mean()) / (d.std() + 1e-9)
    H = np.concatenate([s for s in streams.values()])
    taus = {}
    for far in [0.01, 0.02, 0.05]:
        cand = np.quantile(np.maximum(0, H - KREF), np.linspace(0.9, 0.999, 200))
        best = cand[-1]
        for t in cand:  # per-frame FAR on healthy streams
            alarms = sum(len(cusum_hits(s, t)) for s in streams.values())
            if alarms / len(H) <= far:
                best = t
                break
        taus[far] = float(best)
    res, delays = {}, {}
    for shift in [1.5, 3.0]:
        delays[shift] = {}
        for far, tau in taus.items():
            dl = []
            for c, s in streams.items():
                x = s.copy()
                t0 = len(x) // 3
                x[t0:] += shift * 0.6  # injected mean shift in z-deviation units
                hits = cusum_hits(x, tau)
                dl.append(next((h - t0 for h in hits if h >= t0), len(x) - t0))
            delays[shift][far] = [round(float(np.mean(dl)), 1), len(dl)]
        res[f'shift_{shift}'] = delays[shift]
    print('taus:', {k: round(v, 3) for k, v in taus.items()})
    print(res)
    xs = sorted(taus)
    plt.figure(figsize=(6, 4))
    for shift in [1.5, 3.0]:
        plt.plot(xs, [res[f'shift_{shift}'][f][0] for f in xs], 'o-', label=f'shift={shift}')
    plt.xlabel('false-alarm rate (per-frame, healthy)')
    plt.ylabel('mean detection delay (frames)')
    plt.title('CUSUM delay vs FAR (injected deviations, 5 cows)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'cusum_delay_far.png'), dpi=100)
    # example trace
    c0 = list(streams)[0]
    s = streams[c0].copy()
    s[len(s)//3:] += 1.5 * 0.6
    S, run = [0.0], 0.0
    for v in s:
        run = max(0.0, run + (v - KREF))
        S.append(run)
    plt.figure(figsize=(8, 3))
    plt.plot(s, alpha=0.6, label='z-deviation stream')
    plt.plot(S[1:], label='CUSUM statistic')
    plt.axhline(taus[0.02], color='red', ls='--', label='tau FAR=0.02')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'cusum_trace.png'), dpi=100)
    import json
    json.dump({'taus': {str(k): v for k, v in taus.items()}, 'delays': res, 'k': KREF},
              open(os.path.join(CACHE, 'cusum.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
