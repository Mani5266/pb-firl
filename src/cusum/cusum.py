"""Phase 5: one-sided upper CUSUM on per-cow z-scored deviation streams (leakage-safe).
Chronological protocol (review fix): baseline fit on first 40% of each cow's stream,
threshold calibration on next 30% (healthy), delay evaluation on final 30% with injection.
Eval segments standardised with reference stats only."""
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
    mu_g, sd_g = None, None  # scaler fit per-cow on reference below (was: global fit = leak)
    Z = X
    cal, evl = {}, {}
    for c in df.cow.unique():  # frame order as stream order
        idx = np.where(df.cow.values == c)[0]
        order = np.argsort(df.file.values[idx])  # zero-padded frame names = temporal order
        Xc = X[idx[order]]
        n = len(Xc)
        i1, i2 = int(0.4 * n), int(0.7 * n)
        mu_g, sd_g = Xc[:i1].mean(0), Xc[:i1].std(0) + 1e-9  # ref-only scaler
        Zc = (Xc - mu_g) / sd_g
        n = len(Zc)
        i1, i2 = int(0.4 * n), int(0.7 * n)
        lw = LedoitWolf().fit(Zc[:i1])
        inv = np.linalg.inv(lw.covariance_)
        md = lambda ZZ: np.sqrt(((ZZ - lw.location_) @ inv * (ZZ - lw.location_)).sum(1))
        dref = md(Zc[:i1])
        mu_d, sd_d = dref.mean(), dref.std() + 1e-9
        cal[c] = (md(Zc[i1:i2]) - mu_d) / sd_d
        evl[c] = (md(Zc[i2:]) - mu_d) / sd_d
    H = np.concatenate([s for s in cal.values()])
    taus, met = {}, {}
    for far in [0.01, 0.02, 0.05]:
        cand = np.quantile(np.maximum(0, H - KREF), np.linspace(0.9, 0.999, 200))
        best, ok = cand[-1], False
        for t in cand:  # per-frame FAR on calibration segments only
            alarms = sum(len(cusum_hits(s, t)) for s in cal.values())
            if alarms / len(H) <= far:
                best, ok = t, True
                break
        taus[far] = float(best)
        met[far] = ok  # False = constraint NOT met at any candidate (saturated)
    res = {}
    for shift in [1.5, 3.0]:
        res[f'shift_{shift}'] = {}
        for far, tau in taus.items():
            ach = sum(len(cusum_hits(s, tau)) for s in evl.values()) / sum(len(s) for s in evl.values())
            dl, miss = [], 0
            for c, s in evl.items():
                x = s.copy()
                t0 = len(x) // 3
                x[t0:] += shift * 0.6  # injected mean shift in z-deviation units
                hits = cusum_hits(x, tau)
                det = [h - t0 for h in hits if h >= t0]
                if det:
                    dl.append(det[0])
                else:
                    miss += 1  # censored: reported separately, not averaged in
            res[f'shift_{shift}'][far] = {'delay': round(float(np.mean(dl)), 1) if dl else None,
                                          'miss': miss, 'n': len(evl),
                                          'achieved_far': round(float(ach), 4)}
    print('taus:', {k: round(v, 3) for k, v in taus.items()}, 'constraint_met:', met)
    print(res)
    xs = sorted(taus)
    plt.figure(figsize=(6, 4))
    for shift in [1.5, 3.0]:
        plt.plot(xs, [res[f'shift_{shift}'][f]['delay'] for f in xs], 'o-', label=f'shift={shift}')
    plt.xlabel('false-alarm target (calibration)')
    plt.ylabel('mean detection delay, detected only (frames, held-out eval)')
    plt.title('CUSUM delay vs FAR target (chronological, 5 cows)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'cusum_delay_far.png'), dpi=100)
    c0 = list(evl)[0]
    s = evl[c0].copy()
    s[len(s)//3:] += 1.5 * 0.6
    S, run = [0.0], 0.0  # same reset logic as cusum_hits
    for v in s:
        run = max(0.0, run + (v - KREF))
        if run > taus[0.02]:
            S.append(run)
            run = 0.0
        else:
            S.append(run)
    plt.figure(figsize=(8, 3))
    plt.plot(s, alpha=0.6, label='z-deviation stream (eval)')
    plt.plot(S[1:], label='CUSUM statistic (resets on alarm, as scored)')
    plt.axhline(taus[0.02], color='red', ls='--', label='tau FAR=0.02')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'cusum_trace.png'), dpi=100)
    import json
    json.dump({'taus': {str(k): v for k, v in taus.items()}, 'constraint_met': {str(k): v for k, v in met.items()},
               'delays': res, 'k': KREF,
               'protocol': 'ref-only scaler; fit 40% / calibrate 30% / eval 30% chronological per cow; misses censored'},
              open(os.path.join(CACHE, 'cusum.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
