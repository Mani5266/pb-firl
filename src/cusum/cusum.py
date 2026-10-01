"""Chronological CUSUM evaluation on reference-derived feature scores.

This module is intentionally a descriptive early-warning experiment, not a
claim of real pain detection.  The reference, calibration, and evaluation
windows are chronological and disjoint.  Threshold candidates are derived
from the *running CUSUM statistic* (not per-frame excess), and synthetic
changes are injected into feature coordinates before rescoring the Gaussian
model rather than being added to its scalar output.
"""

import json
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
from src.eval.stats import binomial_miss_upper_bound

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED, KREF = 42, 0.5


def cusum_statistic(x, k=KREF):
    """Return the one-sided running statistic without alarm resets."""
    run = 0.0
    out = np.zeros(len(x), dtype=float)
    for i, value in enumerate(np.asarray(x, dtype=float)):
        run = max(0.0, run + (value - k))
        out[i] = run
    return out


def cusum_hits(x, tau, k=KREF):
    """Return alarm indices for a resetting one-sided upper CUSUM."""
    run, hits = 0.0, []
    for i, value in enumerate(np.asarray(x, dtype=float)):
        run = max(0.0, run + (value - k))
        if run > tau:
            hits.append(i)
            run = 0.0
    return hits


def summarize_false_alarm(streams, tau, k=KREF):
    """Report both frame and stream false-alarm rates.

    The stream-level rate is the estimand used for threshold calibration.  The
    frame-level rate is included because it is often reported in legacy runs;
    with five cows neither estimate has useful precision, so both are marked
    as descriptive in the saved protocol.
    """
    streams = [np.asarray(s, dtype=float) for s in streams]
    hits = [cusum_hits(s, tau, k) for s in streams]
    n_frames = sum(len(s) for s in streams)
    n_streams = len(streams)
    return {
        'stream_far': round(float(sum(bool(h) for h in hits) / max(n_streams, 1)), 4),
        'frame_alarm_rate': round(float(sum(len(h) for h in hits) / max(n_frames, 1)), 4),
        'alarms': int(sum(len(h) for h in hits)),
        'streams': n_streams,
        'frames': int(n_frames),
    }


def calibrate_threshold(streams, target_far, k=KREF):
    """Choose the smallest threshold whose calibration stream-FAR is met.

    Candidate values are maxima of the no-reset running statistic.  Thus the
    threshold is selected from the same CUSUM quantity that is scored.  With a
    tiny number of streams a 0.01 or 0.02 target is not resolvable; the result
    explicitly records that limitation rather than silently treating a
    frame-level rate as an event-level FAR.
    """
    streams = [np.asarray(s, dtype=float) for s in streams]
    if not streams:
        raise ValueError('at least one calibration stream is required')
    maxima = np.array([float(np.max(cusum_statistic(s, k))) if len(s) else 0.0
                       for s in streams])
    candidates = np.unique(np.r_[0.0, maxima])
    candidates.sort()
    chosen = float(candidates[-1])
    met = False
    for tau in candidates:
        summary = summarize_false_alarm(streams, float(tau), k)
        if summary['stream_far'] <= target_far:
            chosen, met = float(tau), True
            break
    # The strict ``> tau`` rule means the largest observed maximum is already
    # a valid no-alarm threshold. Add epsilon only if floating-point ties make
    # the check ambiguous.
    summary = summarize_false_alarm(streams, chosen, k)
    return chosen, met, summary, maxima


def inject_feature_shift(features, t0, dims, shift):
    """Inject a persistent coordinate shift into a feature stream."""
    out = np.asarray(features, dtype=float).copy()
    if out.ndim != 2:
        raise ValueError('features must have shape (time, dimensions)')
    if t0 < 0 or t0 > len(out):
        raise ValueError('t0 outside feature stream')
    dims = np.asarray(dims, dtype=int)
    if np.any(dims < 0) or np.any(dims >= out.shape[1]):
        raise ValueError('injection dimension outside feature stream')
    out[t0:, dims] += float(shift)
    return out


def _model_streams(Xc):
    """Fit the reference model and return disjoint score/feature windows."""
    n = len(Xc)
    i1, i2 = int(0.4 * n), int(0.7 * n)
    if i1 < 2 or i2 <= i1 or i2 >= n:
        raise ValueError(f'stream too short for 40/30/30 split: n={n}')
    mu, sd = Xc[:i1].mean(0), Xc[:i1].std(0) + 1e-9
    Zc = (Xc - mu) / sd
    lw = LedoitWolf().fit(Zc[:i1])
    inv = lw.precision_

    def md(Z):
        D = Z - lw.location_
        return np.sqrt(((D @ inv) * D).sum(1))

    dref = md(Zc[:i1])
    mu_d, sd_d = dref.mean(), dref.std() + 1e-9

    def score(Z):
        return (md(Z) - mu_d) / sd_d

    return {
        'cal': score(Zc[i1:i2]),
        'evl': score(Zc[i2:]),
        'eval_features': Zc[i2:],
        'score': score,
        'n_features': Zc.shape[1],
    }


def main():
    rng = np.random.RandomState(SEED)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    K = np.stack([as_kpts(k) for k in df.kpts.values])
    X = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])

    streams = {}
    for c in df.cow.unique():
        idx = np.where(df.cow.values == c)[0]
        order = np.argsort(df.file.values[idx])  # zero-padded frame names
        streams[c] = _model_streams(X[idx[order]])

    cal_streams = [v['cal'] for v in streams.values()]
    taus, calibration, met = {}, {}, {}
    for far in [0.01, 0.02, 0.05]:
        tau, ok, summary, maxima = calibrate_threshold(cal_streams, far, KREF)
        taus[far] = float(tau)
        met[far] = bool(ok)
        calibration[far] = {**summary,
                            'target_stream_far': far,
                            'max_cusum': [round(float(x), 4) for x in maxima],
                            'resolution': f'1/{len(cal_streams)} stream(s)',
                            'target_below_one_stream_resolution': bool(
                                far < 1 / len(cal_streams))}

    # Use a fixed, predeclared random direction per cow.  The perturbation is
    # applied to Z features and then rescored by the fitted Gaussian model.
    direction = {c: rng.choice(v['n_features'], min(3, v['n_features']), replace=False)
                 for c, v in streams.items()}
    healthy_eval = [v['evl'] for v in streams.values()]
    res = {}
    for shift in [1.5, 3.0]:
        res[f'shift_{shift}'] = {}
        injected = {}
        for c, info in streams.items():
            f = inject_feature_shift(info['eval_features'], len(info['eval_features']) // 3,
                                     direction[c], shift)
            injected[c] = info['score'](f)
        for far, tau in taus.items():
            healthy = summarize_false_alarm(healthy_eval, tau, KREF)
            delays, misses, pre_event = [], 0, 0
            n_events = len(injected)
            for c, x in injected.items():
                t0 = len(x) // 3
                hits = cusum_hits(x, tau, KREF)
                pre_event += sum(h < t0 for h in hits)
                det = [h - t0 for h in hits if h >= t0]
                if det:
                    delays.append(det[0])
                else:
                    misses += 1
            res[f'shift_{shift}'][far] = {
                'delay': round(float(np.mean(delays)), 1) if delays else None,
                'delay_detected_only': True,
                'miss': int(misses),
                'miss_upper95': round(binomial_miss_upper_bound(misses, n_events), 4),
                'detected': int(len(delays)),
                'n': int(n_events),
                'pre_event_alarms': int(pre_event),
                'healthy_eval_far': healthy,
            }

    print('taus:', {k: round(v, 3) for k, v in taus.items()},
          'calibration_target_met:', met)
    print(res)

    xs = sorted(taus)
    plt.figure(figsize=(6, 4))
    for shift in [1.5, 3.0]:
        ys = [res[f'shift_{shift}'][f]['delay'] for f in xs]
        plt.plot(xs, [np.nan if y is None else y for y in ys], 'o-', label=f'shift={shift}')
    plt.xlabel('target calibration stream FAR')
    plt.ylabel('mean detection delay (frames; detected only)')
    plt.title('CUSUM delay vs target FAR (chronological, descriptive)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'cusum_delay_far.png'), dpi=100)

    c0 = list(streams)[0]
    s = streams[c0]['evl'].copy()
    t0 = len(s) // 3
    sf = inject_feature_shift(streams[c0]['eval_features'], t0, direction[c0], 1.5)
    s = streams[c0]['score'](sf)
    run = cusum_statistic(s, KREF)
    plt.figure(figsize=(8, 3))
    plt.plot(s, alpha=0.6, label='feature-rescored deviation stream')
    plt.plot(run, label='CUSUM statistic (no reset, display)')
    plt.axvline(t0, color='black', ls=':', label='injection onset')
    plt.axhline(taus[0.02], color='red', ls='--', label='tau target FAR=0.02')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'cusum_trace.png'), dpi=100)

    result = {
        'taus': {str(k): v for k, v in taus.items()},
        'constraint_met': {str(k): v for k, v in met.items()},
        'calibration': {str(k): v for k, v in calibration.items()},
        'delays': {str(k): {str(f): v for f, v in val.items()} for k, val in res.items()},
        'k': KREF,
        'protocol': ('reference 40% / calibration 30% / evaluation 30% chronological per cow; '
                     'threshold candidates are calibration max-CUSUM values; healthy evaluation '
                     'FAR reported separately; misses include exact one-sided 95% upper bound'),
        'injection': ('persistent shift in three randomly selected standardized feature '
                      'coordinates at one-third of the evaluation window, then rescored through '
                      'the fitted Ledoit-Wolf model; not a real event'),
        'n_events': len(streams),
        'interpretation': 'descriptive synthetic-shift stress test; no real pain/change labels',
    }
    json.dump(result, open(os.path.join(CACHE, 'cusum.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
