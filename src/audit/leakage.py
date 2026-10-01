"""Identity-decodability audit with imbalance-aware reporting.

The audit has two deliberately separate interpretations:

* raw/procrustes representations test whether identity is decodable without a
  cow label at inference;
* per-cow-z and z-deviation representations are identity-conditioned controls.
  They use the true cow's reference statistics and must not be described as an
  identity-blind deployment readout.

All fitting remains reference-only.  Accuracy is retained for compatibility,
but balanced accuracy, macro-F1, the majority baseline, and uniform chance are
reported together so an imbalanced class split cannot masquerade as leakage
suppression.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.covariance import LedoitWolf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.front_end.geometry import as_kpts, align, roi_features, canonical_template
from src.eval.stats import classification_summary

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED = 42


def _fit_score(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return classification_summary(y_test, pred, y_train)


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    K = np.stack([as_kpts(k) for k in df.kpts.values])
    y = df.cow.values
    cows = np.unique(y)
    R = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])

    # Reference/probe split per cow.  No probe frame contributes to a fitted
    # template, scaler, covariance, classifier, or per-cow normalization.
    ref_idx, probe_idx = [], []
    for c in cows:
        ix = np.where(y == c)[0].copy()
        rng.shuffle(ix)
        h = len(ix) // 2
        ref_idx.extend(ix[:h])
        probe_idx.extend(ix[h:])
    ref_idx, probe_idx = np.array(ref_idx), np.array(probe_idx)
    yr, yp = y[ref_idx], y[probe_idx]

    tpl = canonical_template(K[ref_idx])
    Pr = np.stack([align(k, tpl) for k in K[ref_idx]]).reshape(len(ref_idx), -1)
    Pp = np.stack([align(k, tpl) for k in K[probe_idx]]).reshape(len(probe_idx), -1)
    Rr, Rp = R[ref_idx], R[probe_idx]

    rmean = {c: Rr[yr == c].mean(0) for c in cows}
    rstd = {c: Rr[yr == c].std(0) + 1e-9 for c in cows}
    Zr = np.stack([(r - rmean[c]) / rstd[c] for r, c in zip(Rr, yr)])
    Zp = np.stack([(r - rmean[c]) / rstd[c] for r, c in zip(Rp, yp)])

    # z-deviation scalar: global scaler + per-cow Ledoit-Wolf fit on reference.
    mu_g, sd_g = Rr.mean(0), Rr.std(0) + 1e-9
    Gr = (Rr - mu_g) / sd_g
    lws = {c: LedoitWolf().fit(Gr[yr == c]) for c in cows}
    dev = np.zeros(len(R))
    G = (R - mu_g) / sd_g
    for c in cows:
        m = y == c
        inv = lws[c].precision_
        d = np.sqrt(((G[m] - lws[c].location_) @ inv * (G[m] - lws[c].location_)).sum(1))
        cow_indices = np.where(m)[0]
        dref = d[np.isin(cow_indices, ref_idx)]
        dev[m] = (d - dref.mean()) / (dref.std() + 1e-9)

    reps = {
        'raw ROI-12D': (Rr, Rp),
        'procrustes 26D': (Pr, Pp),
        'per-cow-z ROI-12D': (Zr, Zp),
        'z-deviation scalar': (dev[ref_idx].reshape(-1, 1),
                               dev[probe_idx].reshape(-1, 1)),
    }
    probes = {'LDA': LinearDiscriminantAnalysis(), '5NN': KNeighborsClassifier(5)}
    accuracy = {}
    probe_metrics = {}
    for pname, clf in probes.items():
        for rname, (Xtr, Xte) in reps.items():
            m = _fit_score(clf, Xtr, yr, Xte, yp)
            key = f'{pname}_{rname}'
            accuracy[key] = m['accuracy']
            probe_metrics[key] = m
            print(f"{pname} {rname}: accuracy={m['accuracy']:.3f}, "
                  f"balanced={m['balanced_accuracy']:.3f}, "
                  f"majority={m['majority_accuracy']:.3f}, "
                  f"chance={m['uniform_chance']:.3f}")

    # Plot the primary imbalance-aware metric, not the majority-sensitive raw
    # accuracy that caused the old LDA headline to be misleading.
    plot_keys = list(probe_metrics)
    plt.figure(figsize=(10, 4))
    plt.bar(plot_keys, [probe_metrics[k]['balanced_accuracy'] for k in plot_keys])
    majority = probe_metrics[plot_keys[0]]['majority_accuracy']
    chance = probe_metrics[plot_keys[0]]['uniform_chance']
    constant_balanced = probe_metrics[plot_keys[0]]['balanced_constant_baseline']
    plt.axhline(constant_balanced, color='darkorange', ls='--',
                label=f'constant-class balanced baseline ({constant_balanced:.2f})')
    plt.axhline(chance, color='red', ls=':', label=f'uniform guess ({chance:.2f})')
    plt.ylabel('balanced cow-identity accuracy')
    plt.title('Identity decodability (reference-fit; balanced metrics)')
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    plt.savefig(os.path.join(FIG, 'identity_decodability.png'), dpi=100)

    result = {
        # Keep the old flat field so existing report consumers do not break.
        'acc': accuracy,
        'metrics': probe_metrics,
        'majority_accuracy': majority,
        'uniform_chance': chance,
        'class_counts': {str(c): int((yp == c).sum()) for c in cows},
        'n_cows': len(cows),
        'ref_n': len(ref_idx),
        'probe_n': len(probe_idx),
        'protocol': ('per-cow ref/probe split; template/stats/classifiers fit on ref; '
                     'probe scored once; balanced accuracy is primary'),
        'identity_conditioned_controls': ['per-cow-z ROI-12D', 'z-deviation scalar'],
        'identity_conditioned_warning': (
            'per-cow-z and z-deviation use the true cow label to select reference '
            'statistics; they are controls, not identity-blind deployment features'),
        'session_probe': 'not available: manifest_rgb has no session label',
        'limitations': ('random within-sequence split; adjacent frames and cow/session '
                        'confounding remain; no cow-clustered probe confidence intervals'),
    }
    json.dump(result, open(os.path.join(CACHE, 'identity_audit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
