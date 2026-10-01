"""Matched sheep proxy evaluation and quality-shift sensitivity.

This is not species LODO: no equine/cattle pain domain with compatible labels is
available.  The only executable robustness analysis is a test-set quality
stratification whose cut-points are fit on train_raw, never on test_raw.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pain.train_sheep import MLP, AttMIL, metrics, boot_auroc, GRID, FILES, SHEEP
from src.pain.improve_mil import grouped_strat_split
from src.pain.fair_compare import MeanPool, fit_cell
from src.eval.stats import cluster_bootstrap_auc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEEDS = [42, 43, 44]


def md_feat(P, mu, inv):
    D = P - mu
    return np.sqrt((D @ inv * D).sum(1))


def main():
    tr = np.load(os.path.join(CACHE, f'sheep_g{GRID}_train_raw.npz'))
    P, Xi, y = tr['pooled'], tr['inst'], tr['y'].astype(float)
    df_tr = pd.read_parquet(os.path.join(SHEEP, FILES['train_raw']))
    tn, va = grouped_strat_split(tr['y'], df_tr.source_filename.values)
    pos_rate = (y[tn] == 1).mean()
    pw = torch.tensor([(1 - pos_rate) / pos_rate])
    mu = P[tn][y[tn] == 0].mean(0)
    inv = np.linalg.inv(np.cov(P[tn][y[tn] == 0].T) + 1e-3 * np.eye(P.shape[1]))
    md_tr = md_feat(P[tn], mu, inv)
    Xb1_tr = torch.from_numpy(np.c_[P[tn] / (np.abs(P[tn]).max() + 1e-9),
                                    md_tr / (md_tr.max() + 1e-9)]).float()
    md_va = md_feat(P[va], mu, inv)
    Xb1_va = torch.from_numpy(np.c_[P[va] / (np.abs(P[tn]).max() + 1e-9),
                                    md_va / (md_tr.max() + 1e-9)]).float()
    d = np.load(os.path.join(CACHE, f'sheep_g{GRID}_test_raw.npz'))
    Pt, It, yt = d['pooled'], d['inst'], d['y']
    md_te = md_feat(Pt, mu, inv)
    Xb1_te = torch.from_numpy(np.c_[Pt / (np.abs(P[tn]).max() + 1e-9),
                                    md_te / (md_tr.max() + 1e-9)]).float()
    Xt, Xti = torch.from_numpy(Pt).float(), torch.from_numpy(It).float()
    S = {'B0': [], 'B1': [], 'B2-MIL': []}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        b0 = fit_cell(MeanPool(), torch.from_numpy(Xi[tn]).float(), y[tn],
                      torch.from_numpy(Xi[va]).float(), y[va], bce, seed)
        net = nn.Sequential(nn.Linear(2049, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        b1 = fit_cell(_Head(net), Xb1_tr, y[tn], Xb1_va, y[va], bce, seed)
        b2 = fit_cell(AttMIL(), torch.from_numpy(Xi[tn]).float(), y[tn],
                      torch.from_numpy(Xi[va]).float(), y[va], bce, seed)
        for m in (b0, b1, b2):
            m.eval()
        with torch.no_grad():
            S['B0'].append(torch.sigmoid(b0(Xti)[0]).numpy())
            S['B1'].append(torch.sigmoid(b1(Xb1_te)[0]).numpy().squeeze())
            S['B2-MIL'].append(torch.sigmoid(b2(Xti)[0]).numpy())
    S['B3-fusion'] = [(a + b) / 2 for a, b in zip(S['B2-MIL'], S['B1'])]
    res = {'protocol': 'matched: shared split, BCE-only, 3 seeds; primary=test_raw', 'test_raw': {}}
    mean_scores = {}
    for name, ss in S.items():
        sm = np.mean(ss, axis=0)
        mean_scores[name] = sm
        res['test_raw'][name] = {**metrics(yt, sm), 'ci': boot_auroc(yt, sm), 'n': len(yt),
                                 'seed_aurocs': [round(float(__import__('sklearn').metrics.roc_auc_score(yt, s)), 4) for s in ss]}
        print(name, res['test_raw'][name]['auroc'], res['test_raw'][name]['seed_aurocs'])
    plt.figure(figsize=(6, 4))
    for name, sm in mean_scores.items():
        fpr, tpr, _ = roc_curve(yt, sm)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC: test_raw (matched models, mean of 3 seeds)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'roc_per_domain.png'), dpi=100)
    df = pd.read_parquet(os.path.join(SHEEP, FILES['test_raw']))
    # Test cut-points are not allowed to define the groups being evaluated.
    train_blur = df_tr.blur_score.astype(float).values
    cuts = np.quantile(train_blur, [0.25, 0.5, 0.75])
    grp = np.digitize(df.blur_score.astype(float).values, cuts)
    wg = {}
    for name, sm in mean_scores.items():
        groups = []
        for g in range(4):
            m = grp == g
            entry = {'group': g, 'n': int(m.sum()), 'positives': int(yt[m].sum())}
            if m.sum() > 5 and len(np.unique(yt[m])) == 2:
                from sklearn.metrics import roc_auc_score
                entry['auroc'] = round(float(roc_auc_score(yt[m], sm[m])), 4)
                entry['ci_cluster'] = cluster_bootstrap_auc(
                    yt[m], sm[m], df.source_filename.values[m], n_resamples=1000)
            else:
                entry['auroc'] = None
                entry['ci_cluster'] = [None, None]
            groups.append(entry)
        valid = [g['auroc'] for g in groups if g['auroc'] is not None]
        wg[name] = {
            'worst': round(float(min(valid)), 4) if valid else None,
            'groups': groups,
        }
    print('worst-group:', {k: v['worst'] for k, v in wg.items()})
    res['worst_group_blur'] = wg
    res['quality_group_cuts_train_raw'] = [round(float(x), 6) for x in cuts]
    res['quality_group_protocol'] = (
        'quartile cut-points fit on train_raw; groups evaluated on test_raw; '
        'source-file clustered bootstrap CIs')
    res['lodo_status'] = 'blocked: no compatible equine/cattle pain domain labels'
    res['note'] = ('Blur stratification is a quality-shift proxy, not species LODO. '
                   'B4 is deferred because pain data have no identity labels.')
    torch.save(b0.state_dict(), os.path.join(CKPT, 'sheep_b0_matched.pth'))
    import json
    json.dump(res, open(os.path.join(CACHE, 'ablation.json'), 'w'), indent=1)


class _Head(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(-1), None


if __name__ == '__main__':
    main()
