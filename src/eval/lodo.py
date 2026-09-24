"""Phase 7: robust eval. LODO across species is BLOCKED (equine gated, cattle unlabeled) ->
honest proxies: (a) blur-quartile domain shift within sheep (train sharp -> test blurry bins,
worst-group AUROC, Group-DRO style); (b) ablations B0/B1/B2/B3 with bootstrap CIs;
B3 = late-fusion baseline+MIL; B4 deferred (no identity labels on pain data)."""
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED = 42


def md_feat(P, mu, inv):
    D = P - mu
    return np.sqrt((D @ inv * D).sum(1))


def main():
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    tr = np.load(os.path.join(CACHE, f'sheep_g{GRID}_train_raw.npz'))
    P, y = tr['pooled'], tr['y']
    mu = P[y == 0].mean(0)
    inv = np.linalg.inv(np.cov(P[y == 0].T) + 1e-3 * np.eye(P.shape[1]))
    md_tr = md_feat(P, mu, inv)
    # B1: MLP on [pooled, MD]
    Xb1 = torch.from_numpy(np.c_[P / (np.abs(P).max() + 1e-9), md_tr / (md_tr.max() + 1e-9)]).float()
    net = nn.Sequential(nn.Linear(2049, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.from_numpy(y * 0.9 + 0.05).float()
    idx = rng.permutation(len(y))
    va, tn = idx[:int(0.15 * len(y))], idx[int(0.15 * len(y)):]
    best, w, bs = -1, 0, None
    for ep in range(80):
        net.train()
        opt.zero_grad()
        lf(net(Xb1[tn]).squeeze(-1), yt[tn]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            from sklearn.metrics import roc_auc_score
            try:
                a = roc_auc_score(y[va], torch.sigmoid(net(Xb1[va])).numpy().squeeze())
            except ValueError:
                a = 0.5
        if a > best:
            best, w, bs = a, 0, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            w += 1
            if w >= 12:
                break
    net.load_state_dict(bs)
    torch.save(net.state_dict(), os.path.join(CKPT, 'sheep_b1.pth'))
    b0 = MLP()
    b0.load_state_dict(torch.load(os.path.join(CKPT, 'sheep_b0.pth')))
    mil = AttMIL()
    mil.load_state_dict(torch.load(os.path.join(CKPT, 'sheep_mil_best.pth')))
    b0.eval()
    mil.eval()
    net.eval()
    res, curves = {}, {}
    plt.figure(figsize=(11, 4))
    for pi, split in enumerate(['test', 'test_raw']):
        d = np.load(os.path.join(CACHE, f'sheep_g{GRID}_{split}.npz'))
        Pt, It, yt = d['pooled'], d['inst'], d['y']
        md = md_feat(Pt, mu, inv)
        Xb = torch.from_numpy(np.c_[Pt / (np.abs(P).max() + 1e-9), md / (md_tr.max() + 1e-9)]).float()
        with torch.no_grad():
            s0 = torch.sigmoid(b0(torch.from_numpy(Pt).float())).numpy()
            sm = torch.sigmoid(mil(torch.from_numpy(It).float())[0]).numpy()
            s1 = torch.sigmoid(net(Xb)).numpy().squeeze()
        s3 = (sm + s1) / 2  # B3 late-fusion baseline+MIL
        res[split] = {}
        for name, s in [('B0', s0), ('B1', s1), ('B2-MIL', sm), ('B3-fusion', s3)]:
            res[split][name] = {**metrics(yt, s), 'ci': boot_auroc(yt, s), 'n': len(yt)}
        print(split, {k: (v['auroc'], v['recall'], v['f1']) for k, v in res[split].items()})
        ax = plt.subplot(1, 2, pi + 1)
        for name, s in [('B0', s0), ('B1', s1), ('B2-MIL', sm), ('B3-fusion', s3)]:
            fpr, tpr, _ = roc_curve(yt, s)
            ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
            curves[f'{split}_{name}'] = [fpr.tolist(), tpr.tolist()]
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f'ROC: {split} (n={len(yt)})')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'roc_per_domain.png'), dpi=100)
    # worst-group (blur quartiles) on test_raw
    df = pd.read_parquet(os.path.join(SHEEP, FILES['test_raw']))
    blur = df.blur_score.astype(float).values
    qs = np.quantile(blur, [0.25, 0.5, 0.75])
    grp = np.digitize(blur, qs)
    d = np.load(os.path.join(CACHE, f'sheep_g{GRID}_test_raw.npz'))
    Pt, It, yt = d['pooled'], d['inst'], d['y']
    md = md_feat(Pt, mu, inv)
    Xb = torch.from_numpy(np.c_[Pt / (np.abs(P).max() + 1e-9), md / (md_tr.max() + 1e-9)]).float()
    with torch.no_grad():
        S = {'B0': torch.sigmoid(b0(torch.from_numpy(Pt).float())).numpy(),
             'B1': torch.sigmoid(net(Xb)).numpy().squeeze(),
             'B2-MIL': torch.sigmoid(mil(torch.from_numpy(It).float())[0]).numpy()}
    S['B3-fusion'] = (S['B2-MIL'] + S['B1']) / 2
    wg = {}
    for name, s in S.items():
        aucs = []
        for g in range(4):
            m = grp == g
            if m.sum() > 5 and len(np.unique(yt[m])) == 2:
                from sklearn.metrics import roc_auc_score
                aucs.append(roc_auc_score(yt[m], s[m]))
        wg[name] = {'worst': round(float(min(aucs)), 4), 'groups': [round(float(a), 4) for a in aucs]}
    print('worst-group:', wg)
    res['worst_group_blur'] = wg
    res['note'] = ('LODO across species blocked: equine gated, cattle unlabeled. '
                   'Blur-quartile shift used as quality-shift proxy. B4 deferred: no identity labels on pain data.')
    import json
    json.dump(res, open(os.path.join(CACHE, 'ablation.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
