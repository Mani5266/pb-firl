"""Fair matched B0-vs-MIL comparison (review fix). ONE shared stratified 80/20 split of
train_raw (seed 42 for the split); 4 cells = {mean-pool, attention} x {BCE, focal} with matched
budgets (AdamW 1e-4, patience 15, max 120 epochs, no label smoothing anywhere); 3 train seeds.
Primary eval: test_raw (dhash-disjoint from train). Balanced `test` reported ONLY with
contamination flag (60/74 dhashes overlap train_raw). Clean-source subset as sensitivity check."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pain.train_sheep import AttMIL, metrics, boot_auroc, GRID, SHEEP, FILES
from src.pain.improve_mil import FocalLoss, grouped_strat_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEEDS = [42, 43, 44]
EPOCHS, PATIENCE = 120, 15


class MeanPool(nn.Module):
    """Same head capacity as B0-MLP but over mean-pooled instances (matched comparator)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, h):
        return self.net(h.mean(1)).squeeze(-1), None


def fit_cell(model, Xtr, ytr, Xva, yva, lossf, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    yt = torch.from_numpy(ytr).float()
    best, wait, bs = -1, 0, None
    from sklearn.metrics import roc_auc_score
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        lossf(model(Xtr)[0], yt).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            try:
                a = roc_auc_score(yva, torch.sigmoid(model(Xva)[0]).numpy())
            except ValueError:
                a = 0.5
        if a > best:
            best, wait = a, 0
            bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(bs)
    return model


def main():
    tr = np.load(os.path.join(CACHE, f'sheep_g{GRID}_train_raw.npz'))
    Xi, y = tr['inst'], tr['y'].astype(float)
    df_tr = pd.read_parquet(os.path.join(SHEEP, FILES['train_raw']))
    tn, va = grouped_strat_split(tr['y'], df_tr.source_filename.values)  # source-grouped shared split
    pos_rate = (y[tn] == 1).mean()
    cells = {
        'pool+BCE': (MeanPool, False),
        'pool+focal': (MeanPool, True),
        'attn+BCE': (AttMIL, False),
        'attn+focal': (AttMIL, True),
    }
    # eval sets
    ev = {}
    for split in ['test_raw', 'test']:
        d = np.load(os.path.join(CACHE, f'sheep_g{GRID}_{split}.npz'))
        ev[split] = (d['inst'], d['y'])
    df_tr = pd.read_parquet(os.path.join(SHEEP, FILES['train_raw']))
    df_te = pd.read_parquet(os.path.join(SHEEP, FILES['test_raw']))
    clean_mask = ~df_te.source_filename.isin(set(df_tr.source_filename))
    d = np.load(os.path.join(CACHE, f'sheep_g{GRID}_test_raw.npz'))
    ev['test_raw_cleanSrc'] = (d['inst'][clean_mask.values], d['y'][clean_mask.values])
    print('clean-src rows:', clean_mask.sum(), 'pos:', int(d['y'][clean_mask.values].sum()))
    res = {'protocol': 'shared strat 80/20 split; matched budgets; 3 seeds; primary=test_raw',
           'contamination_note': 'balanced test shares 60/74 dhashes with train_raw: FOR DIRECTION ONLY',
           'cells': {}}
    for name, (cls, focal) in cells.items():
        aucs = {k: [] for k in ev}
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = cls()
            if focal:
                lf = FocalLoss(alpha=1 - pos_rate)
            else:
                lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos_rate) / pos_rate]))
            m = fit_cell(m, torch.from_numpy(Xi[tn]).float(), y[tn],
                         torch.from_numpy(Xi[va]).float(), y[va], lf, seed)
            m.eval()
            with torch.no_grad():
                for k, (Xe, ye) in ev.items():
                    s = torch.sigmoid(m(torch.from_numpy(Xe).float())[0]).numpy()
                    from sklearn.metrics import roc_auc_score
                    try:
                        aucs[k].append(float(roc_auc_score(ye, s)))
                    except ValueError:
                        aucs[k].append(float('nan'))
        res['cells'][name] = {k: {'seeds': [round(v, 4) for v in v],
                                  'mean': round(float(np.nanmean(v)), 4)} for k, v in aucs.items()}
        print(name, {k: res['cells'][name][k]['mean'] for k in aucs})
    import json
    json.dump(res, open(os.path.join(CACHE, 'pain_fair.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
