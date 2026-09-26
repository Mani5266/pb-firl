"""Sheep test_raw stats for the matched B0 (pool+BCE, seed 42): AUPRC + cluster bootstrap
(resample source files, not frames) for AUROC/AUPRC. Torch inference, fast."""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pain.train_sheep import GRID, SHEEP, FILES
from src.pain.improve_mil import grouped_strat_split
from src.pain.fair_compare import MeanPool, fit_cell
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42


def main():
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    tr = np.load(os.path.join(CACHE, f'sheep_g{GRID}_train_raw.npz'))
    Xi, y = tr['inst'], tr['y'].astype(float)
    df_tr = pd.read_parquet(os.path.join(SHEEP, FILES['train_raw']))
    tn, va = grouped_strat_split(tr['y'], df_tr.source_filename.values)
    pos_rate = (y[tn] == 1).mean()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos_rate) / pos_rate]))
    m = fit_cell(MeanPool(), torch.from_numpy(Xi[tn]).float(), y[tn],
                 torch.from_numpy(Xi[va]).float(), y[va], bce, SEED)
    m.eval()
    d = np.load(os.path.join(CACHE, f'sheep_g{GRID}_test_raw.npz'))
    with torch.no_grad():
        s = torch.sigmoid(m(torch.from_numpy(d['inst']).float())[0]).numpy()
    yt = d['y']
    print('auroc:', round(roc_auc_score(yt, s), 4), 'auprc:', round(average_precision_score(yt, s), 4),
          'base rate:', round(float(yt.mean()), 4))
    df = pd.read_parquet(os.path.join(SHEEP, FILES['test_raw']))
    files = df.source_filename.values
    uniq = np.unique(files)
    au_r, ap_r = [], []
    for _ in range(1000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.where(np.isin(files, pick))[0]
        if len(np.unique(yt[idx])) < 2:
            continue
        au_r.append(roc_auc_score(yt[idx], s[idx]))
        ap_r.append(average_precision_score(yt[idx], s[idx]))
    res = {'auroc': round(float(roc_auc_score(yt, s)), 4),
           'auprc': round(float(average_precision_score(yt, s)), 4),
           'auroc_cluster_ci': [round(float(np.percentile(au_r, 2.5)), 4),
                                round(float(np.percentile(au_r, 97.5)), 4)],
           'auprc_cluster_ci': [round(float(np.percentile(ap_r, 2.5)), 4),
                                round(float(np.percentile(ap_r, 97.5)), 4)],
           'n': len(yt), 'n_clusters': len(uniq)}
    print(res)
    import json
    json.dump(res, open(os.path.join(CACHE, 'sheep_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
