"""Round 1: MIL improvement sweep. Val-selected (stratified 80/20 of train_raw),
test/test_raw reported once. Variants: per-image instance std, focal loss, capacity."""
import os
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pain.train_sheep import AttMIL, metrics, boot_auroc, GRID

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints')
SEED = 42


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.a, self.g = alpha, gamma

    def forward(self, logit, y):
        bce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction='none')
        p = torch.sigmoid(logit)
        w = self.a * (1 - p) ** self.g * y + (1 - self.a) * p ** self.g * (1 - y)
        return (w * bce).mean()


def strat_split(y, frac=0.2):
    rng = np.random.RandomState(SEED)
    va = np.r_[rng.choice(np.where(y == 0)[0], int((y == 0).sum() * frac), replace=False),
               rng.choice(np.where(y == 1)[0], int((y == 1).sum() * frac), replace=False)]
    rng.shuffle(va)
    return np.setdiff1d(np.arange(len(y)), va), va


def grouped_strat_split(y, groups, frac=0.2):
    """Split by source group (all crops of one source file in one partition)."""
    rng = np.random.RandomState(SEED)
    y = np.asarray(y)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    glab = np.array([np.round(y[groups == g].mean()) for g in uniq]).astype(int)
    va_g = np.r_[rng.choice(uniq[glab == 0], max(1, int((glab == 0).sum() * frac)), replace=False),
                 rng.choice(uniq[glab == 1], max(1, int((glab == 1).sum() * frac)), replace=False)]
    va = np.where(np.isin(groups, va_g))[0]
    rng.shuffle(va)
    return np.setdiff1d(np.arange(len(y)), va), va


def fit(model, Xtr, ytr, Xva, yva, lossf, epochs=120):
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    yt = torch.from_numpy(ytr).float()
    best, wait, bs = -1, 0, None
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
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
            if wait >= 15:
                break
    model.load_state_dict(bs)
    return model, round(float(best), 4)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tr = np.load(os.path.join(CACHE, f'sheep_g{GRID}_train_raw.npz'))
    Xi, y = tr['inst'], tr['y'].astype(float)
    tn, va = strat_split(tr['y'])
    pos_rate = (y[tn] == 1).mean()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos_rate) / pos_rate]))
    variants = {
        'V0-baseline': (Xi, False, bce, 128),
        'V1-imgstd': (None, False, bce, 128),
        'V2-imgstd+focal': (None, True, None, 128),
        'V3-focal': (Xi, True, None, 128),
        'V4-imgstd+focal+D256': (None, True, None, 256),
    }
    scored = {}
    for name, (Xv, focal, lossf, d) in variants.items():
        X = ((Xi - Xi.mean(1, keepdims=True)) / (Xi.std(1, keepdims=True) + 1e-9)
             if Xv is None else Xv)
        lf = FocalLoss(alpha=1 - pos_rate) if focal else lossf
        m = AttMIL(d=d)
        m, vacc = fit(m, torch.from_numpy(X[tn]).float(), y[tn],
                      torch.from_numpy(X[va]).float(), y[va], lf)
        scored[name] = (vacc, m, X)
        print(f'{name}: val_auroc={vacc}', flush=True)
    best_name = max(scored, key=lambda k: scored[k][0])
    print('SELECTED:', best_name)
    _, m, X = scored[best_name]
    torch.save(m.state_dict(), os.path.join(CKPT, f'sheep_mil_best.pth'))
    m.eval()
    res = {'selected': best_name,
           'val': {k: v[0] for k, v in scored.items()}}
    with torch.no_grad():
        for split in ['test', 'test_raw']:
            if split == 'test' and os.environ.get('PB_ALLOW_CONTAMINATED') != '1':
                print('SKIP contaminated balanced test (60/74 dhash overlap with train); '
                      'set PB_ALLOW_CONTAMINATED=1 to score it anyway')
                continue
            d = np.load(os.path.join(CACHE, f'sheep_g{GRID}_{split}.npz'))
            Xt, yt = d['inst'], d['y']
            if 'imgstd' in best_name:
                Xt = (Xt - Xt.mean(1, keepdims=True)) / (Xt.std(1, keepdims=True) + 1e-9)
            sm, A = m(torch.from_numpy(Xt).float())
            sm = torch.sigmoid(sm).numpy()
            res[f'MIL_{split}'] = {**metrics(yt, sm), 'ci': boot_auroc(yt, sm), 'n': len(yt)}
            print(split, res[f'MIL_{split}'])
    import json
    json.dump(res, open(os.path.join(CACHE, 'pain_mil_v2.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
