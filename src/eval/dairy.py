"""Dairy-scale personalisation validation (ReCowGnition, 161 Holstein cows, no pain labels).
Frozen ResNet-50 embeddings -> PCA-64: variance decomposition, LDA identity probe (raw vs
per-cow-z), LOIO per-cow vs population injected-shift AUROC, cross-session distance analysis."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED = 42


def main():
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_recow.parquet'))
    out = os.path.join(CACHE, 'dairy_emb.npz')
    if not os.path.exists(out):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = resnet50(weights=ResNet50_Weights.DEFAULT)
        feat = nn.Sequential(*list(m.children())[:-2]).eval().to(dev)
        tf = ResNet50_Weights.DEFAULT.transforms()
        E = []
        with torch.no_grad():
            for i in range(0, len(df), 64):
                imgs = [tf(Image.open(p).convert('RGB')) for p in df.file.iloc[i:i+64]]
                E.append(feat(torch.stack(imgs).to(dev)).mean((2, 3)).cpu().numpy())
        E = np.concatenate(E)
        np.savez_compressed(out, E=E)
        print('embedded', E.shape)
    E = np.load(out)['E']
    y = df.cow.values
    sess = df.session.values
    cows = np.unique(y)
    print('cows:', len(cows), 'chance:', round(1/len(cows), 4))
    P = PCA(n_components=64, random_state=SEED).fit_transform(E)
    # variance decomposition
    mu = P.mean(0)
    sb = sum((y == c).sum() * ((P[y == c].mean(0) - mu) ** 2).sum() for c in cows)
    st = ((P - mu) ** 2).sum()
    between = float(sb / (st + 1e-12))
    acc_raw = float(cross_val_score(LinearDiscriminantAnalysis(), P, y, cv=5).mean())
    Z = np.stack([(r - P[y == c].mean(0)) / (P[y == c].std(0) + 1e-9) for r, c in zip(P, y)])
    acc_z = float(cross_val_score(LinearDiscriminantAnalysis(), Z, y, cv=5).mean())
    print(f'between={between:.3f} LDA raw={acc_raw:.3f} LDA per-cow-z={acc_z:.3f}')
    # LOIO injected-shift: cows with >=10 imgs
    big = [c for c in cows if (y == c).sum() >= 10]
    print('loio cows:', len(big))
    res_loio = {}
    for shift in [1.5, 3.0]:
        apc, apo = [], []
        dim = rng.choice(64, 8, replace=False)
        for c in big:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            h = idx[:len(idx)//2]
            rest = np.setdiff1d(np.arange(len(y)), idx)
            mu_g = P[rest].mean(0)
            sd_g = P[rest].std(0) + 1e-9
            Zf, Ze = (P[h] - mu_g)/sd_g, (P[np.setdiff1d(idx, h)] - mu_g)/sd_g
            lw_s, lw_p = LedoitWolf().fit(Zf), LedoitWolf().fit((P[rest]-mu_g)/sd_g)
            md = lambda lw, ZZ: np.sqrt(((ZZ-lw.location_) @ np.linalg.inv(lw.covariance_) * (ZZ-lw.location_)).sum(1))
            Zi = Ze.copy()
            Zi[:, dim] += shift
            d, di = md(lw_s, Ze), md(lw_s, Zi)
            lab = np.r_[np.zeros(len(Ze)), np.ones(len(Zi))]
            apc.append(roc_auc_score(lab, np.r_[(d-d.mean())/(d.std()+1e-9), (di-d.mean())/(d.std()+1e-9)]))
            apo.append(roc_auc_score(lab, np.r_[md(lw_p, Ze), md(lw_p, Zi)]))
        res_loio[f'shift_{shift}'] = {'mean_per_cow': round(float(np.mean(apc)), 4),
                                      'mean_pop': round(float(np.mean(apo)), 4)}
        print(f'shift={shift} per-cow={np.mean(apc):.4f} pop={np.mean(apo):.4f}')
    # cross-session distances (cosine on PCA-64, sampled pairs)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    d_same_sess, d_cross_sess, d_diff = [], [], []
    by_cow = {c: np.where(y == c)[0] for c in big}
    for c, ix in by_cow.items():
        s = sess[ix]
        for _ in range(20):
            a, b = rng.choice(ix, 2, replace=False)
            (d_same_sess if s[ix.tolist().index(a)] == s[ix.tolist().index(b)] else d_cross_sess).append(
                float(1 - Pn[a] @ Pn[b]))
    for _ in range(2000):
        a = rng.randint(len(y))
        b = rng.randint(len(y))
        if y[a] != y[b]:
            d_diff.append(float(1 - Pn[a] @ Pn[b]))
    res = {'n_cows': len(cows), 'n': len(df), 'chance': round(1/len(cows), 4),
           'between': round(between, 4), 'lda_raw': round(acc_raw, 4), 'lda_percowz': round(acc_z, 4),
           'loio': res_loio, 'loio_cows': len(big),
           'dist': {'same_sess': round(float(np.mean(d_same_sess)), 4),
                    'cross_sess': round(float(np.mean(d_cross_sess)), 4),
                    'diff_cow': round(float(np.mean(d_diff)), 4)}}
    print(res['dist'])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(['raw', 'per-cow-z'], [acc_raw, acc_z])
    plt.axhline(1/len(cows), color='red', ls='--')
    plt.title(f'Dairy cow-ID LDA (161 cows, chance={1/len(cows):.3f})')
    plt.subplot(1, 2, 2)
    plt.bar(['same\nsession', 'cross\nsession', 'different\ncow'],
            [res['dist']['same_sess'], res['dist']['cross_sess'], res['dist']['diff_cow']])
    plt.title('Mean cosine distance (PCA-64)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'dairy_audit.png'), dpi=100)
    import json
    json.dump(res, open(os.path.join(CACHE, 'dairy.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
