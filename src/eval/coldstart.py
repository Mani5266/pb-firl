"""Cold-start: how many reference frames does a new cow need before per-cow beats population?
ReCowGnition embeddings, PCA-64 (per-cow PCA-on-rest), injected shift 1.5, AUROC vs n_ref."""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED = 42
NREFS = [5, 10, 20, 40]
SHIFT = 1.5


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_recow.parquet'))
    E = np.load(os.path.join(CACHE, 'dairy_emb.npz'))['E']
    y = df.cow.values
    big = [c for c in np.unique(y) if (y == c).sum() >= 60]
    cows = list(rng.choice(big, min(40, len(big)), replace=False))
    print('cows:', len(cows))
    curves = {n: ([], []) for n in NREFS}
    for c in cows:
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        evl = idx[:20]
        pool = idx[20:]
        rest = np.setdiff1d(np.arange(len(y)), idx)
        pc = PCA(n_components=64, random_state=SEED).fit(E[rest])
        R0, Re, Rp = pc.transform(E[rest]), pc.transform(E[evl]), pc.transform(E[pool])
        mu_g = R0.mean(0)
        sd_g = R0.std(0) + 1e-9
        Z0, Ze = (R0 - mu_g) / sd_g, (Re - mu_g) / sd_g
        lw_p = LedoitWolf().fit(Z0)
        md = lambda lw, ZZ: np.sqrt(((ZZ - lw.location_) @ lw.precision_ * (ZZ - lw.location_)).sum(1))
        dim = rng.choice(64, 8, replace=False)
        Zi = Ze.copy()
        Zi[:, dim] += SHIFT
        lab = np.r_[np.zeros(len(Ze)), np.ones(len(Zi))]
        pop_auc = roc_auc_score(lab, np.r_[md(lw_p, Ze), md(lw_p, Zi)])
        for n in NREFS:
            ref = Rp[rng.choice(len(Rp), min(n, len(Rp)), replace=False)]
            Zr = (ref - mu_g) / sd_g
            lw_s = LedoitWolf().fit(Zr)
            d, di = md(lw_s, Ze), md(lw_s, Zi)
            a = roc_auc_score(lab, np.r_[(d - d.mean()) / (d.std() + 1e-9),
                                         (di - d.mean()) / (d.std() + 1e-9)])
            curves[n][0].append(a)
            curves[n][1].append(pop_auc)
    res = {'n_ref': {}, 'pop_mean': round(float(np.mean(curves[NREFS[0]][1])), 4)}
    for n in NREFS:
        res['n_ref'][str(n)] = {'per_cow_mean': round(float(np.mean(curves[n][0])), 4),
                                'cows': len(curves[n][0])}
    print(res)
    plt.figure(figsize=(6, 4))
    plt.plot(NREFS, [res['n_ref'][str(n)]['per_cow_mean'] for n in NREFS], 'o-', label='per-cow')
    plt.axhline(res['pop_mean'], color='red', ls='--', label=f"population ({res['pop_mean']})")
    plt.xscale('log')
    plt.xlabel('reference frames for new cow')
    plt.ylabel('injected-shift AUROC (shift=1.5)')
    plt.title('Cold-start: per-cow vs population')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'coldstart.png'), dpi=100)
    import json
    json.dump(res, open(os.path.join(CACHE, 'coldstart.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
