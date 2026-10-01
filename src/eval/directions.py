"""Perturbation-direction sensitivity (P0-4): 100 random 8-dim directions per cow at
shift 1.5, same PCA pool/cows/budgets as falsify. Reports mean + SD over directions
per control: is the own-vs-pop gap direction-conditional?"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42
NREF, NEVL, NDIM, SHIFT, NDIR = 20, 20, 8, 1.5, 100


def md(loc, inv, Z):
    D = Z - loc
    return np.sqrt((D @ inv * D).sum(1))


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_recow.parquet'))
    E = np.load(os.path.join(CACHE, 'dairy_emb.npz'))['E']
    y = df.cow.values
    counts = {c: (y == c).sum() for c in np.unique(y)}
    ordered = sorted(counts, key=counts.get, reverse=True)
    pool_cows = ordered[:30]
    pca = PCA(n_components=64, random_state=SEED).fit(E[np.isin(y, pool_cows)])
    P = pca.transform(E)
    eval_cows = [c for c in ordered[30:] if counts[c] >= NREF + NEVL][:60]
    dirs = [rng.choice(64, NDIM, replace=False) for _ in range(NDIR)]
    gap_d_a, gap_e_d, gap_d_f, own_m, pop_m = [], [], [], [], []
    for c in eval_cows:
        ix = np.where(y == c)[0].copy()
        rng.shuffle(ix)
        ref, evl = ix[:NREF], ix[NREF:NREF + NEVL]
        others = np.setdiff1d(np.where(~np.isin(y, [c] + pool_cows))[0], ix)
        mu_g = P[others].mean(0)
        sd_g = P[others].std(0) + 1e-9
        Z0 = (P[others] - mu_g) / sd_g
        Zr, Ze = (P[ref] - mu_g) / sd_g, (P[evl] - mu_g) / sd_g
        lw_p = LedoitWolf().fit(Z0)
        inv_p = np.linalg.inv(lw_p.covariance_)
        lw_s = LedoitWolf().fit(Zr)
        inv_s = np.linalg.inv(lw_s.covariance_)
        m_o = Zr.mean(0)
        cov_e = 0.5 * lw_s.covariance_ + 0.5 * lw_p.covariance_
        inv_e = np.linalg.inv(cov_e)
        w = rng.choice([o for o in eval_cows if o != c])
        ixw = np.where(y == w)[0].copy()
        rng.shuffle(ixw)
        lw_w = LedoitWolf().fit((P[ixw[:NREF]] - mu_g) / sd_g)
        inv_w = np.linalg.inv(lw_w.covariance_)
        a, d, e, f = [], [], [], []
        for dim in dirs:
            Zi = Ze.copy()
            Zi[:, dim] += SHIFT
            lab = np.r_[np.zeros(len(Ze)), np.ones(len(Zi))]
            a.append(roc_auc_score(lab, np.r_[md(lw_p.location_, inv_p, Ze),
                                              md(lw_p.location_, inv_p, Zi)]))
            dd = md(lw_s.location_, inv_s, Ze)
            di = md(lw_s.location_, inv_s, Zi)
            d.append(roc_auc_score(lab, np.r_[(dd - dd.mean()) / (dd.std() + 1e-9),
                                              (di - dd.mean()) / (dd.std() + 1e-9)]))
            e.append(roc_auc_score(lab, np.r_[md(m_o, inv_e, Ze), md(m_o, inv_e, Zi)]))
            f.append(roc_auc_score(lab, np.r_[md(lw_w.location_, inv_w, Ze),
                                              md(lw_w.location_, inv_w, Zi)]))
        a, d, e, f = (np.array(v) for v in (a, d, e, f))
        gap_d_a.append((d - a).mean())
        gap_e_d.append((e - d).mean())
        gap_d_f.append((d - f).mean())
        own_m.append(d.mean())
        pop_m.append(a.mean())
    res = {
        'n_cows': len(eval_cows), 'n_dirs': NDIR, 'shift': SHIFT,
        'own_mean_over_dirs': round(float(np.mean(own_m)), 4),
        'pop_mean_over_dirs': round(float(np.mean(pop_m)), 4),
        'gap_D_minus_A': {'mean': round(float(np.mean(gap_d_a)), 4),
                          'sd_over_cows': round(float(np.std(gap_d_a)), 4)},
        'gap_E_minus_D': {'mean': round(float(np.mean(gap_e_d)), 4),
                          'sd_over_cows': round(float(np.std(gap_e_d)), 4)},
        'gap_D_minus_F': {'mean': round(float(np.mean(gap_d_f)), 4),
                          'sd_over_cows': round(float(np.std(gap_d_f)), 4)},
        'frac_dirs_D_beats_A': round(float(np.mean([g > 0 for g in gap_d_a])), 4),
    }
    print(res)
    import json
    json.dump(res, open(os.path.join(CACHE, 'directions.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
