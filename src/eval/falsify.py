"""Falsification battery for the central claim (Astra review S3).
Same PCA features, same eval cows/frames, same frozen perturbation dims, same reference
budget (20 frames) across all controls. PCA fit once on a disjoint pool of cows.
Controls: A pop-zero-shot; B pop+own-ref calibration; C own-mean+pop-cov; D own mean+cov;
E shrinkage(lambda sweep); F matched wrong-animal reference; G own-mean+pooled-within-cov.
Per-cow values saved + cow-clustered bootstrap CIs on gaps."""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42
NREF, NEVL, NDIM = 20, 20, 8
SHIFTS = [1.5, 3.0]
LAMBDAS = [0.25, 0.5, 0.75]


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
    print('eval cows:', len(eval_cows))
    dim = rng.choice(64, NDIM, replace=False)  # frozen across shifts AND cows
    res = {}
    for shift in SHIFTS:
        keys = ['A_pop', 'B_popCal', 'C_meanOnly', 'D_full',
                *[f'E_shrink_{l}' for l in LAMBDAS], 'F_wrong', 'G_withinCov']
        acc = {k: [] for k in keys}
        for c in eval_cows:
            ix = np.where(y == c)[0].copy()
            rng.shuffle(ix)
            ref, evl = ix[:NREF], ix[NREF:NREF + NEVL]
            others = np.setdiff1d(np.where(~np.isin(y, [c] + pool_cows))[0], ix)
            mu_g = P[others].mean(0)
            sd_g = P[others].std(0) + 1e-9
            Z0 = (P[others] - mu_g) / sd_g
            Zr, Ze = (P[ref] - mu_g) / sd_g, (P[evl] - mu_g) / sd_g
            Zi = Ze.copy()
            Zi[:, dim] += shift
            lab = np.r_[np.zeros(len(Ze)), np.ones(len(Zi))]
            lw_p = LedoitWolf().fit(Z0)
            inv_p = np.linalg.inv(lw_p.covariance_)
            sA_h, sA_i = md(lw_p.location_, inv_p, Ze), md(lw_p.location_, inv_p, Zi)
            acc['A_pop'].append(roc_auc_score(lab, np.r_[sA_h, sA_i]))
            cal = sA_h  # healthy scores of THIS cow under population model
            acc['B_popCal'].append(roc_auc_score(
                lab, np.r_[(sA_h - cal.mean()) / (cal.std() + 1e-9),
                            (sA_i - cal.mean()) / (cal.std() + 1e-9)]))
            m_o = Zr.mean(0)
            acc['C_meanOnly'].append(roc_auc_score(
                lab, np.r_[md(m_o, inv_p, Ze), md(m_o, inv_p, Zi)]))
            lw_s = LedoitWolf().fit(Zr)
            inv_s = np.linalg.inv(lw_s.covariance_)
            d, di = md(lw_s.location_, inv_s, Ze), md(lw_s.location_, inv_s, Zi)
            acc['D_full'].append(roc_auc_score(
                lab, np.r_[(d - d.mean()) / (d.std() + 1e-9), (di - d.mean()) / (d.std() + 1e-9)]))
            for l in LAMBDAS:
                cov_e = l * lw_s.covariance_ + (1 - l) * lw_p.covariance_
                inv_e = np.linalg.inv(cov_e)
                acc[f'E_shrink_{l}'].append(roc_auc_score(
                    lab, np.r_[md(m_o, inv_e, Ze), md(m_o, inv_e, Zi)]))
            w = rng.choice([o for o in eval_cows if o != c])
            ixw = np.where(y == w)[0].copy()
            rng.shuffle(ixw)
            Zw = (P[ixw[:NREF]] - mu_g) / sd_g
            lw_w = LedoitWolf().fit(Zw)
            inv_w = np.linalg.inv(lw_w.covariance_)
            acc['F_wrong'].append(roc_auc_score(
                lab, np.r_[md(lw_w.location_, inv_w, Ze), md(lw_w.location_, inv_w, Zi)]))
            # G: own mean + pooled within-cow covariance (strong baseline)
            centered = []
            for o in eval_cows:
                if o == c:
                    continue
                ixo = np.where(y == o)[0]
                Zo = (P[ixo[:NREF]] - mu_g) / sd_g
                centered.append(Zo - Zo.mean(0))
            lw_wc = LedoitWolf().fit(np.concatenate(centered))
            inv_wc = np.linalg.inv(lw_wc.covariance_)
            acc['G_withinCov'].append(roc_auc_score(
                lab, np.r_[md(m_o, inv_wc, Ze), md(m_o, inv_wc, Zi)]))
        means = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
        res[f'shift_{shift}'] = means
        gaps = {}
        for a, b in [('D_full', 'A_pop'), ('E_shrink_0.5', 'D_full'),
                     ('D_full', 'F_wrong'), ('G_withinCov', 'A_pop')]:
            gap = np.array(acc[a]) - np.array(acc[b])
            boots = [float(np.mean(rng.choice(gap, len(gap), replace=True))) for _ in range(2000)]
            gaps[f'{a}_minus_{b}'] = {'mean': round(float(gap.mean()), 4),
                                      'ci95': [round(float(np.percentile(boots, 2.5)), 4),
                                               round(float(np.percentile(boots, 97.5)), 4)]}
        res[f'shift_{shift}']['gaps'] = gaps
        res[f'shift_{shift}']['per_cow'] = {k: [round(float(v), 4) for v in vals]
                                            for k, vals in acc.items()}
        print(f'shift={shift}', means)
    import json
    json.dump({'controls': res, 'n_cows': len(eval_cows), 'nref': NREF,
               'protocol': 'shared PCA pool; frozen dims; matched 20-frame budgets'},
              open(os.path.join(CACHE, 'falsify.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
