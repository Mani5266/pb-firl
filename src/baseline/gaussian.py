"""Phase 3: per-animal Gaussian baseline (Ledoit-Wolf) on assumed-reference ROI-12D frames.
CattleFace has no pain labels -> all frames are UNLABELLED assumed reference (not confirmed healthy).
Eval: LOIO injected-deviation AUROC, per-cow vs population."""
import os
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.front_end.geometry import as_kpts, roi_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42
INJECT_DIMS = [3, 4, 5, 7]  # ear_bend_L/R, nostril_dist, muzzle_nostril_R (pain-plausible)
INJECT_STD = 3.0


class PerCowBaseline:
    """Per-animal MVN (Ledoit-Wolf) over globally-standardized ROI features + cold-start fallback."""
    def __init__(self):
        self.cows = {}

    def fit(self, df, K):
        X = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])
        self.mu_g, self.sd_g = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu_g) / self.sd_g
        self.pop = LedoitWolf().fit(Z)
        dpop = np.sqrt(((Z - self.pop.location_) @ self.pop.precision_ *
                        (Z - self.pop.location_)).sum(1))
        self.pop_d_mu, self.pop_d_sd = dpop.mean(), dpop.std() + 1e-9
        for c in df.cow.unique():
            Zc = Z[df.cow.values == c]
            lw = LedoitWolf().fit(Zc)
            dtr = np.sqrt(((Zc - lw.location_) @ lw.precision_ * (Zc - lw.location_)).sum(1))
            self.cows[c] = {'lw': lw, 'd_mu': dtr.mean(), 'd_sd': dtr.std() + 1e-9, 'n': len(Zc)}
        return self

    def _md(self, lw, Z):
        D = Z - lw.location_
        return np.sqrt((D @ lw.precision_ * D).sum(1))

    def deviation(self, X, cows):
        """z-scored Mahalanobis distance on one shared scale; unknown cow ->
        population-calibrated score + low-confidence flag + explicit interval."""
        Z = (X - self.mu_g) / self.sd_g
        out, conf = np.zeros(len(X)), np.ones(len(X))
        for i, (z, c) in enumerate(zip(Z, cows)):
            if c in self.cows:
                e = self.cows[c]
                out[i] = (self._md(e['lw'], z[None])[0] - e['d_mu']) / e['d_sd']
            else:  # cold-start: population z-score (same scale), flag + wider interval
                out[i] = (self._md(self.pop, z[None])[0] - self.pop_d_mu) / self.pop_d_sd
                conf[i] = 0
        return out, conf


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    K = np.stack([as_kpts(k) for k in df.kpts.values])
    X = np.stack([roi_features(k, a) for k, a in zip(K, df.area.values)])
    cows = df.cow.unique()
    res = {}
    for shift in [1.5, 3.0]:
        aucs_pc, aucs_pop, per, pop = [], [], {}, {}
        for c in cows:  # per-cow fit on own half; population fit excludes cow c
            idx = np.where(df.cow.values == c)[0]
            rng.shuffle(idx)
            fit_i, evl_i = idx[:len(idx)//2], idx[len(idx)//2:]
            mu_g = X[~np.isin(np.arange(len(X)), idx)].mean(0)  # LOIO global scaler
            sd_g = X[~np.isin(np.arange(len(X)), idx)].std(0) + 1e-9
            Z, Zfit, Zevl = (X - mu_g)/sd_g, None, None
            Zfit, Zevl = Z[fit_i], Z[evl_i]
            lw_self, lw_pop = LedoitWolf().fit(Zfit), LedoitWolf().fit(Z[np.logical_not(np.isin(np.arange(len(X)), idx))])
            Xinj = X[evl_i].copy()
            Xinj[:, INJECT_DIMS] += shift * sd_g[INJECT_DIMS]
            Zinj = (Xinj - mu_g) / sd_g
            md = lambda lw, ZZ: np.sqrt(((ZZ - lw.location_) @ lw.precision_ * (ZZ - lw.location_)).sum(1))
            d_self, d_inj = md(lw_self, Zevl), md(lw_self, Zinj)
            z_mu, z_sd = d_self.mean(), d_self.std() + 1e-9
            y = np.r_[np.zeros(len(Zevl)), np.ones(len(Zinj))]
            a_pc = roc_auc_score(y, np.r_[(d_self - z_mu)/z_sd, (d_inj - z_mu)/z_sd])
            a_pop = roc_auc_score(y, np.r_[md(lw_pop, Zevl), md(lw_pop, Zinj)])
            aucs_pc.append(a_pc)
            aucs_pop.append(a_pop)
            per[c], pop[c] = round(float(a_pc), 4), round(float(a_pop), 4)
        gap = np.array(aucs_pc) - np.array(aucs_pop)
        t_res = sstats.ttest_rel(aucs_pc, aucs_pop)
        try:
            w_res = sstats.wilcoxon(aucs_pc, aucs_pop)
            w_p = round(float(w_res.pvalue), 4)
        except ValueError:
            w_p = None
        rng_b = np.random.RandomState(SEED)
        boots = [float(np.mean(rng_b.choice(gap, len(gap), replace=True))) for _ in range(2000)]
        res[f'shift_{shift}'] = {'per_cow': per, 'pop': pop,
                                 'mean_per_cow': round(float(np.mean(aucs_pc)), 4),
                                 'mean_pop': round(float(np.mean(aucs_pop)), 4),
                                 'paired_gap_mean': round(float(gap.mean()), 4),
                                 'paired_gap_cluster_ci95': [round(float(np.percentile(boots, 2.5)), 4),
                                                             round(float(np.percentile(boots, 97.5)), 4)],
                                 'paired_t_p': round(float(t_res.pvalue), 4),
                                 'wilcoxon_p': w_p, 'n_cows': len(cows)}
        print(f"shift={shift}", res[f'shift_{shift}']['mean_per_cow'], res[f'shift_{shift}']['mean_pop'])
    import json
    json.dump(res, open(os.path.join(CACHE, 'baseline_loio.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
