"""P0-1 cross-session detection: reference = cow's largest session, probe = second
largest. Compares own-ref vs session-matched pop vs other-session pop vs wrong-cow,
same frozen dims/budgets. Cow-clustered bootstrap CI on gaps."""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42
NREF, NEVL, NDIM, SHIFT = 20, 20, 8, 1.5


def md(loc, inv, Z):
    D = Z - loc
    return np.sqrt((D @ inv * D).sum(1))


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_recow.parquet'))
    E = np.load(os.path.join(CACHE, 'dairy_emb.npz'))['E']
    y, sess = df.cow.values, df.session.values
    counts = {c: (y == c).sum() for c in np.unique(y)}
    ordered = sorted(counts, key=counts.get, reverse=True)
    pool = ordered[:30]
    pca = PCA(n_components=64, random_state=SEED).fit(E[np.isin(y, pool)])
    P = pca.transform(E)
    gmu = P[np.isin(y, pool)].mean(0)
    gsd = P[np.isin(y, pool)].std(0) + 1e-9
    Z = (P - gmu) / gsd
    # eligible: >=NREF in largest session, >=NEVL in second largest
    elig = []
    for c in ordered[30:]:
        ix = np.where(y == c)[0]
        by_s = sorted(((s, ix[sess[ix] == s]) for s in np.unique(sess[ix])),
                        key=lambda t: len(t[1]), reverse=True)
        if len(by_s) >= 2 and len(by_s[0][1]) >= NREF and len(by_s[1][1]) >= NEVL:
            elig.append((c, by_s[0][0], by_s[1][0]))
    elig = elig[:60]
    print('eligible cows:', len(elig))
    dim = rng.choice(64, NDIM, replace=False)
    acc = {k: [] for k in ['own', 'pop_same', 'pop_other', 'wrong']}
    for c, s_ref, s_prb in elig:
        ixr = np.where((y == c) & (sess == s_ref))[0].copy()
        ixp = np.where((y == c) & (sess == s_prb))[0].copy()
        rng.shuffle(ixr)
        rng.shuffle(ixp)
        Zr, Ze = Z[ixr[:NREF]], Z[ixp[:NEVL]]
        Zi = Ze.copy()
        Zi[:, dim] += SHIFT
        lab = np.r_[np.zeros(len(Ze)), np.ones(len(Zi))]
        others = [o for o, _, _ in elig if o != c]
        # session-matched pop: other cows' probe-session frames (>=5 each)
        Ip = np.concatenate([np.where((y == o) & (sess == s_prb))[0][:10] for o in others
                             if ((y == o) & (sess == s_prb)).sum() >= 5])
        # other-session pop: other cows' ref-session frames (>=5 each)
        Io = np.concatenate([np.where((y == o) & (sess == s_ref))[0][:10] for o in others
                             if ((y == o) & (sess == s_ref)).sum() >= 5])
        if len(Ip) < 10 or len(Io) < 10:
            continue
        lw_o = LedoitWolf().fit(Zr)
        inv_o = lw_o.precision_
        lw_ps = LedoitWolf().fit(Z[Ip])
        inv_ps = lw_ps.precision_
        lw_po = LedoitWolf().fit(Z[Io])
        inv_po = lw_po.precision_
        w = rng.choice(others)
        # wrong cow's own largest-session frames (session-matched to ITS reference)
        sw = pd.Series(sess[np.where(y == w)[0]]).value_counts().idxmax()
        ixw = np.where((y == w) & (sess == sw))[0].copy()
        rng.shuffle(ixw)
        lw_w = LedoitWolf().fit(Z[ixw[:NREF]])
        inv_w = lw_w.precision_
        acc['own'].append(roc_auc_score(lab, np.r_[md(lw_o.location_, inv_o, Ze),
                                                   md(lw_o.location_, inv_o, Zi)]))
        acc['pop_same'].append(roc_auc_score(lab, np.r_[md(lw_ps.location_, inv_ps, Ze),
                                                        md(lw_ps.location_, inv_ps, Zi)]))
        acc['pop_other'].append(roc_auc_score(lab, np.r_[md(lw_po.location_, inv_po, Ze),
                                                         md(lw_po.location_, inv_po, Zi)]))
        acc['wrong'].append(roc_auc_score(lab, np.r_[md(lw_w.location_, inv_w, Ze),
                                                              md(lw_w.location_, inv_w, Zi)]))
    out = {'n_cows': len(elig),
           'means': {k: round(float(np.mean(v)), 4) for k, v in acc.items()}}
    for base in ['pop_same', 'pop_other', 'wrong']:
        gap = np.array(acc['own']) - np.array(acc[base])
        boots = [float(np.mean(rng.choice(gap, len(gap), replace=True))) for _ in range(2000)]
        out[f'gap_own_minus_{base}'] = {
            'mean': round(float(gap.mean()), 4),
            'ci95': [round(float(np.percentile(boots, 2.5)), 4),
                     round(float(np.percentile(boots, 97.5)), 4)]}
    print(out)
    import json
    json.dump(out, open(os.path.join(CACHE, 'xsession.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
