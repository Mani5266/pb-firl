"""Cross-session dairy probe (Astra S2): reference = cow's largest session, probe = other
sessions. PCA + stats fit on pooled reference only; LDA/5NN trained on reference, scored
on probe. Reports eligibility (cows with >=8 ref and >=4 probe frames)."""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from src.eval.stats import classification_summary

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_recow.parquet'))
    E = np.load(os.path.join(CACHE, 'dairy_emb.npz'))['E']
    y, sess = df.cow.values, df.session.values
    cows = np.unique(y)
    ref_idx, probe_idx = [], []
    elig = 0
    for c in cows:
        ix = np.where(y == c)[0]
        by_s = {s: ix[sess[ix] == s] for s in np.unique(sess[ix])}
        if len(by_s) < 2:
            continue
        big = max(by_s, key=lambda s: len(by_s[s]))
        r, p = by_s[big], np.concatenate([v for k, v in by_s.items() if k != big])
        if len(r) >= 8 and len(p) >= 4:
            elig += 1
            ref_idx.extend(r)
            probe_idx.extend(p)
    ref_idx, probe_idx = np.array(ref_idx), np.array(probe_idx)
    print('eligible cows:', elig, 'ref:', len(ref_idx), 'probe:', len(probe_idx))
    pca = PCA(n_components=64, random_state=SEED).fit(E[ref_idx])
    Pr, Pp = pca.transform(E[ref_idx]), pca.transform(E[probe_idx])
    yr, yp = y[ref_idx], y[probe_idx]
    rmean = {c: Pr[yr == c].mean(0) for c in np.unique(yr)}
    rstd = {c: Pr[yr == c].std(0) + 1e-9 for c in np.unique(yr)}
    Zr = np.stack([(r - rmean[c]) / rstd[c] for r, c in zip(Pr, yr)])
    Zp = np.stack([(r - rmean[c]) / rstd[c] for r, c in zip(Pp, yp)])
    res = {'eligible_cows': elig, 'ref_n': len(ref_idx), 'probe_n': len(yp),
           'chance': round(1 / len(np.unique(yr)), 4), 'probe_metrics': {}}
    for pname, clf in [('LDA', LinearDiscriminantAnalysis()),
                       ('5NN', KNeighborsClassifier(5))]:
        for rname, Xtr, Xte in [('raw', Pr, Pp), ('per-cow-z', Zr, Zp)]:
            clf.fit(Xtr, yr)
            pred = clf.predict(Xte)
            m = classification_summary(yp, pred, yr)
            key = f'{pname}_{rname}'
            res[key] = m['accuracy']
            res['probe_metrics'][key] = m
            print(f"{pname} {rname}: accuracy={m['accuracy']:.4f}, "
                  f"balanced={m['balanced_accuracy']:.4f}, "
                  f"majority={m['majority_accuracy']:.4f}, "
                  f"chance={m['uniform_chance']:.4f}")
    first = next(iter(res['probe_metrics'].values()))
    res['majority_accuracy'] = first['majority_accuracy']
    res['uniform_chance'] = first['uniform_chance']
    res['identity_conditioned_warning'] = (
        'per-cow-z uses the true cow label to select reference statistics; it is an '
        'identity-conditioned control, not an identity-blind deployment feature')
    res['protocol'] = ('largest session as reference, remaining sessions as probe; all '
                       'fitting on reference; balanced accuracy and majority baseline reported')
    import json
    json.dump(res, open(os.path.join(CACHE, 'session_order.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
