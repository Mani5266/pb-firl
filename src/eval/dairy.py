"""Dairy-scale personalisation validation (ReCowGnition, 161 Holstein cows, no pain labels).
Leakage-safe protocol (review fix): per-cow reference/probe split; PCA fit on reference only;
per-cow stats from reference only; probes (LDA + 5NN) trained on reference, scored on probe.
LOIO uses per-cow PCA-on-rest. Frozen ResNet-50 embeddings (cached; torch imported lazily)."""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from src.eval.stats import classification_summary
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED = 42


def embed_if_missing(df, out):
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision.models import resnet50, ResNet50_Weights
    torch.manual_seed(SEED)
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


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_recow.parquet'))
    out = os.path.join(CACHE, 'dairy_emb.npz')
    if not os.path.exists(out):
        embed_if_missing(df, out)
    E = np.load(out)['E']
    y = df.cow.values
    sess = df.session.values
    cows = np.unique(y)
    print('cows:', len(cows), 'chance:', round(1/len(cows), 4))
    # per-cow reference/probe split (singletons -> reference only, never scored)
    ref_idx, probe_idx = [], []
    for c in cows:
        ix = np.where(y == c)[0].copy()
        rng.shuffle(ix)
        h = max(1, len(ix) // 2)
        ref_idx.extend(ix[:h])
        probe_idx.extend(ix[h:])
    ref_idx, probe_idx = np.array(ref_idx), np.array(probe_idx)
    pca = PCA(n_components=64, random_state=SEED).fit(E[ref_idx])
    Pr, Pp = pca.transform(E[ref_idx]), pca.transform(E[probe_idx])
    yr, yp = y[ref_idx], y[probe_idx]
    # scatter share on probe
    mu = Pp.mean(0)
    sb = sum((yp == c).sum() * ((Pp[yp == c].mean(0) - mu) ** 2).sum() for c in np.unique(yp))
    between = float(sb / (((Pp - mu) ** 2).sum() + 1e-12))
    # per-cow stats from reference only
    ref_mean = {c: Pr[yr == c].mean(0) for c in cows}
    ref_std = {c: Pr[yr == c].std(0) + 1e-9 for c in cows}
    Zr = np.stack([(r - ref_mean[c]) / ref_std[c] for r, c in zip(Pr, yr)])
    Zp = np.stack([(r - ref_mean[c]) / ref_std[c] for r, c in zip(Pp, yp)])
    probes = {'LDA': LinearDiscriminantAnalysis(), '5NN': KNeighborsClassifier(5)}
    acc = {}
    probe_metrics = {}
    for pname, clf in probes.items():
        for rname, Xtr, Xte in [('raw', Pr, Pp), ('per-cow-z', Zr, Zp)]:
            clf.fit(Xtr, yr)
            pred = clf.predict(Xte)
            m = classification_summary(yp, pred, yr)
            key = f'{pname}_{rname}'
            acc[key] = m['accuracy']
            probe_metrics[key] = m
            print(f"{pname} {rname}: accuracy={m['accuracy']:.4f}, "
                  f"balanced={m['balanced_accuracy']:.4f}, "
                  f"majority={m['majority_accuracy']:.4f}, "
                  f"chance={m['uniform_chance']:.4f}")
    # LOIO injected-shift with per-cow PCA-on-rest (cows with >=10 imgs)
    big = [c for c in cows if (y == c).sum() >= 10]
    print('loio cows:', len(big))
    res_loio = {}
    for shift in [1.5, 3.0]:
        apc, apo = [], []
        dim = rng.choice(64, 8, replace=False)
        for c in big:
            idx = np.where(y == c)[0].copy()
            rng.shuffle(idx)
            h = idx[:len(idx)//2]
            rest = np.setdiff1d(np.arange(len(y)), idx)
            pc = PCA(n_components=64, random_state=SEED).fit(E[rest])
            R0, Cf, Ce = pc.transform(E[rest]), pc.transform(E[h]), \
                pc.transform(E[np.setdiff1d(idx, h)])
            mu_g = R0.mean(0)
            sd_g = R0.std(0) + 1e-9
            Zf, Ze = (Cf - mu_g)/sd_g, (Ce - mu_g)/sd_g
            lw_s, lw_p = LedoitWolf().fit(Zf), LedoitWolf().fit((R0-mu_g)/sd_g)
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
    # cross-session distances on probe embeddings
    Pn = Pp / (np.linalg.norm(Pp, axis=1, keepdims=True) + 1e-9)
    sp = sess[probe_idx]
    d_same, d_cross, d_diff = [], [], []
    by_cow = {}
    for j, c in enumerate(yp):
        by_cow.setdefault(c, []).append(j)
    for c, js in by_cow.items():
        if len(js) < 2:
            continue
        for _ in range(20):
            a, b = rng.choice(js, 2, replace=False)
            (d_same if sp[a] == sp[b] else d_cross).append(float(1 - Pn[a] @ Pn[b]))
    for _ in range(2000):
        a = rng.randint(len(yp))
        b = rng.randint(len(yp))
        if yp[a] != yp[b]:
            d_diff.append(float(1 - Pn[a] @ Pn[b]))
    first_metrics = next(iter(probe_metrics.values()))
    res = {'n_cows': len(cows), 'n': len(df), 'probe_n': len(yp), 'chance': round(1/len(cows), 4),
           'majority_accuracy': first_metrics['majority_accuracy'],
           'uniform_chance': first_metrics['uniform_chance'],
           'between': round(between, 4), **acc,
           'probe_metrics': probe_metrics,
           'identity_conditioned_controls': ['LDA_per-cow-z', '5NN_per-cow-z'],
           'identity_conditioned_warning': ('per-cow-z uses the true cow label to select '
                                             'reference statistics; it is not an identity-blind '
                                             'deployment feature'),
           'loio': res_loio, 'loio_cows': len(big),
           'dist': {'same_sess': round(float(np.mean(d_same)), 4),
                    'cross_sess': round(float(np.mean(d_cross)), 4),
                    'diff_cow': round(float(np.mean(d_diff)), 4)},
           'protocol': ('ref/probe split per cow; PCA+stats on ref; probes trained ref scored '
                        'probe; balanced accuracy and majority baseline reported for identity')}
    print(res['dist'])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(list(probe_metrics), [m['balanced_accuracy'] for m in probe_metrics.values()])
    plt.axhline(first_metrics['balanced_constant_baseline'], color='darkorange', ls='--',
                label='constant-class balanced baseline')
    plt.axhline(first_metrics['uniform_chance'], color='red', ls=':',
                label=f'uniform chance ({first_metrics["uniform_chance"]:.2f})')
    plt.xticks(rotation=15)
    plt.title(f'Dairy cow-ID probes (161 cows, balanced accuracy)')
    plt.legend(fontsize=8)
    plt.subplot(1, 2, 2)
    plt.bar(['same\nsession', 'cross\nsession', 'different\ncow'],
            [res['dist']['same_sess'], res['dist']['cross_sess'], res['dist']['diff_cow']])
    plt.title('Mean cosine distance, probe PCA-64')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'dairy_audit.png'), dpi=100)
    import json
    json.dump(res, open(os.path.join(CACHE, 'dairy.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
