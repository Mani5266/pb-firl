"""Phase 4: sheep EXPRESSION-PROXY module (NOT a cattle-pain model).

Frozen ResNet-50 embeddings; B0 pooled-MLP (hard labels) vs attention-MIL over
patch instances. Labels are facial-expression proxies, not clinical pain diagnoses.
The balanced `test` split overlaps train (see data/README.md): it is reported for
direction only and must NEVER be used for model selection. Primary eval: test_raw.
"""
import io
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHEEP = os.path.join(ROOT, 'data', 'sheep_raw', 'data')
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints')
FIG = os.path.join(ROOT, 'reports', 'figures')
SEED = 42
GRID = 5  # ponytail: 3x3 tried first (MIL auroc 0.834/0.769); 5x5 = finer instances
FILES = {'train': 'train-00000-of-00001.parquet', 'test': 'test-00000-of-00001.parquet',
         'train_raw': 'train_raw-00000-of-00001.parquet', 'test_raw': 'test_raw-00000-of-00001.parquet'}


def seed_all():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def embed_all():
    seed_all()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = resnet50(weights=ResNet50_Weights.DEFAULT)
    feat = nn.Sequential(*list(m.children())[:-2]).eval().to(dev)
    tf = ResNet50_Weights.DEFAULT.transforms()
    os.makedirs(CACHE, exist_ok=True)
    for split, fn in FILES.items():
        out = os.path.join(CACHE, f'sheep_g{GRID}_{split}.npz')
        if os.path.exists(out):
            print(split, 'cached')
            continue
        df = pd.read_parquet(os.path.join(SHEEP, fn))
        P, I, Y = [], [], []
        with torch.no_grad():
            for i in range(0, len(df), 32):
                imgs = [tf(Image.open(io.BytesIO(r['bytes'])).convert('RGB'))
                        for r in df.image.iloc[i:i+32]]
                f = feat(torch.stack(imgs).to(dev))  # N,2048,7,7
                p = torch.nn.functional.adaptive_avg_pool2d(f, GRID)  # N,2048,G,G
                I.append(p.permute(0, 2, 3, 1).reshape(len(imgs), GRID * GRID, 2048).cpu().numpy())
                P.append(f.mean((2, 3)).cpu().numpy())
        I, P = np.concatenate(I), np.concatenate(P)
        Y = df.label.astype(int).values
        np.savez_compressed(out, pooled=P, inst=I, y=Y)
        print(split, P.shape, I.shape, 'pos_rate=', round(float(Y.mean()), 3))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttMIL(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.V = nn.Linear(2048, d)
        self.U = nn.Linear(2048, d)
        self.w = nn.Linear(d, 1)
        self.head = nn.Linear(2048, 1)

    def forward(self, h):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # N,9,1
        a = torch.softmax(a, 1)
        bag = (a * h).sum(1)
        return self.head(bag).squeeze(-1), a.squeeze(-1)


def metrics(y, s):
    from sklearn.metrics import recall_score, f1_score, roc_auc_score
    p = (s > 0.5).astype(int)
    return {'recall': round(float(recall_score(y, p, zero_division=0)), 4),
            'f1': round(float(f1_score(y, p, zero_division=0)), 4),
            'auroc': round(float(roc_auc_score(y, s)), 4)}


def boot_auroc(y, s, n=1000):
    rng = np.random.RandomState(SEED)
    v = [__import__('sklearn').metrics.roc_auc_score(y[b], s[b])
         for b in (rng.randint(0, len(y), len(y)) for _ in range(n))
         if len(np.unique(y[b])) == 2]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def fit(model, Xtr, ytr, Xva, yva, is_mil, epochs=60):
    seed_all()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.from_numpy(ytr * 0.9 + 0.05).float()  # label smoothing
    best, wait, bs = -1, 0, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xtr)[0] if is_mil else model(Xtr)
        lossf(out, yt).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(Xva)[0] if is_mil else model(Xva)).numpy()
        from sklearn.metrics import roc_auc_score
        try:
            a = roc_auc_score(yva, s)
        except ValueError:
            a = 0.5
        if a > best:
            best, wait = a, 0
            bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 10:
                break
    model.load_state_dict(bs)
    return model, round(float(best), 4)


def main():
    seed_all()
    embed_all()
    tr = np.load(os.path.join(CACHE, f'sheep_g{GRID}_train_raw.npz'))
    Xp, Xi, y = tr['pooled'], tr['inst'], tr['y']
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(y))
    nva = int(0.15 * len(y))
    va, trn = idx[:nva], idx[nva:]
    Xp, Xi, y = torch.from_numpy(Xp).float(), torch.from_numpy(Xi).float(), y
    res = {}
    b0 = fit(MLP(), Xp[trn], y[trn], Xp[va], y[va], False)[0]
    mil = fit(AttMIL(), Xi[trn], y[trn], Xi[va], y[va], True)[0]
    os.makedirs(CKPT, exist_ok=True)
    torch.save(b0.state_dict(), os.path.join(CKPT, 'sheep_b0.pth'))
    torch.save(mil.state_dict(), os.path.join(CKPT, f'sheep_mil_g{GRID}.pth'))
    with torch.no_grad():
        for split in ['test', 'test_raw']:
            if split == 'test' and os.environ.get('PB_ALLOW_CONTAMINATED') != '1':
                print('SKIP contaminated balanced test (60/74 dhash overlap with train); '
                      'set PB_ALLOW_CONTAMINATED=1 to score it anyway')
                continue
            d = np.load(os.path.join(CACHE, f'sheep_g{GRID}_{split}.npz'))
            Xt, Xti, yt = torch.from_numpy(d['pooled']).float(), torch.from_numpy(d['inst']).float(), d['y']
            s0 = torch.sigmoid(b0(Xt)).numpy()
            sm, A = mil(Xti)
            sm = torch.sigmoid(sm).numpy()
            res[f'B0_{split}'] = {**metrics(yt, s0), 'ci': boot_auroc(yt, s0), 'n': len(yt)}
            res[f'MIL_{split}'] = {**metrics(yt, sm), 'ci': boot_auroc(yt, sm), 'n': len(yt)}
            print(split, 'B0', res[f'B0_{split}'], 'MIL', res[f'MIL_{split}'])
            if split == 'test_raw':
                # attention figure uses the primary (uncontaminated) eval split
                df = pd.read_parquet(os.path.join(SHEEP, FILES['test_raw']))
                pos = np.where(yt == 1)[0][:2]
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                for a, j in zip(ax, pos):
                    img = Image.open(io.BytesIO(df.image.iloc[j]['bytes'])).convert('RGB').resize((224, 224))
                    a.imshow(img)
                    a.imshow(A[j].cpu().numpy().reshape(GRID, GRID), alpha=0.55, cmap='jet',
                             extent=(0, 224, 224, 0), interpolation='bilinear')
                    a.set_title(f'pain p={sm[j]:.2f}')
                    a.axis('off')
                fig.suptitle(f'MIL attention over {GRID}x{GRID} patches (sheep pain examples)')
                fig.tight_layout()
                fig.savefig(os.path.join(FIG, f'mil_attention_g{GRID}.png'), dpi=100)
    import json
    json.dump(res, open(os.path.join(CACHE, f'pain_sheep_g{GRID}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
