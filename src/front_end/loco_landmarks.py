"""LOCO landmark eval: 5-fold leave-one-cow-out OKS success@0.50.
Per fold: 3 train cows + 1 val cow (checkpoint selection) + 1 held-out cow.
Val cow = fewest images among non-held-out cows. Never tunes on held-out."""
import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from .geometry import oks, oks_map
from .train_landmarks import build, collate, seed_all

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints', 'loco')
SEED = 42
EPOCHS = 8
LR = 1e-4
BATCH = 2
MIN_SIZE = 512

COWS = ['1036', '1063', '1059', '1087', '1090']


class FlexKpt(torch.utils.data.Dataset):
    def __init__(self, cows):
        df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
        self.df = df[df.cow.isin(cows) & df.exists].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = to_tensor(Image.open(r.file).convert('RGB'))
        x, y, w, h = r.bbox
        box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        k = np.array([list(p) for p in r.kpts], dtype=np.float32)
        kv = np.zeros((13, 3), np.float32)
        kv[:, :2] = k
        kv[:, 2] = np.where(np.asarray(r.vis) > 0, 2, 0)
        tgt = {'boxes': box, 'labels': torch.ones(1, dtype=torch.int64),
               'keypoints': torch.from_numpy(kv).unsqueeze(0),
               'image_id': torch.tensor([r.id])}
        return img, tgt


def val_boxhit(m, dl, dev):
    m.eval()
    hit, n = 0, 0
    with torch.no_grad():
        for imgs, tgts in dl:
            out = m([imgs[0].to(dev)])[0]
            if len(out['boxes']):
                b = out['boxes'][0].cpu().numpy()
                g = tgts[0]['boxes'][0].numpy()
                ix1, iy1 = max(b[0], g[0]), max(b[1], g[1])
                ix2, iy2 = min(b[2], g[2]), min(b[3], g[3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                union = (b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter
                hit += inter / (union + 1e-9) > 0.5
            n += 1
    return hit / max(n, 1)


def eval_oks(m, dl, dev):
    m.eval()
    rec, pcks = [], []
    with torch.no_grad():
        for imgs, tgts in dl:
            out = m([imgs[0].to(dev)])[0]
            r = dl.dataset.df.iloc[len(rec)]
            g = np.array([list(p) for p in r.kpts], dtype=np.float32)
            gv = np.asarray(r.vis) > 0
            if not len(out['boxes']):
                rec.append((0.0, 0.0))
                pcks.append(0.0)
                continue
            j = int(out['scores'].argmax())
            p = out['keypoints'][j].cpu().numpy()
            rec.append((oks(p[:, :2], g[:, :2], gv, r.area), 1.0))
            d = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
            pcks.append(float(((d[gv] / np.sqrt(r.area)) < 0.1).mean()))
    return {**oks_map(rec), 'PCK@0.1': float(np.mean(pcks))}


def run_fold(heldout, train_cows, val_cow, fold_id):
    seed_all()
    dev = torch.device('cuda')
    m = build(MIN_SIZE).to(dev)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.1)
    sc = torch.cuda.amp.GradScaler()
    tl = DataLoader(FlexKpt(train_cows), batch_size=BATCH, shuffle=True,
                    collate_fn=collate, num_workers=0)
    vl = DataLoader(FlexKpt([val_cow]), batch_size=1, shuffle=False, collate_fn=collate)
    best, best_ep = -1, -1
    os.makedirs(CKPT, exist_ok=True)
    ck_path = os.path.join(CKPT, f'fold{fold_id}_heldout{heldout}.pth')
    t0 = time.time()
    for ep in range(EPOCHS):
        m.train()
        tot = 0.0
        for imgs, tgts in tl:
            imgs = [im.to(dev) for im in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = sum(v for v in m(imgs, tgts).values())
            sc.scale(loss).backward()
            sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 5.0)
            sc.step(opt)
            sc.update()
            tot += float(loss)
        sch.step()
        score = val_boxhit(m, vl, dev)
        print(f'  fold{fold_id} ep{ep} loss={tot/len(tl):.3f} valhit={score:.3f} '
              f't={time.time()-t0:.0f}s', flush=True)
        if score > best:
            best, best_ep = score, ep
            torch.save({'model': m.state_dict(), 'ep': ep, 'val_hit': score,
                        'heldout': heldout, 'train': train_cows, 'val': val_cow}, ck_path)
    return ck_path


def main():
    assert torch.cuda.is_available(), 'GPU required'
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    counts = df[df.exists].groupby('cow').size().to_dict()
    results = []
    for fi, heldout in enumerate(COWS, 1):
        others = [c for c in COWS if c != heldout]
        val_cow = min(others, key=lambda c: counts[c])
        train_cows = [c for c in others if c != val_cow]
        ck_path = os.path.join(CKPT, f'fold{fi}_heldout{heldout}.pth')
        print(f'Fold {fi}: heldout={heldout} train={train_cows} val={val_cow}', flush=True)
        if os.path.exists(ck_path):
            print(f'  checkpoint exists, skipping training', flush=True)
        else:
            ck_path = run_fold(heldout, train_cows, val_cow, fi)
        dev = torch.device('cuda')
        m = build(MIN_SIZE).to(dev)
        m.load_state_dict(torch.load(ck_path, map_location=dev)['model'])
        el = DataLoader(FlexKpt([heldout]), batch_size=1, shuffle=False, collate_fn=collate)
        met = eval_oks(m, el, dev)
        row = {'heldout': heldout, 'train': train_cows, 'val': val_cow,
               'n': len(el.dataset), 'mean_success': met['mean_success'],
               'success@0.50': met['success@0.50'], 'PCK@0.1': met['PCK@0.1']}
        results.append(row)
        print(f'  -> heldout {heldout}: success@0.50={met["success@0.50"]:.4f}', flush=True)
        del m
        torch.cuda.empty_cache()
    mean_s = float(np.mean([r['success@0.50'] for r in results]))
    out = {'folds': results, 'mean_success@0.50': mean_s,
           'val_cow_rule': 'fewest images among non-held-out cows'}
    path = os.path.join(CACHE, 'loco_results.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nMean LOCO success@0.50 = {mean_s:.4f}')
    print(f'saved {path}')


if __name__ == '__main__':
    main()
