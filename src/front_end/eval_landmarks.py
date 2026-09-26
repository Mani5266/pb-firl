"""Phase 1 eval: OKS success-rate + PCK on held-out cow(s); QC montage; cache preds.
PB_SKIP_GPU=1 recomputes metrics from cached preds (no forward pass)."""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .geometry import as_kpts, oks, oks_map, box_area, KPT_NAMES
from .train_landmarks import build, CattleKpt, collate, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints')
FIG = os.path.join(ROOT, 'reports', 'figures')


def predict_split(split, min_size):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = build(min_size)
    ck = torch.load(os.path.join(CKPT, 'krcnn_r50_13kpt_best.pth'), map_location=dev)
    m.load_state_dict(ck['model'])
    m.eval().to(dev)
    ds = CattleKpt(split)
    rec, preds, pcks = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, tgt = ds[i]
            out = m([img.to(dev)])[0]
            g = tgt['keypoints'][0].numpy()
            gv = g[:, 2] > 0
            ga = float(ds.df.iloc[i].area)  # COCO annotation area (was: x2*y2 bug)
            if not len(out['boxes']):
                rec.append((0.0, 0.0))
                preds.append(np.zeros((13, 2)))
                pcks.append(0.0)
                continue
            j = int(out['scores'].argmax())
            p = out['keypoints'][j].cpu().numpy()  # 13x3 heatmap-decoded
            o = oks(p[:, :2], g[:, :2], gv, ga)
            rec.append((o, float(out['scores'][j])))
            preds.append(p[:, :2])
            d = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
            scale = np.sqrt(ga)
            pcks.append(float(((d[gv] / scale) < 0.1).mean()))
    return ds, rec, np.array(preds), float(np.mean(pcks))


def metrics_from_preds(split):
    """Recompute OKS/PCK from cached preds (CPU). Returns (out_dict)."""
    ds = CattleKpt(split)
    preds = np.load(os.path.join(CACHE, f'kpts_pred_{split}.npy'))
    rec, pcks = [], []
    for i in range(len(ds)):
        _, tgt = ds[i]
        g = tgt['keypoints'][0].numpy()
        gv = g[:, 2] > 0
        ga = float(ds.df.iloc[i].area)
        p = preds[i]
        if not np.abs(p).sum():
            rec.append((0.0, 0.0))
            pcks.append(0.0)
            continue
        rec.append((oks(p, g[:, :2], gv, ga), 1.0))
        d = np.linalg.norm(p - g[:, :2], axis=1)
        pcks.append(float(((d[gv] / np.sqrt(ga)) < 0.1).mean()))
    m = oks_map(rec)
    return {**m, 'PCK@0.1': float(np.mean(pcks)), 'n': len(ds)}


def main():
    os.makedirs(FIG, exist_ok=True)
    out = {}
    skip = os.environ.get('PB_SKIP_GPU') == '1'
    for split in ['val', 'test']:
        if skip:
            o = metrics_from_preds(split)
        else:
            ds, rec, preds, pck = predict_split(split, 512)
            np.save(os.path.join(CACHE, f'kpts_pred_{split}.npy'), preds)
            o = {**oks_map(rec), 'PCK@0.1': pck, 'n': len(ds)}
        out[split] = o
        print(split, {k: round(v, 4) if isinstance(v, float) else v for k, v in o.items() if k != 'per_thr'})
    # QC montage: 4 test samples, GT vs pred
    ds = CattleKpt('test')
    preds = np.load(os.path.join(CACHE, 'kpts_pred_test.npy'))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    idx = np.linspace(0, len(ds) - 1, 4, dtype=int)
    for a, i in zip(ax.ravel(), idx):
        img, tgt = ds[i]
        g = as_kpts(tgt['keypoints'][0][:, :2])
        a.imshow(np.asarray(to_tensor(Image.open(ds.df.iloc[i].file).convert('RGB')).permute(1, 2, 0)))
        a.scatter(g[:, 0], g[:, 1], s=8, c='lime', label='GT')
        a.scatter(preds[i][:, 0], preds[i][:, 1], s=8, c='red', marker='x', label='pred')
        a.set_title(f"test id={int(tgt['image_id'])} cow={ds.df.iloc[i].cow}")
        a.axis('off')
    ax.ravel()[0].legend(markerscale=2)
    fig.suptitle('Landmark QC: GT (green) vs Keypoint R-CNN (red x)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'landmark_qc.png'), dpi=100)
    print('saved landmark_qc.png')
    import json
    json.dump(out, open(os.path.join(CACHE, 'landmark_ap.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
