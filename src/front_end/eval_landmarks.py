"""Phase 1 eval: OKS mAP/AP50/AP75 + PCK on held-out cow(s); QC montage; cache preds."""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .geometry import as_kpts, oks, oks_map, KPT_NAMES
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
            gv, ga = (g[:, 2] > 0), float(tgt['boxes'][0][2] * tgt['boxes'][0][3])
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


def main():
    os.makedirs(FIG, exist_ok=True)
    out = {}
    for split in ['val', 'test']:
        ds, rec, preds, pck = predict_split(split, 512)
        m = oks_map(rec)
        out[split] = {**m, 'PCK@0.1': pck, 'n': len(ds)}
        print(split, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[split].items() if k != 'per_thr'})
        np.save(os.path.join(CACHE, f'kpts_pred_{split}.npy'), preds)
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
