"""Landmark failure diagnosis: per-cow, per-keypoint, box-vs-kpts breakdown.
Uses cached predictions (no GPU). Commits findings as a report."""
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .geometry import oks, KPT_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
FIG = os.path.join(ROOT, 'reports', 'figures')


def per_image_oks(split):
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[(df.split == split) & df.exists].reset_index(drop=True)
    preds = np.load(os.path.join(CACHE, f'kpts_pred_{split}.npy'))
    rows = []
    for i in range(len(df)):
        r = df.iloc[i]
        g = np.array([list(p) for p in r.kpts], dtype=np.float32)
        vis = np.asarray(r.vis)
        gv = vis > 0
        p = preds[i]
        if not np.abs(p).sum():
            rows.append({'idx': i, 'cow': r.cow, 'folder': r.folder, 'oks': 0.0,
                         'box_w': r.bbox[2], 'box_h': r.bbox[3], 'area': r.area,
                         'pck': 0.0, 'n_vis': int(gv.sum())})
            continue
        o = oks(p[:, :2], g[:, :2], gv, r.area)
        d = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
        pck = float(((d[gv] / np.sqrt(r.area)) < 0.1).mean())
        rows.append({'idx': i, 'cow': r.cow, 'folder': r.folder, 'oks': o,
                     'box_w': r.bbox[2], 'box_h': r.bbox[3], 'area': r.area,
                     'pck': pck, 'n_vis': int(gv.sum())})
    return pd.DataFrame(rows), df, preds


def per_keypoint_error(split):
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[(df.split == split) & df.exists].reset_index(drop=True)
    preds = np.load(os.path.join(CACHE, f'kpts_pred_{split}.npy'))
    errs = {k: [] for k in KPT_NAMES}
    for i in range(len(df)):
        r = df.iloc[i]
        g = np.array([list(p) for p in r.kpts], dtype=np.float32)
        vis = np.asarray(r.vis)
        gv = vis > 0
        p = preds[i]
        if not np.abs(p).sum():
            continue
        d = np.linalg.norm(p[:, :2] - g[:, :2], axis=1)
        scale = np.sqrt(r.area)
        for j, name in enumerate(KPT_NAMES):
            if gv[j]:
                errs[name].append(d[j] / scale)
    return {k: {'mean': float(np.mean(v)), 'p90': float(np.percentile(v, 90)), 'n': len(v)}
            for k, v in errs.items()}


def cow_summary(split):
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[(df.split == split) & df.exists].reset_index(drop=True)
    rows, _, _ = per_image_oks(split)
    summary = []
    for cow, grp in rows.groupby('cow'):
        summary.append({
            'cow': str(cow), 'n': len(grp),
            'mean_oks': round(float(grp.oks.mean()), 4),
            'success@0.50': round(float((grp.oks > 0.5).mean()), 4),
            'mean_pck@0.1': round(float(grp.pck.mean()), 4),
            'mean_box_area': round(float(grp.area.mean()), 0),
            'mean_box_w': round(float(grp.box_w.mean()), 1),
            'mean_box_h': round(float(grp.box_h.mean()), 1),
            'mean_n_vis': round(float(grp.n_vis.mean()), 1),
        })
    return summary


def worst_cases(split, n=6):
    rows, df, preds = per_image_oks(split)
    worst = rows.nsmallest(n, 'oks')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (_, r) in zip(axes.ravel(), worst.iterrows()):
        i = int(r.idx)
        img = np.asarray(to_tensor(Image.open(df.iloc[i].file).convert('RGB')).permute(1, 2, 0))
        g = np.array([list(p) for p in df.iloc[i].kpts], dtype=np.float32)
        ax.imshow(img)
        ax.scatter(g[:, 0], g[:, 1], s=12, c='lime', label='GT')
        ax.scatter(preds[i][:, 0], preds[i][:, 1], s=12, c='red', marker='x', label='pred')
        ax.set_title(f"cow={r.cow} OKS={r.oks:.3f}", fontsize=9)
        ax.axis('off')
    axes.ravel()[0].legend(markerscale=2)
    fig.suptitle(f'Worst {n} predictions — {split} split')
    fig.tight_layout()
    out = os.path.join(FIG, f'landmark_worst_{split}.png')
    fig.savefig(out, dpi=100)
    print(f'saved {out}')


def main():
    os.makedirs(FIG, exist_ok=True)
    report = {}
    for split in ['val', 'test']:
        cows = cow_summary(split)
        kpts = per_keypoint_error(split)
        report[split] = {'per_cow': cows, 'per_keypoint': kpts}
        print(f'\n=== {split} ===')
        print('Per-cow:')
        for c in cows:
            print(f"  cow {c['cow']}: OKS={c['mean_oks']:.3f} success@0.50={c['success@0.50']:.3f} "
                  f"PCK@0.1={c['mean_pck@0.1']:.3f} n={c['n']} area={c['mean_box_area']:.0f}")
        print('Per-keypoint (normalized error):')
        for k, v in sorted(kpts.items(), key=lambda x: -x[1]['mean']):
            print(f"  {k:20s} mean={v['mean']:.4f} p90={v['p90']:.4f} n={v['n']}")
        worst_cases(split)
    out = os.path.join(CACHE, 'landmark_diagnosis.json')
    json.dump(report, open(out, 'w'), indent=1)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
