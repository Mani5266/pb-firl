"""Step 4b: GT-vs-predicted deviation battery per cow.
Max per-cow budgets, shift sweep {0.75, 1.0, 1.5, 3.0}, 3 seeds, per-cow intervals.
No pooled number."""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from src.front_end.geometry import roi_features
from src.front_end.train_landmarks import build, collate
from src.front_end.loco_landmarks import FlexKpt, COWS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints', 'loco')
SEEDS = [42, 123, 7]
SHIFTS = [0.75, 1.0, 1.5, 3.0]
NDIM = 8


def predict_roi(m, dl, dev):
    m.eval()
    rois = []
    with torch.no_grad():
        for imgs, tgts in dl:
            out = m([imgs[0].to(dev)])[0]
            if not len(out['boxes']):
                rois.append(np.zeros(12))
                continue
            j = int(out['scores'].argmax())
            p = out['keypoints'][j].cpu().numpy()[:, :2]
            rois.append(roi_features(p))
    return np.array(rois)


def gt_roi(dl):
    rois = []
    for i in range(len(dl.dataset)):
        r = dl.dataset.df.iloc[i]
        k = np.array([list(p) for p in r.kpts], dtype=np.float32)
        rois.append(roi_features(k))
    return np.array(rois)


def md(loc, inv, Z):
    D = Z - loc
    return np.sqrt((D @ inv * D).sum(1))


def run_battery(Z, rng):
    ix = np.arange(len(Z))
    rng.shuffle(ix)
    n = len(ix)
    nref = n // 2
    ref, evl = ix[:nref], ix[nref:]
    mu, sd = Z[ref].mean(0), Z[ref].std(0) + 1e-9
    Zs = (Z - mu) / sd
    Zr, Ze = Zs[ref], Zs[evl]
    dim = rng.choice(Z.shape[1], min(NDIM, Z.shape[1]), replace=False)
    res = {}
    for shift in SHIFTS:
        Zi = Ze.copy()
        Zi[:, dim] += shift
        lab = np.r_[np.zeros(len(Ze)), np.ones(len(Zi))]
        lw_s = LedoitWolf().fit(Zr)
        inv_s = lw_s.precision_
        d, di = md(lw_s.location_, inv_s, Ze), md(lw_s.location_, inv_s, Zi)
        z_mu, z_sd = d.mean(), d.std() + 1e-9
        res[f'shift_{shift}'] = roc_auc_score(
            lab, np.r_[(d - z_mu) / z_sd, (di - z_mu) / z_sd])
    return res


def main():
    assert torch.cuda.is_available(), 'GPU required'
    dev = torch.device('cuda')
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    roi_pred_all = {}
    roi_gt_all = {}
    for fi, heldout in enumerate(COWS, 1):
        import glob
        ck_files = glob.glob(os.path.join(CKPT, f'fold{fi}_*.pth'))
        assert len(ck_files) == 1
        m = build(512).to(dev)
        m.load_state_dict(torch.load(ck_files[0], map_location=dev)['model'])
        dl = DataLoader(FlexKpt([heldout]), batch_size=1, shuffle=False, collate_fn=collate)
        roi_pred_all[heldout] = predict_roi(m, dl, dev)
        roi_gt_all[heldout] = gt_roi(dl)
        print(f'cow {heldout}: {len(roi_pred_all[heldout])} frames', flush=True)
        del m
        torch.cuda.empty_cache()
    results = {}
    for heldout in COWS:
        Z_pred = roi_pred_all[heldout]
        Z_gt = roi_gt_all[heldout]
        gt_runs, pred_runs = [], []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            gt_runs.append(run_battery(Z_gt, rng))
            rng = np.random.RandomState(seed)
            pred_runs.append(run_battery(Z_pred, rng))
        cow_res = {'gt': {}, 'pred': {}}
        for shift in SHIFTS:
            gt_vals = [r[f'shift_{shift}'] for r in gt_runs]
            pred_vals = [r[f'shift_{shift}'] for r in pred_runs]
            cow_res['gt'][f'shift_{shift}'] = {
                'mean': round(float(np.mean(gt_vals)), 4),
                'ci95': [round(float(np.percentile(gt_vals, 2.5)), 4),
                         round(float(np.percentile(gt_vals, 97.5)), 4)]}
            cow_res['pred'][f'shift_{shift}'] = {
                'mean': round(float(np.mean(pred_vals)), 4),
                'ci95': [round(float(np.percentile(pred_vals, 2.5)), 4),
                         round(float(np.percentile(pred_vals, 97.5)), 4)]}
        results[heldout] = cow_res
        print(f'cow {heldout}: gt={cow_res["gt"]} pred={cow_res["pred"]}', flush=True)
    out = {'per_cow': results, 'seeds': SEEDS, 'shifts': SHIFTS,
           'protocol': 'LOCO checkpoints, ROI-12D, max per-cow budgets (half ref/half eval), shift sweep, 3 seeds'}
    path = os.path.join(CACHE, 'gt_vs_predicted.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
