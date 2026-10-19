"""Step 4: GT-vs-predicted deviation battery per cow.
Uses LOCO checkpoints to predict keypoints, computes ROI features, runs deviation battery.
Reports per-cow GT-vs-predicted gap. No pooled number."""
import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from src.front_end.geometry import roi_features, mahalanobis_cov
from src.front_end.train_landmarks import build, collate
from src.front_end.loco_landmarks import FlexKpt, COWS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints', 'loco')
SEED = 42
NREF, NEVL, NDIM = 20, 20, 8
SHIFTS = [1.5, 3.0]


def predict_roi(m, dl, dev):
    """Predict keypoints, compute ROI features per image."""
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
    """Compute ROI features from GT keypoints."""
    rois = []
    for i in range(len(dl.dataset)):
        r = dl.dataset.df.iloc[i]
        k = np.array([list(p) for p in r.kpts], dtype=np.float32)
        rois.append(roi_features(k))
    return np.array(rois)


def md(loc, inv, Z):
    D = Z - loc
    return np.sqrt((D @ inv * D).sum(1))


def run_battery(Z, cow, rng):
    """Run one cow's independent battery using its own 20+20 frames."""
    ix = np.arange(len(Z))
    if len(ix) < NREF + NEVL:
        return None
    rng.shuffle(ix)
    ref, evl = ix[:NREF], ix[NREF:NREF + NEVL]
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
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
        res[f'shift_{shift}_own'] = roc_auc_score(
            lab, np.r_[(d - d.mean()) / (d.std() + 1e-9), (di - di.mean()) / (di.std() + 1e-9)])
    return res


def main():
    assert torch.cuda.is_available(), 'GPU required'
    dev = torch.device('cuda')
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    df = df[df.exists].reset_index(drop=True)
    y = df.cow.values
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
        print(f'cow {heldout}: predicted {len(roi_pred_all[heldout])} frames', flush=True)
        del m
        torch.cuda.empty_cache()
    n = len(df)
    Z_pred = np.zeros((n, 12))
    Z_gt = np.zeros((n, 12))
    for heldout in COWS:
        mask = df.cow.values == heldout
        Z_pred[mask] = roi_pred_all[heldout]
        Z_gt[mask] = roi_gt_all[heldout]
    results = {}
    for heldout in COWS:
        rng = np.random.RandomState(SEED)
        res_pred = run_battery(Z_pred[df.cow.values == heldout], heldout, rng)
        rng = np.random.RandomState(SEED)
        res_gt = run_battery(Z_gt[df.cow.values == heldout], heldout, rng)
        results[heldout] = {'gt': res_gt, 'pred': res_pred}
        print(f'cow {heldout}: gt={res_gt} pred={res_pred}', flush=True)
    out = {'per_cow': results, 'protocol': 'LOCO checkpoints, ROI-12D, matched 20-frame budgets, synthetic shifts 1.5/3.0'}
    path = os.path.join(CACHE, 'gt_vs_predicted.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
