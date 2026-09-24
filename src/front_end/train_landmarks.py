"""Phase 1: fine-tune Keypoint R-CNN R50-FPN (13 cattle kpts) on identity-aware split.
Frozen backbone body; box+keypoint heads trained. AMP, AdamW, seeds fixed."""
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.transforms.functional import to_tensor

from .geometry import as_kpts

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
CKPT = os.path.join(ROOT, 'runs', 'checkpoints')
SEED = 42
CFG = {'seed': SEED, 'model': 'keypointrcnn_r50_fpn_13kpt', 'epochs': 8,
       'lr': 1e-4, 'batch': 2, 'min_size': 512, 'frozen': 'backbone.body',
       'split': {'train': ['1', '25', '50'], 'val': ['17'], 'test': ['64']}}


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class CattleKpt(Dataset):
    def __init__(self, split):
        df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
        self.df = df[(df.split == split) & df.exists].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = to_tensor(Image.open(r.file).convert('RGB'))
        x, y, w, h = r.bbox
        box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        k = as_kpts(r.kpts)
        vis = np.asarray(r.vis)
        kv = np.zeros((13, 3), np.float32)
        kv[:, :2] = k
        kv[:, 2] = np.where(vis > 0, 2, 0)
        tgt = {'boxes': box, 'labels': torch.ones(1, dtype=torch.int64),
               'keypoints': torch.from_numpy(kv).unsqueeze(0),
               'image_id': torch.tensor([r.id])}
        return img, tgt


def collate(b):
    return [x[0] for x in b], [x[1] for x in b]


def build(min_size):
    m = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.keypoint_predictor = KeypointRCNNPredictor(m.roi_heads.keypoint_predictor.kps_score_lowres.in_channels, 13)
    for p in m.backbone.body.parameters():
        p.requires_grad = False
    m.transform.min_size = (min_size,)
    m.transform.max_size = 1024
    return m


def run_once(batch, min_size):
    seed_all()
    dev = torch.device('cuda')
    m = build(min_size).to(dev)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=CFG['lr'])
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.1)
    sc = torch.cuda.amp.GradScaler()
    tl = DataLoader(CattleKpt('train'), batch_size=batch, shuffle=True, collate_fn=collate, num_workers=0)
    vl = DataLoader(CattleKpt('val'), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    best, best_ep = -1, -1
    os.makedirs(CKPT, exist_ok=True)
    for ep in range(CFG['epochs']):
        m.train()
        t0, tot = time.time(), 0.0
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
        # quick val: top-1 box IoU>0.5 rate as proxy (full OKS in eval script)
        m.eval()
        hit, n = 0, 0
        with torch.no_grad():
            for imgs, tgts in vl:
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
        score = hit / max(n, 1)
        print(f'ep{ep} loss={tot/len(tl):.3f} val_boxhit={score:.3f} t={time.time()-t0:.0f}s', flush=True)
        if score > best:
            best, best_ep = score, ep
            torch.save({'model': m.state_dict(), 'cfg': CFG, 'ep': ep, 'val_boxhit': score},
                       os.path.join(CKPT, 'krcnn_r50_13kpt_best.pth'))
    print('BEST', best, 'ep', best_ep, f'batch={batch} min={min_size}')


def main():
    seed_all()
    assert torch.cuda.is_available()
    try:
        run_once(CFG['batch'], CFG['min_size'])
    except torch.cuda.OutOfMemoryError:
        print('OOM at batch=2/min=512 -> fallback batch=1/min=416', flush=True)
        torch.cuda.empty_cache()
        run_once(1, 416)


if __name__ == '__main__':
    main()
