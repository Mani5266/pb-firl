"""Round 2: resume Keypoint R-CNN from best with scale jitter + lower LR.
Best selected by val box-hit (same proxy as round 1). OOM fallback included."""
import os
import time
import torch
from torch.utils.data import DataLoader

from .train_landmarks import build, CattleKpt, collate, seed_all, CFG

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(ROOT, 'runs', 'checkpoints')
LR, EPOCHS = 2e-5, 6


def run_once(batch, sizes):
    seed_all()
    dev = torch.device('cuda')
    m = build(sizes[0])
    m.transform.min_size = tuple(sizes)
    ck = torch.load(os.path.join(CKPT, 'krcnn_r50_13kpt_best.pth'), map_location=dev)
    m.load_state_dict(ck['model'])
    m.to(dev)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
    sc = torch.cuda.amp.GradScaler()
    tl = DataLoader(CattleKpt('train'), batch_size=batch, shuffle=True, collate_fn=collate, num_workers=0)
    vl = DataLoader(CattleKpt('val'), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    best = ck.get('val_boxhit', 0)
    for ep in range(EPOCHS):
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
        m.eval()
        hit, n = 0, 0
        with torch.no_grad():
            for imgs, tgts in vl:
                out = m([imgs[0].to(dev)])[0]
                if len(out['boxes']):
                    b, g = out['boxes'][0].cpu().numpy(), tgts[0]['boxes'][0].numpy()
                    ix1, iy1 = max(b[0], g[0]), max(b[1], g[1])
                    ix2, iy2 = min(b[2], g[2]), min(b[3], g[3])
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    union = (b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter
                    hit += inter / (union + 1e-9) > 0.5
                n += 1
        score = hit / max(n, 1)
        print(f'r2-ep{ep} loss={tot/len(tl):.3f} val_boxhit={score:.3f} t={time.time()-t0:.0f}s', flush=True)
        if score > best:
            best = score
            torch.save({'model': m.state_dict(), 'cfg': {**CFG, 'r2': True},
                        'ep': ep, 'val_boxhit': score},
                       os.path.join(CKPT, 'krcnn_r50_13kpt_best.pth'))
    print('R2 BEST', best)


def main():
    seed_all()
    try:
        run_once(2, [416, 512, 576])
    except torch.cuda.OutOfMemoryError:
        print('OOM -> fallback batch=1 sizes=[416,512]', flush=True)
        torch.cuda.empty_cache()
        run_once(1, [416, 512])


if __name__ == '__main__':
    main()
