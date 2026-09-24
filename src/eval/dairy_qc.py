"""Bonus QC: beef-trained KRCNN on dairy 112px faces (no GT -> visual plausibility only)."""
import os
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor
from src.front_end.train_landmarks import build

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    dev = torch.device('cuda')
    m = build(512)
    m.load_state_dict(torch.load(os.path.join(ROOT, 'runs', 'checkpoints',
                                              'krcnn_r50_13kpt_best.pth'), map_location=dev)['model'])
    m.eval().to(dev)
    df = pd.read_parquet(os.path.join(ROOT, 'runs', 'features_cache', 'manifest_recow.parquet'))
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    det = 0
    for a, (_, r) in zip(ax.ravel(), df.sample(8, random_state=0).iterrows()):
        img = Image.open(r.file).convert('RGB')
        with torch.no_grad():
            out = m([to_tensor(img).to(dev)])[0]
        a.imshow(img.resize((224, 224)))
        if len(out['boxes']):
            det += 1
            p = out['keypoints'][int(out['scores'].argmax())].cpu().numpy()
            a.scatter(p[:, 0] * 2, p[:, 1] * 2, s=6, c='red')
        a.set_title(f"cow={r['cow']} boxes={len(out['boxes'])}")
        a.axis('off')
    fig.suptitle('KRCNN-beef on dairy 112px faces (red = pred, no GT)')
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'reports', 'figures', 'dairy_landmark_qc.png'), dpi=80)
    print('detected fraction:', det, '/8')


if __name__ == '__main__':
    main()
