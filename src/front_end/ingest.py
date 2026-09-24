"""Ingest CattleFace-RGBT annotations -> identity-aware manifest + cached GT features."""
import json
import os
import numpy as np
import pandas as pd
from .geometry import KPT_NAMES, roi_features, canonical_template, as_kpts

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN = os.path.join(ROOT, 'data', 'cattleface_raw', 'annotations')
IMGROOT = os.path.join(ROOT, 'data', 'cattleface')
CACHE = os.path.join(ROOT, 'runs', 'features_cache')

# ponytail: folder->cow via 02_13 sequence match (only date containing seqs 17/25/50/64/2).
# Assumption recorded in data/README + limitations; rgb lacks folder 2.
FOLDER_COW = {'1': '1090', '2': '1115', '17': '1063',
              '25': '1087', '50': '1036', '64': '1059'}

SPLIT = {'train': ['1', '25', '50'], 'val': ['17'], 'test': ['64', '2']}  # identity-aware


def load_coco(name):
    with open(os.path.join(ANN, name)) as f:
        return json.load(f)


def build_manifest(modality='rgb'):
    coco = load_coco('rgb_keypoints.json' if modality == 'rgb' else 'thermal_keypoints.json')
    anns = {a['image_id']: a for a in coco['annotations']}
    rows = []
    for im in coco['images']:
        a = anns.get(im['id'])
        if a is None:
            continue
        kp = np.array(a['keypoints'], float).reshape(-1, 3)
        folder = str(im.get('folder', im['file_name'].split('/')[1]))
        path = os.path.join(IMGROOT, im['file_name'])
        rows.append({
            'id': im['id'], 'file': path, 'exists': os.path.exists(path),
            'w': im['width'], 'h': im['height'], 'folder': folder,
            'cow': FOLDER_COW.get(folder), 'bbox': a['bbox'], 'area': a['area'],
            'kpts': kp[:, :2].tolist(), 'vis': kp[:, 2].astype(int).tolist(),
            'num_kp': a['num_keypoints'],
        })
    df = pd.DataFrame(rows)
    df['split'] = df['folder'].map({f: s for s, fs in SPLIT.items() for f in fs})
    return df


def main():
    os.makedirs(CACHE, exist_ok=True)
    for mod in ['rgb', 'thermal']:
        df = build_manifest(mod)
        print(f'{mod}: n={len(df)} exist={int(df.exists.sum())} '
              f'noid={int(df.cow.isna().sum())} splits={df.split.value_counts().to_dict()}')
        df.to_parquet(os.path.join(CACHE, f'manifest_{mod}.parquet'))
    # GT ROI features + template from RGB train, cache for phases 2-3
    df = pd.read_parquet(os.path.join(CACHE, 'manifest_rgb.parquet'))
    tr = df[(df.split == 'train') & df.exists]
    K = np.stack([as_kpts(k) for k in tr.kpts.values])
    tpl = canonical_template(K)
    np.save(os.path.join(CACHE, 'template_13.npy'), tpl)
    F = np.stack([roi_features(as_kpts(k), a) for k, a in zip(tr.kpts.values, tr.area.values)])
    np.save(os.path.join(CACHE, 'roi_train_gt.npy'), F)
    print('template + roi_train_gt cached:', tpl.shape, F.shape)


if __name__ == '__main__':
    main()
