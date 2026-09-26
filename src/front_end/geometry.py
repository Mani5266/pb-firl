"""Geometry front-end: Procrustes alignment, ROI descriptors, OKS metrics. Numpy-only."""
import numpy as np

KPT_NAMES = ['left_ear_base', 'left_ear_middle', 'left_ear_tip', 'poll',
             'right_ear_base', 'right_ear_middle', 'right_ear_tip',
             'left_eye', 'right_eye', 'muzzle', 'left_nostril',
             'right_nostril', 'mouth']
I = {n: i for i, n in enumerate(KPT_NAMES)}


def as_kpts(k):
    """Coerce parquet/COCO nested keypoints -> (13,2) float array."""
    return np.stack([np.asarray(p, float).ravel()[:2] for p in k]).astype(float)


def umeyama(src, dst):
    """Similarity transform src->dst (no reflection). Returns s, R(2x2), t."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = ((src - mu_s).T @ (dst - mu_d)) / len(src)
    U, D, Vt = np.linalg.svd(S)
    # R maps column vectors: dst ~= s*R*src + t  =>  R = V*diag*U' (Umeyama)
    R = Vt.T @ np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    var = ((src - mu_s) ** 2).sum() / len(src)
    s = (D * np.array([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])).sum() / (var + 1e-12)
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def align(src, template):
    s, R, t = umeyama(src, template)
    return s * (src @ R.T) + t


def canonical_template(kpts, iters=2):
    """kpts: (N,13,2). Iterative mean-shape template."""
    kpts = np.asarray(kpts, float)
    assert kpts.ndim == 3 and kpts.shape[1:] == (13, 2), kpts.shape
    t = np.median(kpts, axis=0)
    for _ in range(iters):
        aligned = np.stack([align(k, t) for k in kpts])
        t = np.median(aligned, axis=0)
    return t


def _ang(a, b, c):
    """Angle at b (radians) between ba and bc."""
    u, v = a - b, c - b
    n = np.linalg.norm(u) * np.linalg.norm(v) + 1e-12
    return float(np.arccos(np.clip(u @ v / n, -1, 1)))


def roi_features(k, area=None):
    """13x2 keypoints -> 12-D scale-normalized descriptor vector."""
    k = np.asarray(k, float)
    d = np.linalg.norm(k[I['poll']] - k[I['muzzle']]) + 1e-6  # face scale
    f = [
        np.linalg.norm(k[I['left_eye']] - k[I['right_eye']]) / d,
        np.linalg.norm(k[I['left_ear_base']] - k[I['left_ear_tip']]) / d,
        np.linalg.norm(k[I['right_ear_base']] - k[I['right_ear_tip']]) / d,
        _ang(k[I['left_ear_base']], k[I['left_ear_middle']], k[I['left_ear_tip']]),
        _ang(k[I['right_ear_base']], k[I['right_ear_middle']], k[I['right_ear_tip']]),
        np.linalg.norm(k[I['left_nostril']] - k[I['right_nostril']]) / d,
        np.linalg.norm(k[I['muzzle']] - k[I['left_nostril']]) / d,
        np.linalg.norm(k[I['muzzle']] - k[I['right_nostril']]) / d,
        np.linalg.norm(k[I['muzzle']] - k[I['mouth']]) / d,
        np.linalg.norm(k[I['left_eye']] - k[I['muzzle']]) / d,
        np.linalg.norm(k[I['right_eye']] - k[I['muzzle']]) / d,
        np.linalg.norm(k[I['left_eye']] - k[I['left_nostril']]) / d +
        np.linalg.norm(k[I['right_eye']] - k[I['right_nostril']]) / d,
    ]
    return np.array(f, float)


FEAT_NAMES = ['inter_eye', 'ear_len_L', 'ear_len_R', 'ear_bend_L', 'ear_bend_R',
              'nostril_dist', 'muzzle_nostril_L', 'muzzle_nostril_R',
              'muzzle_mouth', 'eye_muzzle_L', 'eye_muzzle_R', 'eye_nostril_sum']


def oks(pred, gt, vis, area, sigma=0.10):
    """Object Keypoint Similarity, single instance. sigma uniform (documented)."""
    d2 = ((pred - gt) ** 2).sum(-1)
    e = np.exp(-d2 / (2 * (area + 1e-9) * sigma ** 2))
    v = vis > 0
    return float(e[v].mean()) if v.any() else 0.0


def box_area(b):
    """Area of [x1,y1,x2,y2] box. (x2*y2 is the classic origin-product bug.)"""
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def oks_map(records):
    """Mean OKS success rate over thresholds (NOT COCO AP: single pred/gt, no score ranking)."""
    thrs = np.arange(0.5, 1.0, 0.05)
    s = {}
    for t in thrs:
        s[f'success@{round(float(t), 2):.2f}'] = float(np.mean([o > t for o, _ in records])) if records else 0.0
    vals = list(s.values())
    return {'mean_success': float(np.mean(vals)), 'success@0.50': s['success@0.50'],
            'success@0.75': s['success@0.75'], 'per_thr': s}
