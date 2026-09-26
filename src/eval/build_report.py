"""Phase 8: aggregate every cached result into reports/eval_report.json with config hash + seed."""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42
SRC = ['src/front_end/geometry.py', 'src/front_end/ingest.py', 'src/front_end/train_landmarks.py',
       'src/front_end/eval_landmarks.py', 'src/eval/variance_decomp.py', 'src/baseline/gaussian.py',
       'src/pain/train_sheep.py', 'src/pain/improve_mil.py', 'src/pain/fair_compare.py',
       'src/eval/dairy.py', 'src/eval/dairy_qc.py',
       'src/cusum/cusum.py', 'src/audit/leakage.py', 'src/eval/lodo.py']


def main():
    h = hashlib.sha256()
    for s in SRC:
        h.update(open(os.path.join(ROOT, s), 'rb').read())
    rep = {'seed': SEED, 'config_hash': h.hexdigest()[:16],
           'splits': {'cattle_rgb': {'train': ['1', '25', '50'], 'val': ['17'], 'test': ['64']},
                       'sheep': 'mirror train_raw(898)/test(74)/test_raw(225)'},
           'models': {'mil': 'matched cells in pain_fair.json (shared split, BCE-only); legacy val-selected -> sheep_mil_best.pth',
                      'landmarks': 'krcnn R50-FPN R2: jitter(416,512,576) lr2e-5 x6ep (box-score top-1)'}}
    for name in ['landmark_ap', 'variance', 'baseline_loio', 'pain_sheep_g5', 'pain_mil_v2', 'pain_fair',
                 'dairy', 'cusum', 'identity_audit', 'ablation']:
        p = os.path.join(CACHE, name + '.json')
        rep[name] = json.load(open(p)) if os.path.exists(p) else None
    if os.path.exists(os.path.join(CACHE, 'pain_sheep.json')) and rep.get('pain_sheep_g5') is None:
        rep['pain_sheep_g5'] = json.load(open(os.path.join(CACHE, 'pain_sheep.json')))
    out = os.path.join(ROOT, 'reports', 'eval_report.json')
    json.dump(rep, open(out, 'w'), indent=1)
    missing = [k for k, v in rep.items() if v is None]
    print('wrote', out, 'missing:', missing)


if __name__ == '__main__':
    main()
