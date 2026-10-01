"""Build a reproducibility manifest and the consolidated evaluation report.

The builder fails loudly when a primary artifact is missing.  A report with
silently inserted ``null`` values is not a reproducible result, and legacy
pain runs are kept under an explicit ``legacy_results`` key rather than being
mixed with the primary matched evaluation.
"""

import hashlib
import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'runs', 'features_cache')
SEED = 42
SRC = [
    'src/front_end/geometry.py', 'src/front_end/ingest.py',
    'src/front_end/train_landmarks.py', 'src/front_end/eval_landmarks.py',
    'src/eval/variance_decomp.py', 'src/baseline/gaussian.py',
    'src/pain/train_sheep.py', 'src/pain/improve_mil.py',
    'src/pain/fair_compare.py', 'src/eval/sheep_stats.py',
    'src/eval/stats.py', 'src/eval/dairy.py', 'src/eval/dairy_qc.py',
    'src/eval/coldstart.py', 'src/eval/session_order.py',
    'src/eval/falsify.py', 'src/cusum/cusum.py', 'src/audit/leakage.py',
    'src/eval/lodo.py', 'src/eval/build_report.py',
]
PRIMARY_RESULTS = [
    'landmark_ap', 'variance', 'baseline_loio', 'pain_fair', 'sheep_stats',
    'dairy', 'session_order', 'coldstart', 'falsify', 'cusum',
    'identity_audit', 'ablation',
]
LEGACY_RESULTS = ['pain_sheep_g5', 'pain_mil_v2']
INPUTS = [
    'requirements.txt', 'configs/phase0.yaml',
    'runs/features_cache/manifest_rgb.parquet',
    'runs/features_cache/manifest_thermal.parquet',
    'runs/features_cache/manifest_recow.parquet',
    'runs/features_cache/dairy_emb.npz',
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    missing_sources = [p for p in SRC if not os.path.exists(os.path.join(ROOT, p))]
    if missing_sources:
        raise FileNotFoundError('missing source files: ' + ', '.join(missing_sources))
    missing_results = [name for name in PRIMARY_RESULTS
                       if not os.path.exists(os.path.join(CACHE, name + '.json'))]
    if missing_results:
        raise FileNotFoundError(
            'missing primary result artifacts; run the ordered evaluation first: '
            + ', '.join(missing_results))

    h = hashlib.sha256()
    source_hashes = {}
    for rel in SRC:
        path = os.path.join(ROOT, rel)
        digest = sha256_file(path)
        source_hashes[rel] = digest
        h.update(rel.encode('utf-8'))
        h.update(bytes.fromhex(digest))

    input_hashes = {}
    for rel in INPUTS:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            input_hashes[rel] = sha256_file(path)

    rep = {
        'seed': SEED,
        'config_hash': h.hexdigest()[:16],
        'source_hashes': source_hashes,
        'input_hashes': input_hashes,
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
        },
        'splits': {
            'cattle_rgb': {'train': ['1', '25', '50'], 'val': ['17'], 'test': ['64', '2']},
            'sheep': 'mirror train_raw(898)/test(74)/test_raw(225)',
        },
        'models': {
            'mil': 'matched cells in pain_fair.json; legacy val-selected runs are under legacy_results',
            'landmarks': 'krcnn R50-FPN R2: jitter(416,512,576) lr2e-5 x6ep (box-score top-1)',
        },
        'primary_protocol': (
            'Results are synthetic-shift, identity, proxy-label, or quality-shift '
            'evaluations as documented per artifact; no real cattle pain/change label '
            'is present in this public-data run.'),
    }
    for name in PRIMARY_RESULTS:
        with open(os.path.join(CACHE, name + '.json'), encoding='utf-8') as f:
            rep[name] = json.load(f)
    rep['legacy_results'] = {}
    for name in LEGACY_RESULTS:
        path = os.path.join(CACHE, name + '.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                rep['legacy_results'][name] = json.load(f)

    out = os.path.join(ROOT, 'reports', 'eval_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=1)
    print('wrote', out, 'primary artifacts:', len(PRIMARY_RESULTS),
          'legacy artifacts:', len(rep['legacy_results']))


if __name__ == '__main__':
    main()
