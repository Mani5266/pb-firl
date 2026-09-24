"""ReCowGnition loader: 6838 dairy-cow faces, 161 identities, 5 sessions.
Filename: [session]_[cowID]_[a]_[b].jpg. No pain labels (identity-only)."""
import glob
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'data', 'ReCowGnition - Dataset', 'ReCowGnition')
CACHE = os.path.join(ROOT, 'runs', 'features_cache')


def main():
    files = sorted(glob.glob(os.path.join(SRC, '**', '*.jpg'), recursive=True))
    rows = []
    for p in files:
        b = os.path.basename(p).rsplit('.', 1)[0].split('_')
        rows.append({'file': p, 'session': b[0], 'cow': b[1]})
    df = pd.DataFrame(rows)
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(os.path.join(CACHE, 'manifest_recow.parquet'))
    print(f'n={len(df)} cows={df.cow.nunique()} sessions={sorted(df.session.unique())}')
    print('per-cow:', df.cow.value_counts().describe()[['min', '50%', 'max']].to_dict())


if __name__ == '__main__':
    main()
