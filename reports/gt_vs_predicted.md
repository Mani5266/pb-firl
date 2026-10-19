# GT-vs-Predicted Deviation Battery (2026-10-19, Step 4b)

## RETRACTION (2026-10-19)

The initial version of this report concluded "the battery is not sensitive at n=5 cows with 12-D geometric features." **That conclusion is retracted.** The experiment had a scoring bug: per-group standardisation subtracted the shift signal by construction, forcing AUROC to ≈ 0.5 regardless of keypoint source. The original battery (`src/baseline/gaussian.py`) uses a shared scale from unshifted distances and reaches GT AUROC 0.917/0.961 on the same 5 cows and same ROI-12D features.

## Bug Description

```python
# BUG (per-group standardisation — kills the shift signal):
np.r_[(d - d.mean()) / (d.std() + 1e-9), (di - di.mean()) / (di.std() + 1e-9)]

# FIX (shared scale from unshifted distances):
z_mu, z_sd = d.mean(), d.std() + 1e-9
np.r_[(d - z_mu) / z_sd, (di - z_mu) / z_sd]
```

Additionally, feature standardisation (mu, sd) is now fit on reference frames only, matching the repo's reference-only discipline.

## Step 4b Protocol

- **Max per-cow budgets**: each cow uses all its frames, split half ref / half eval
  - 1036: 224/225, 1063: 93/93, 1059: 87/88, 1087: 274/275, 1090: 265/266
- **Shift sweep**: {0.75, 1.0, 1.5, 3.0}
- **3 seeds**: 42, 123, 7 — per-cow 95% intervals across seeds
- Per cow only, no pooled number

## Step 4b Results

### Cow 1036 (LOCO OKS 0.998)

| Shift | GT Mean | GT CI | Pred Mean | Pred CI |
|-------|---------|-------|-----------|---------|
| 0.75 | 0.737 | [0.688, 0.789] | 0.705 | [0.675, 0.733] |
| 1.0 | 0.846 | [0.788, 0.902] | 0.801 | [0.764, 0.839] |
| 1.5 | 0.968 | [0.941, 0.989] | 0.935 | [0.919, 0.959] |
| 3.0 | 1.000 | [1.000, 1.000] | 0.988 | [0.984, 0.991] |

### Cow 1063 (LOCO OKS 0.156 — worst cow)

| Shift | GT Mean | GT CI | Pred Mean | Pred CI |
|-------|---------|-------|-----------|---------|
| 0.75 | 0.943 | [0.916, 0.963] | 0.698 | [0.636, 0.803] |
| 1.0 | 0.996 | [0.994, 0.997] | 0.799 | [0.730, 0.909] |
| 1.5 | 1.000 | [1.000, 1.000] | 0.936 | [0.893, 0.986] |
| 3.0 | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |

### Cow 1059 (LOCO OKS 1.000)

| Shift | GT Mean | GT CI | Pred Mean | Pred CI |
|-------|---------|-------|-----------|---------|
| 0.75 | 0.786 | [0.721, 0.852] | 0.874 | [0.870, 0.878] |
| 1.0 | 0.886 | [0.824, 0.945] | 0.928 | [0.913, 0.938] |
| 1.5 | 0.977 | [0.950, 0.999] | 0.961 | [0.956, 0.965] |
| 3.0 | 1.000 | [1.000, 1.000] | 0.963 | [0.957, 0.967] |

### Cow 1087 (LOCO OKS 1.000)

| Shift | GT Mean | GT CI | Pred Mean | Pred CI |
|-------|---------|-------|-----------|---------|
| 0.75 | 0.686 | [0.628, 0.734] | 0.684 | [0.613, 0.799] |
| 1.0 | 0.791 | [0.717, 0.845] | 0.773 | [0.697, 0.899] |
| 1.5 | 0.937 | [0.881, 0.970] | 0.896 | [0.848, 0.970] |
| 3.0 | 0.998 | [0.995, 1.000] | 0.984 | [0.975, 0.996] |

### Cow 1090 (LOCO OKS 0.746)

| Shift | GT Mean | GT CI | Pred Mean | Pred CI |
|-------|---------|-------|-----------|---------|
| 0.75 | 0.733 | [0.729, 0.741] | 0.908 | [0.857, 0.947] |
| 1.0 | 0.829 | [0.824, 0.834] | 0.958 | [0.933, 0.979] |
| 1.5 | 0.898 | [0.882, 0.920] | 0.979 | [0.958, 0.995] |
| 3.0 | 0.964 | [0.957, 0.967] | 0.998 | [0.996, 1.000] |

## Key Findings

1. **Cow 1063 shows the predicted-keypoint gap.** At shift=0.75, predicted AUROC drops to 0.698 vs GT 0.943 — a 0.245 gap. This is the only cow where predicted keypoints substantially lose sensitivity at small shifts.
2. **Cow 1090 shows predicted keypoints OUTPERFORM GT.** At all shifts, predicted > GT (e.g., shift=0.75: 0.908 vs 0.733). Plausibly because the detector's smoothing reduces within-cow variance, making the reference tighter.
3. **Cows 1036, 1059, 1087 show small gaps.** Predicted-vs-GT differences are within or near seed noise at most shifts.
4. **The battery survives on 4/5 cows.** Only cow 1063 shows a meaningful predicted-keypoint degradation, and only at small shifts (0.75–1.0).
5. **Shift 3.0 is detected by all cows regardless of keypoint source.** The battery is robust at large shifts.

## Interpretation Rules

1. **Shifts are injected in FEATURE space on predicted ROI features.** This proves: detector noise does not destroy feature-space shift sensitivity at these magnitudes, plausibly because deviation is scored against the cow's OWN reference from the SAME detector, so consistent per-cow detector bias cancels. This does NOT prove: that a real facial change in the image survives detection end-to-end, or that the detector is accurate in absolute terms (LOCO OKS stands as reported).

2. **Per-cow gaps are reported as-is.** At these budgets the intervals are tight enough to see cow-level differences, but no pooled claim is made.

## Artifacts
- `runs/features_cache/gt_vs_predicted.json` — Step 4b per-cow results with 3-seed intervals
- `src/eval/gt_vs_predicted.py` — battery script (fixed scoring, max budgets, shift sweep)
- `tests/test_gt_vs_predicted.py` — positive-control tests (GT shift-3.0 AUROC > 0.9)
