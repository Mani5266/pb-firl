# GT-vs-Predicted Deviation Battery (2026-10-19, corrected)

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

## Corrected Protocol
- Per cow: LOCO checkpoint → predicted keypoints → ROI-12D features
- Per cow: GT keypoints → ROI-12D features
- Per cow: own-mean + own-cov baseline, 20 ref + 20 eval frames, synthetic shifts 1.5/3.0
- Shared-scale scoring from unshifted distances; feature scaler fit on reference frames only
- No pooled number — per-cow only

## Corrected Results

| Cow | GT shift=1.5 | Pred shift=1.5 | GT shift=3.0 | Pred shift=3.0 |
|-----|-------------|---------------|-------------|---------------|
| 1036 | 0.900 | 0.920 | 1.000 | 1.000 |
| 1063 | 0.948 | 0.898 | 1.000 | 0.985 |
| 1059 | 0.958 | 1.000 | 1.000 | 1.000 |
| 1087 | 0.870 | 0.823 | 1.000 | 0.952 |
| 1090 | 0.840 | 0.945 | 0.945 | 0.980 |

## Sanity Gate

All GT shift-3.0 arms score ≥ 0.945 — well above the 0.9 threshold. The battery is sensitive.

## Per-Cow GT-vs-Predicted Gap

| Cow | Gap shift=1.5 (Pred−GT) | Gap shift=3.0 (Pred−GT) |
|-----|------------------------|------------------------|
| 1036 | +0.020 | 0.000 |
| 1063 | −0.050 | −0.015 |
| 1059 | +0.042 | 0.000 |
| 1087 | −0.048 | −0.048 |
| 1090 | +0.105 | +0.035 |

## Key Findings

1. **The battery survives on all 5 cows.** Predicted keypoints from LOCO checkpoints produce AUROC 0.82–1.00 across cows and shifts.
2. **No systematic degradation.** Predicted-vs-GT gaps range from −0.05 to +0.11, with no consistent direction. Some cows improve with predicted keypoints (1090: +0.105 at shift=1.5), others degrade slightly (1087: −0.048).
3. **Cow 1063 does not collapse the battery.** Despite landmark OKS success@0.50 of 0.156, the deviation battery still detects shifts (pred AUROC 0.898/0.985). The geometric features remain informative even with noisy keypoints.
4. **The worst predicted-arm result is 0.823** (cow 1087, shift=1.5) — still well above chance.

## Conclusion

The GT-vs-predicted deviation battery **survives** on CattleFace public data with LOCO checkpoints. Predicted keypoints preserve 82–100% of the GT battery's sensitivity, with no systematic degradation. The earlier "battery not sensitive" conclusion was an artefact of a scoring bug and is retracted.

## Artifacts
- `runs/features_cache/gt_vs_predicted.json` — corrected per-cow battery results
- `src/eval/gt_vs_predicted.py` — battery script (fixed scoring)
- `tests/test_gt_vs_predicted.py` — positive-control tests (GT shift-3.0 AUROC > 0.9)
