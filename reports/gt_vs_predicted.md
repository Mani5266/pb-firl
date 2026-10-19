# GT-vs-Predicted Deviation Battery (2026-10-19)

## Protocol
- Per cow: LOCO checkpoint → predicted keypoints → ROI-12D features
- Per cow: GT keypoints → ROI-12D features
- Per cow: own-mean + own-cov baseline, 20 ref + 20 eval frames, synthetic shifts 1.5/3.0
- No pooled number — per-cow only

## Results

| Cow | GT shift=1.5 | Pred shift=1.5 | GT shift=3.0 | Pred shift=3.0 |
|-----|-------------|---------------|-------------|---------------|
| 1036 | 0.465 | 0.513 | 0.478 | 0.515 |
| 1063 | 0.540 | 0.498 | 0.540 | 0.523 |
| 1059 | 0.505 | 0.473 | 0.515 | 0.475 |
| 1087 | 0.505 | 0.505 | 0.513 | 0.505 |
| 1090 | 0.523 | 0.545 | 0.535 | 0.568 |

## Key Finding: Battery Is Not Sensitive on CattleFace

**All AUROC values are ≈ 0.5 (random guess), for both GT and predicted keypoints.**

This is a **negative result** — the deviation battery cannot distinguish shifted from unshifted frames on CattleFace ROI-12D features, regardless of keypoint source.

## Why This Happens

### 1. Too few cows for population baseline
- CattleFace: 5 cows total
- ReCowGnition (falsify.json): 32 cows, D_full AUROC = 0.849 at shift=1.5
- The battery needs a population reference to detect deviations; 5 cows cannot provide this

### 2. ROI-12D features are geometry-only
- CattleFace ROI: 12 geometric descriptors (ear/eye/muzzle ratios)
- ReCowGnition embeddings: 64-D PCA of deep features
- Geometric features capture shape but not appearance/ texture changes that dominate real pain responses

### 3. Small per-cow sample size
- 20 ref + 20 eval = 40 frames per cow per shift
- With 12-D features and 20 reference frames, LedoitWolf covariance is poorly conditioned
- Results in near-random AUROC

## Comparison to ReCowGnition Falsification

| Metric | CattleFace (this battery) | ReCowGnition (falsify.json) |
|--------|--------------------------|----------------------------|
| Cows | 5 | 32 |
| Features | ROI-12D | Embedding-64D |
| Shift 1.5 own-mean AUROC | 0.465–0.540 | 0.849 |
| Shift 3.0 own-mean AUROC | 0.478–0.540 | — |
| Battery sensitive | **No** | Yes |

## Implications

1. **GT-vs-predicted gap is unmeasurable on CattleFace** — the battery itself is not sensitive enough with 5 cows and 12-D geometric features
2. **This is NOT evidence that predicted keypoints are "good enough"** — it means the evaluation tool lacks power
3. **The ReCowGnition dairy-scale battery remains the only validated deviation detector** — but it requires dairy embeddings, not beef cattle ROI features
4. **For a meaningful GT-vs-predicted gap, we would need:**
   - More cows (≥ 20) for a stable population baseline
   - Deep features (not just geometry) for sensitivity
   - Or accept that this gap cannot be measured on public data

## Conclusion

The GT-vs-predicted deviation battery **does not survive** on CattleFace public data — not because predicted keypoints are bad, but because the battery itself is underpowered at n=5 cows with 12-D geometric features. This is a limitation of the evaluation tool, not a validation of the landmark detector.

## Artifacts
- `runs/features_cache/gt_vs_predicted.json` — full per-cow battery results
- `src/eval/gt_vs_predicted.py` — battery script (reproducible)
