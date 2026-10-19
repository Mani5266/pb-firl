# Landmark Failure Diagnosis (2026-10-19)

## Phenomenon
- Val cow 1063: OKS success@0.50 = **0.242** (n=186)
- Test cow 1059: OKS success@0.50 = **1.000** (n=175)
- Both cows are unseen during training (train = cows 1090, 1087, 1036)

## Per-Cow Breakdown

| Cow | Split | n | Mean OKS | Success@0.50 | PCK@0.1 | Mean Box Area | Box Area Range |
|-----|-------|---|----------|--------------|---------|---------------|----------------|
| 1036 | train | 449 | — | — | — | 395,545 ± 14,252 | 321k–430k |
| 1087 | train | 549 | — | — | — | 415,448 ± 118,254 | 327k–1,307k |
| 1090 | train | 531 | — | — | — | 781,005 ± 192,224 | 239k–1,414k |
| **1063** | **val** | **186** | **0.354** | **0.242** | **0.360** | **679,571 ± 270,823** | **307k–997k** |
| **1059** | **test** | **175** | **0.980** | **1.000** | **0.997** | **397,034 ± 16,459** | **343k–432k** |

## Per-Keypoint Error (normalized by √area)

**Val cow 1063 (worst → best):**
| Keypoint | Mean Error | P90 Error |
|----------|-----------|-----------|
| right_ear_base | 0.625 | 0.861 |
| right_eye | 0.599 | 0.959 |
| right_ear_middle | 0.517 | 1.197 |
| right_ear_tip | 0.472 | 1.314 |
| left_eye | 0.417 | 1.041 |
| poll | 0.400 | 0.914 |
| left_ear_base | 0.344 | 0.927 |
| left_ear_middle | 0.336 | 0.775 |
| left_ear_tip | 0.308 | 0.757 |
| right_nostril | 0.282 | 0.897 |
| muzzle | 0.274 | 0.897 |
| mouth | 0.249 | 0.727 |
| left_nostril | 0.229 | 0.853 |

**Test cow 1059 (all keypoints):** mean error 0.012–0.042, p90 0.017–0.037

## Key Findings

### 1. Box detection is NOT the issue
Model produces predictions for cow 1063 (no zero-score samples). The failure is in keypoint localization, not face detection.

### 2. Scale variance is the primary differentiator
- Cow 1063 box area: **3.2× spread** (307k–997k, std=271k)
- Cow 1059 box area: **1.3× spread** (343k–432k, std=16k)
- Training cows: 1036 (1.3×), 1087 (4.0×), 1090 (5.9×) — but 1087/1090 have larger absolute areas

The model saw large boxes during training (cows 1087, 1090 up to 1.4M), but cow 1063's **distribution shape** is different — it has both very small and very large boxes in the same session.

### 3. Severe left-right asymmetry on cow 1063
- Right ear base: 0.625 vs Left ear base: 0.344 (1.8× worse)
- Right eye: 0.599 vs Left eye: 0.417 (1.4× worse)
- Right ear middle: 0.517 vs Left ear middle: 0.336 (1.5× worse)

This asymmetry is absent on cow 1059 (all keypoints 0.01–0.04). Suggests cow 1063 has a **persistent head pose bias** (yaw/pitch) not well-represented in training data.

### 4. Worst-case visualization
See `reports/figures/landmark_worst_val.png` — shows systematic offset of predicted keypoints from GT, particularly on the right side of the face.

## Root Cause Hypothesis

The model overfits to the **specific scale-pose distribution** of the 3 training cows. Cow 1063's combination of:
- High box-area variance within a single session
- Head pose bias (evidenced by L-R asymmetry)
- Possibly different coat color/pattern from training cows

...falls outside the training distribution, causing keypoint localization to degrade selectively (right side worse than left).

## Implications for LOCO

1. **Single held-out cow is not representative** — need rotation across all 5 cows
2. **Training data diversity is the bottleneck** — 3 cows with specific scale/pose distributions
3. **Augmentation priority**: scale jitter (already added in round 2) + rotation augmentation to handle pose variation
4. **Box size normalization** could help — resize all crops to fixed resolution before keypoint prediction

## Artifacts
- `runs/features_cache/landmark_diagnosis.json` — full per-cow, per-keypoint metrics
- `reports/figures/landmark_worst_val.png` — worst 6 predictions on cow 1063
- `reports/figures/landmark_worst_test.png` — worst 6 predictions on cow 1059
- `src/front_end/diagnose_landmarks.py` — diagnostic script (reproducible)
