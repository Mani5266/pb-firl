# LOCO Landmark Results (2026-10-19)

## Protocol
- 5-fold leave-one-cow-out cross-validation
- Per fold: 3 train cows + 1 val cow (checkpoint selection) + 1 held-out cow
- Val cow = fewest images among non-held-out cows (never tunes on held-out)
- Same training config as round 1: 8 epochs, lr=1e-4, batch=2, min_size=512, frozen backbone

## Results

| Held-out Cow | Train Cows | Val Cow | n | Success@0.50 | PCK@0.1 |
|---|---|---|---|---|---|
| 1036 | 1063, 1087, 1090 | 1059 | 449 | **0.998** | 0.862 |
| 1063 | 1036, 1087, 1090 | 1059 | 186 | **0.156** | 0.339 |
| 1059 | 1036, 1087, 1090 | 1063 | 175 | **1.000** | 0.998 |
| 1087 | 1036, 1063, 1090 | 1059 | 549 | **1.000** | 0.784 |
| 1090 | 1036, 1063, 1087 | 1059 | 531 | **0.746** | 0.546 |
| **Mean** | | | | **0.780** | |

## Key Findings

### 1. Cow 1063 remains a systematic failure
- Success@0.50 = 0.156, worse than the original 0.24 single held-out
- Confirms diagnosis: cow 1063's scale variance (3.2× spread) and pose asymmetry (R:L ≈ 1.8:1) are outside the training distribution
- The model cannot generalize to this cow even with 3 other cows in training

### 2. Cow 1090 degrades as held-out
- Was in training set for original model (success@0.50 = 1.0 on test cows)
- As held-out: success@0.50 = 0.746 — significant drop
- Suggests the model memorizes training cow appearances rather than learning generalizable features

### 3. Performance分布是 bimodal 的，由 cow 决定
- Performance is bimodal and cow-dependent: 3 cows ≥ 0.998, one at 0.746, one at 0.156
- Like-for-like on cow 1063 (same cow, old config vs LOCO config): 0.242 → 0.156 — 没有改善
- Mean 0.780 掩盖了 cow 级别的差异 — worst-cow 才是 deployability 的瓶颈
- Cross-animal generalization is the confirmed bottleneck

### 4. Val cow selection worked correctly
- Cow 1059 (175 images) used as val for 4/5 folds
- Cow 1063 (186 images) used as val for 1 fold
- Checkpoint selection never touched held-out cows

## Comparison to Original Split

| Metric | Original (val=1063) | LOCO Mean |
|--------|---------------------|-----------|
| Success@0.50 | 0.242 | 0.780 |
| n held-out | 186 | 1,890 (total) |
| Generalization evidence | Single cow | 5 cows rotated |

## Implications

1. **Cross-animal generalization is the #1 blocker** — confirmed by LOCO
2. **Data diversity is insufficient** — 3 training cows cannot cover the appearance space
3. **Augmentation is necessary but not sufficient** — scale jitter (round 2) helps but doesn't solve the fundamental gap
4. **More training data is required** — the model needs to see more cows to generalize

## Artifacts
- `runs/features_cache/loco_results.json` — full per-fold metrics
- `runs/checkpoints/loco/fold{1-5}_heldout*.pth` — 5 fold checkpoints (~226 MB each)
- `src/front_end/loco_landmarks.py` — LOCO training + eval script
- `runs/loco_train_summary.json` — per-fold training config, epochs, wall-clock, checkpoint hash
