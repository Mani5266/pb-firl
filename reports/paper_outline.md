# PB-FIRL paper outline (every claim -> reports/eval_report.json)

## Title
Personalised Baselines with Weakly-Supervised Attention for Cross-Domain Livestock Pain
Recognition: a proof-of-concept on public data

## Abstract
No public dataset jointly provides cattle identity + pain labels, so we validate the
personalisation mechanism cattle-side (CattleFace-RGBT, 5 cows x 1890 RGB frames) and the pain
detector cross-species (sheep SPFES mirror, 1123 images). Per-cow deviation features cut
cow-identity decodability 0.566 -> 0.125 while lifting injected-shift AUROC 0.825 -> 0.917;
focal-loss patch-attention MIL beats hard-label pooling on pain AUROC 0.959 vs 0.885 (balanced)
and 0.898 vs 0.874 (natural prevalence), with best worst-group 0.856 vs 0.809.

## 1. Introduction
Problem, hypothesis, RQ1-RQ4 (see build guide). Honest scope: NOT a validated cattle pain
detector; cross-species proof-of-concept + cattle personalisation validation.

## 2. Related work
Zhang et al. 2025 (unseen-video recall 0.56, our B0 analogue 0.973 balanced / 0.931 natural);
Feighelstein et al. 2026 (post-hoc z-score vs our learned per-animal MVN); Dhaliwal et al. 2025
(identity shortcut -> our audit); Martvel et al. 2024 (geometry-first); Neethirajan et al. 2026
(transfer fails, cow-AUC 0.400 -> our blur-shift worst-group 0.809).

## 3. Method
A. Geometry front-end: Keypoint R-CNN R50-FPN, 13 kpts, frozen backbone, identity-aware split.
B. Per-animal MVN (Ledoit-Wolf) + Mahalanobis deviation, per-cow z-score, cold-start fallback.
C. Patch-attention MIL (5x5 instances) vs hard-label MLP.
D. Upper CUSUM (k=0.5) on z-deviation, FAR-calibrated thresholds.
E. LDA identity probe on frozen representations.
F. Blur-quartile shift proxy (species LODO blocked: equine gated).

## 4. Results (all from eval_report.json)
- Landmarks (`landmark_ap`, round 2 with scale jitter): val mAP 0.157 / AP50 0.661 /
  PCK 0.395; test mAP 0.996 / AP50 1.0 / PCK 0.997. High inter-cow variance (n=1 cow each) -
  reported, not hidden; downstream geometry uses GT keypoints.
- RQ1 (`variance`, `baseline_loio`): procrustes concentrates identity (between-share 0.235 ->
  0.615); per-cow-z removes it (cow-LDA 0.566 -> 0.125); injected-shift AUROC per-cow 0.917 /
  0.961 vs population 0.825 / 0.892 (shifts 1.5 / 3.0).
- RQ2 (`pain_mil_v2`, `ablation`): focal-loss MIL (val-selected) beats hard inheritance -
  balanced test AUROC 0.959 vs 0.885, F1 0.904 vs 0.809; natural prevalence AUROC 0.898 vs 0.874,
  F1 0.571 vs 0.383. B3 late-fusion keeps recall 1.00/0.86 at AUROC 0.948/0.893. Per-image
  instance standardisation hurt (val 0.74 vs 0.87) - dropped.
- RQ3 (`identity_audit`): raw ROI 0.566, procrustes 0.862, per-cow-z 0.125, z-dev scalar 0.262
  vs chance 0.20. Pain scalar near chance while holding Phase-3 signal.
- RQ4 (`ablation.worst_group_blur`): worst-group AUROC B2-MIL 0.856, B3-fusion 0.844,
  B0 0.809, B1 0.753. MIL is also the most quality-shift-robust. Species LODO deferred to §6.5 data.
- CUSUM (`cusum`): taus {0.01: 6.94, 0.02: 4.05, 0.05: 1.55}; delays shift1.5: 20.6/16.8/9.6
  frames; shift3.0: 5.6/3.0/0.8 frames.

## 5. Limitations
96 (not 108) unique cow tags; folder=02_13-sequence identity assumption; 1-cow val/test;
sheep mirror is a 1123-image dedup subset; no equine/ReCowGnition yet; hours-ahead claim deferred.

## 6. Next (on-request cattle-pain videos)
Re-run Phases 3-7 unchanged via BasePainVideoDataset; sequence-level MIL; B4 gradient reversal.
