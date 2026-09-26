# PB-FIRL paper outline (every claim -> reports/eval_report.json; corrected 2026-09-26 review pass)

## Title
Personalised Baselines with Weakly-Supervised Attention for Cross-Domain Livestock Pain
Recognition: a proof-of-concept on public data

## Abstract
No public dataset jointly provides cattle identity + pain labels, so we validate the
personalisation mechanism cattle-side (CattleFace-RGBT, 5 cows x 1890 RGB frames; assumed
reference, unlabelled) and the pain classifier cross-species (sheep SPFES proxy mirror).
Per-cow deviation cuts linear cow-decodability (beef LDA 0.59 -> 0.29, dairy LDA 0.72 -> 0.03)
while lifting injected-shift AUROC (0.917 vs 0.825 beef; 0.848 vs 0.796 dairy, 151 cows).
Matched comparison: mean pooling >= attention (0.897 vs 0.864 BCE) on the dhash-clean split;
nonlinear identity leakage persists (5NN 0.72 beef / 0.41 dairy on deviation features).

## 1. Introduction
Problem, hypothesis, RQ1-RQ4 (see build guide). Honest scope: NOT a validated cattle pain
detector; cross-species proof-of-concept + cattle personalisation validation.

## 2. Related work
Zhang et al. 2025 (unseen-video recall 0.56); Feighelstein et al. 2026 (post-hoc z-score vs our
learned per-animal MVN); Dhaliwal et al. 2025 (identity shortcut -> our audit); Martvel et al.
2024 (geometry-first); Neethirajan et al. 2026 (transfer fails, cow-AUC 0.400).

## 3. Method
A. Geometry front-end: Keypoint R-CNN R50-FPN, 13 kpts, frozen backbone, identity-aware split;
   OKS success-rate metric (not COCO AP), true box areas.
B. Per-animal MVN (Ledoit-Wolf) + Mahalanobis deviation, per-cow z-score, cold-start fallback.
C. Patch-attention MIL (5x5 instances) vs mean-pool MLP, matched budgets, shared split, 3 seeds.
D. Upper CUSUM (k=0.5) chronological: fit 40% / calibrate 30% / eval 30% per cow.
E. Identity probes (LDA + 5NN), reference-fit stats, train-reference score-probe.
F. Blur-quartile shift proxy (species LODO blocked: equine gated).

## 4. Results (all from eval_report.json)
- Landmarks (`landmark_ap`, corrected areas): val mean_success 0.046 / success@0.50 0.242 /
  PCK 0.360; test 0.994 / 1.0 / 0.997. High inter-cow variance (n=1 cow each); downstream
  geometry uses GT keypoints.
- RQ1-beef (`variance`, `baseline_loio`): procrustes concentrates identity (between 0.24 -> 0.63,
  LDA 0.87 -> 0.96); ref-fit per-cow-z cuts it (between 0.010, LDA 0.29); injected-shift AUROC
  per-cow 0.917/0.961 vs population 0.825/0.892 (shifts 1.5/3.0).
- RQ1-dairy (`dairy`, 161 Holsteins): between 0.40; LDA 0.72 -> 0.03, 5NN 0.67 -> 0.41;
  LOIO (151 cows) per-cow 0.848/0.995 vs population 0.796/0.993. Cross-session cosine:
  same 0.52 < cross 0.76 < different-cow 1.01.
- RQ2 (`pain_fair`, matched, primary=test_raw dhash-clean): pool+BCE 0.897, pool+focal 0.893,
  attn+BCE 0.864, attn+focal 0.880 (means of 3 seeds). Attention does not beat mean pooling;
  loss effect small. Balanced `test` (0.92-0.93) flagged contaminated (60/74 dhash overlap).
  Clean-source subset ~0.81 all cells (185 rows, 8 pos: noisy).
- RQ3 (`identity_audit`): linear readout suppressed (LDA 0.29 vs chance 0.20; dairy 0.03 vs 0.006)
  but nonlinear leakage persists (5NN 0.72 beef / 0.41 dairy). Claim: reduction, not removal.
- RQ4 (`ablation.worst_group_blur`, matched BCE models): worst-group B0 0.844, B1 0.788,
  B2-MIL 0.761, B3 0.773. Species LODO deferred to §6.5 data.
- CUSUM (`cusum`, chronological): taus {0.01: 50.7, 0.02: 50.7, 0.05: 27.9}; held-out delays
  shift1.5: 25.2/25.2/11.6 frames; shift3.0: 11.4/11.4/6.6 frames.

## 5. Limitations
Beef geometry (5 cows, folder=sequence assumption) + dairy embeddings (161 cows); no joint
geometry+identity pain data. Landmark val weak (success@0.50 0.24, 1 cow). Sheep labels are
expression proxies, not diagnoses; balanced test contaminated (use test_raw). Nonlinear identity
leakage persists. No equine yet; hours-ahead claim deferred.

## 6. Next (on-request cattle-pain videos)
Real labelled cattle eval with video/animal-grouped splits; sequence-level MIL; B4 gradient
reversal; GT-vs-predicted landmark inference path; demo CLI.
