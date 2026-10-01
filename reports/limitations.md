# Limitations (updated 2026-09-26, independent-review fix pass)

1. CattleFace-RGBT identity count: 96 unique cow_tag IDs observed in cow_mapping.json/metadata.csv
   vs 108 claimed. Pipeline uses 5 folder-mapped cows (data/README assumption, UNVERIFIED).
2. CattleFace-RGBT temperature cows: 25 unique observed vs 21 claimed.
3. ReCowGnition: LOCAL (6838/161 verified 2026-09-24, CC BY-NC-SA 4.0, kept out of git).
   No pain labels -> B4 adversarial still deferred; beef-trained landmarks don't transfer to
   dairy 112px crops (dairy_landmark_qc.png) -> dairy validation in embedding space only.
4. Sheep full Mendeley archive (2350 imgs) unverified; mirror labels are expression PROXIES, not
   diagnoses. Balanced `test` CONTAMINATED (60/74 dhashes overlap train_raw): all `test` numbers
   are direction-only; primary eval is test_raw (1 shared dhash) + clean-source subset
   (185 rows / 8 pos: noisy, sensitivity only; 40 source files shared with train as other crops).
5. UU Equine: no direct download; request-only. Species-level LODO is blocked; the
   executable robustness analysis is a train-defined blur-group proxy on test_raw. The
   current matched run reports worst-group B0 0.8081, B1 0.7500, B2-MIL 0.7854, and
   B3-fusion 0.7778, with source-file clustered intervals in `ablation.json`.
6. Landmark detector weak on held-out cow (val success@0.50 0.242, corrected area; n=1 cow).
   Earlier 0.66 AP50 was an x2*y2 area bug (fixed). Metric renamed: mean OKS success rate,
   not COCO AP. Downstream geometry uses GT keypoints.
7. Cattle frames are UNLABELLED assumed reference (not confirmed healthy); baseline validated on
   injected deviations, not real pain. Beef n=5 gap not significant (paired t p=0.24) — beef is
   supporting evidence for the dairy-scale finding, not standalone proof.
8. Matched comparison (shared split, matched budgets, 3 seeds): the cached fair comparison
   gives pool+BCE 0.8825 and attn+BCE 0.8592 on test_raw; attn+focal is 0.8746. This is
   underpowered for equivalence and must not be phrased as pooling being proven superior.
   Sequence-level MIL + B4 are deferred to the video phase.
9. Identity audit (leakage-safe: ref-fit stats, train-ref/score-probe): per-cow-z LDA accuracy
   equals the majority-class accuracy (beef 0.2904, dairy 0.0318) and its balanced accuracy
   equals the constant-class baseline (beef 0.20, dairy 0.0063). Nonlinear leakage persists
   (5NN accuracy 0.72 beef / 0.41 dairy; balanced accuracy 0.70 / 0.24). These are
   identity-conditioned controls, not identity-blind deployment features. Procrustes rotation
   convention fixed (was transposed).
10. CUSUM is chronological and uses a reference-only scaler/model. Thresholds are calibrated
    from the running CUSUM statistic on five calibration streams (0.01/0.02/0.05 are below
    the one-stream resolution); the held-out healthy stream FAR is reported separately because
    temporal drift transfers poorly. The synthetic feature
    injection run has 5 events, 3 misses at shifts 1.5/3.0, and a one-sided 95% miss-rate
    upper bound of 0.9236. Delays are detected-only frame counts, not real-event performance.
11. No validated cattle pain detector. BasePainVideoDataset is an interface stub: loader, frame
    sampling, temporal bags still to build when video data arrives.
12. Cold-start curve (30 dairy cows): the apparent crossing around 10–20 reference frames
    is a small-n embedding-space synthetic-shift result without a confidence band; it is not
    a deployment sample-complexity guarantee.
13. Falsification battery (32 cows, per-cow CIs, precision_ inverses, max cond 58.6):
    own-vs-pop gap +0.020 CI [-0.008, 0.049] (NOT significant); pooled-within-cov gap +0.029
    CI [0.006, 0.052]; shrinkage flat across lambda; wrong-animal gap +0.102 CI [0.064, 0.141].
    Permuted-identity null (50 shuffles): mean -0.039 CI [-0.050, -0.028], observed +0.020
    above all shuffles — the gap requires the true grouping, but this does NOT verify that
    folders are cows.
    Direction sweep (100 dirs): own beats pop in 59% of directions, mean gap +0.010.
    Within-session, synthetic, single-magnitude. Calibration-only AUROC is algebraically
    monotone and cannot support an operational threshold claim.
14. Cross-session detection (ref session A, probe B): at n_ref=20 (12 cows) own 0.798 vs
    session-matched pop 0.739, gap CI crosses zero; at n_ref=10 (31-39 cows) session-matched
    pop WINS significantly (gap -0.06..-0.09). Personalization advantage not established
    across sessions; small-ref regime favors same-session population fit.
