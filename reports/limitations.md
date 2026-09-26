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
5. UU Equine: no direct download; request-only. Species-level LODO blocked; blur-quartile
   quality shift used as proxy (matched worst-group B0 0.844).
6. Landmark detector weak on held-out cow (val success@0.50 0.242, corrected area; n=1 cow).
   Earlier 0.66 AP50 was an x2*y2 area bug (fixed). Metric renamed: mean OKS success rate,
   not COCO AP. Downstream geometry uses GT keypoints.
7. Cattle frames are UNLABELLED assumed reference (not confirmed healthy); baseline validated on
   injected deviations, not real pain.
8. Matched comparison (shared split, matched budgets, 3 seeds): mean pooling >= attention
   (test_raw 0.897 vs 0.864 BCE); focal-loss gain was a protocol artifact. Earlier MIL-win
   claim withdrawn. Sequence-level MIL + B4 deferred to video phase.
9. Identity audit (leakage-safe: ref-fit stats, train-ref/score-probe): linear readout suppressed
   (beef LDA 0.29, dairy LDA 0.03) but NONLINEAR leakage persists (5NN 0.72 beef / 0.41 dairy).
   Claim is reduction, not removal. Procrustes rotation convention fixed (was transposed).
10. CUSUM chronological (fit/calibrate/eval split): held-out delays larger than retrospective
    ones (shift1.5: 25.2/11.6 frames). No longitudinal pain-onset data; no farm-deployment claims.
11. No validated cattle pain detector. BasePainVideoDataset is an interface stub: loader, frame
    sampling, temporal bags still to build when video data arrives.
