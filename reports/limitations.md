# Limitations (updated 2026-09-24, Phases 0-8 complete, public-data scope)

1. CattleFace-RGBT identity count: 96 unique cow_tag IDs observed in cow_mapping.json/metadata.csv
   vs 108 claimed. Pipeline uses 5 folder-mapped cows (data/README assumption).
2. CattleFace-RGBT temperature cows: 25 unique observed vs 21 claimed.
3. ReCowGnition: LOCAL (6838/161 verified 2026-09-24, CC BY-NC-SA 4.0, kept out of git).
   No pain labels -> B4 adversarial still deferred; beef-trained landmarks don't transfer to
   dairy 112px crops (dairy_landmark_qc.png) -> dairy validation in embedding space only.
4. Sheep full Mendeley archive (2350 imgs) unverified; results from deduped HF mirror (1123 raw).
5. UU Equine: no direct download; request-only. Species-level LODO blocked; blur-quartile
   quality shift used as proxy (worst-group B0 0.809).
6. Landmark detector generalises unevenly across unseen cows (val mAP 0.15 vs test 1.0, n=1 cow
   each). Downstream geometry uses GT keypoints; detector AP reported honestly.
7. Cattle frames have no pain labels: baseline validated on injected deviations, not real pain.
8. Patch-MIL upgraded to focal loss (val-selected): now beats B0 on AUROC/F1/worst-group on
   both splits; B3-fusion best balance. Per-image instance standardisation tried and dropped
   (hurt val AUROC). Sequence-level MIL + B4 adversarial regulariser still deferred to video phase.
9. Hours-ahead CUSUM claim deferred (no longitudinal pain-onset data); delays are in frames on
   injected shifts. No validated cattle pain detector; no farm-deployment claims.
