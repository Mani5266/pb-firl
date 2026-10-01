# PB-FIRL paper outline (public-data audit scope)

## Proposed title
**Per-animal reference baselines for controlled facial-change detection: an adversarial
public-data audit**

## Central claim supported by the current repository
The repository implements and audits a per-animal Gaussian reference pipeline. On public
cattle/dairy data, it can compare detectors under **synthetic feature perturbations** and can
measure identity/session/quality confounds. It does **not** yet establish detection of
biologically meaningful change, welfare state, disease, or cattle pain.

## 1. Introduction
Population models can confuse stable identity and acquisition/session variation with state.
A per-animal reference may reduce that nuisance variation, but a convincing test requires
chronological or cross-session probes, strong population baselines, uncertainty, and real
longitudinal labels. The public-data study therefore treats the synthetic-shift results as a
stress test and makes the missing real-change experiment explicit.

## 2. Related work
Use the cited identity-shortcut, animal-geometry, per-animal normalization, and animal-pain
literature from the final bibliography. Do not describe the sheep expression proxy as a pain
label or the ReCowGnition identity data as a pain dataset.

## 3. Method
- CattleFace-RGBT: ROI geometry from annotated keypoints; folder-to-cow mapping is an
  unverified assumption and downstream geometry results use GT keypoints.
- ReCowGnition: frozen ResNet embeddings for identity/session and synthetic-shift stress tests;
  no pain labels.
- Sheep mirror: frozen embeddings and matched pooling/attention proxy classifiers; primary
  test is `test_raw`, because the balanced test has source overlap.
- Reference/probe discipline: fit scalers, PCA, covariance, and classifiers on the reference
  partition only wherever the protocol says so.
- Statistical unit: cow for cattle/dairy experiments and source filename for the sheep proxy;
  cluster-bootstrap intervals are reported where the artifact provides them.
- CUSUM: chronological 40/30/30 split, threshold candidates from the running statistic,
  feature-level synthetic injection followed by model rescoring, and held-out healthy FAR.

## 4. Results to report
- `falsify`: population/own-mean/own-covariance/shrinkage/wrong-animal comparisons under a
  frozen synthetic direction and matched reference budget. State explicitly that these are
  controlled perturbation results, not real-change detection.
- `coldstart` and `dairy`: reference-budget and cross-session identity/session boundaries;
  avoid claiming a universal 10–20-frame deployment crossing.
- `identity_audit` and `session_order`: raw accuracy must be accompanied by balanced accuracy,
  macro-F1, majority accuracy, and uniform chance. Per-cow-z results use the true cow identity
  and are identity-conditioned controls, not identity-blind deployment features.
- `pain_fair`/`ablation`: pooling and attention are compared on the source-disjoint sheep
  proxy test; report seed spread and source-clustered uncertainty. Do not call it cattle pain.
- `landmark_ap`: held-out cattle landmark results are weak and downstream GT-keypoint geometry
  does not measure predicted-keypoint deployment.
- `cusum`: report thresholds, calibration stream resolution, held-out healthy FAR, event count,
  miss count, exact upper bound, and detected-only delays. Do not write “zero misses.”

## 5. Limitations and non-claims
The main blockers are: no real change labels; random/same-session frame protocols in parts of
the battery; unverified folder mapping; synthetic constant offsets; missing strong hierarchical
baselines; small beef sample; residual identity/session structure; weak landmark transfer; and
no compatible equine/cattle species-LODO domain. The paper must not claim a validated cattle
pain detector, biologically meaningful change, causal welfare inference, or generalization from
sheep proxies to cattle.

## 6. Required held-out experiment
Before accessing the authors’ data, freeze the analysis code and protocol:
reference window, calibration window, chronological probe window, event definition, animal and
session IDs, threshold rule, control animals, metrics, cluster unit, and missing-data policy.
The held-out set must include unchanged controls and real labeled events. The primary analysis
should compare own-reference, population, pooled-within-animal, and hierarchical/shrinkage
baselines on cross-session or chronological probes, with cow-clustered intervals.
