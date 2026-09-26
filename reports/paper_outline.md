# PB-FIRL paper outline (every claim -> reports/eval_report.json; single-claim framing, 2026-09-26)

## Title
An Animal's Own Reference History Improves Detection of Facial Change Beyond Population Models

## Central claim
An animal's own reference history improves detection of meaningful change on later, unseen
observations, beyond what population models and simple normalization can explain. Everything
below tests that claim or its boundaries. Landmarks, MIL, CUSUM are tools, not accomplishments.

## 1. Introduction
Population averages confound identity with state (Dhaliwal et al. 2025: near-perfect accuracy
from identity shortcut). Hypothesis: per-animal reference comparison detects change that
population models miss. Honest scope: mechanism validated on proxies + assumed-reference
frames; NOT a validated cattle pain detector.

## 2. Related work
Zhang et al. 2025; Feighelstein et al. 2026 (post-hoc z-score vs learned per-animal MVN);
Dhaliwal et al. 2025 (identity shortcut); Martvel et al. 2024 (geometry-first);
Neethirajan et al. 2026 (transfer fails).

## 3. Method
Reference/probe discipline throughout: stats fit on reference only; probes trained on reference,
scored on probe; chronological streams; source-grouped splits. Falsification controls share
splits, features, perturbations, budgets.

## 4. Results (all from eval_report.json)
- Core (`falsify`, 32 dairy cows, frozen dims/budgets): population 0.829 = calibrated
  population 0.829 < own-mean 0.849 ~= full own-model 0.849 < shrinkage 0.868; matched
  wrong-animal reference 0.747. Benefit is animal-specific (not more data); mechanism is
  centering (covariance adds nothing at n_ref=20); shrinkage best. Simple calibration does
  NOT explain the operational gain.
- Budget (`coldstart`, 30 cows): per-cow crosses population at ~10-20 reference frames
  (0.785/0.812/0.848 at 10/20/40 vs pop 0.802).
- Beef geometry (`variance`, `baseline_loio`, n=5, folder assumption): same direction
  (0.917 vs 0.825) but not significant (paired t p=0.24) — supporting, not standalone.
- Identity boundary (`identity_audit`, `dairy`, `session_order`): linear readout suppressed
  (beef LDA 0.29, dairy 0.03, cross-session 0.008) but nonlinear leakage persists
  (5NN 0.72 / 0.41 / 0.09). Nuisance dependence reduced, not removed.
- Pain proxy (`pain_fair`, matched, test_raw dhash-clean): pool 0.883 ~= attn 0.875;
  AUPRC 0.558 (base 0.129). Pooling is NOT the driver; kept as default.
- Landmarks (`landmark_ap`): held-out success@0.50 0.24 — weak; geometry results are
  oracle (GT keypoints), inference penalty unmeasured.
- Early warning (`cusum`, chronological, ref-only scaler): FAR-0.01 constraint NOT met
  (saturated); achieved FAR overshoots targets (0.044 vs 0.01); delays 22.8/8.0 (shift 1.5),
  13.2/3.4 (shift 3.0), zero misses. Frame units (filenames, not timestamps).

## 5. Limitations
Folder=sequence assumption; n=5 beef; proxy labels + contaminated balanced test (primary:
test_raw); synthetic injections; nonlinear leakage open; front-end not in inference path;
no equine; no validated pain detector. Interface stub for video phase.

## 6. Next
Requested cattle-pain videos become a HELD-OUT test with predeclared protocol (frozen dims,
budgets, controls). Sequence MIL, B4, demo CLI, GT-vs-predicted inference path.
