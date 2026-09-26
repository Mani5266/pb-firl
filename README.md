# PB-FIRL — Does an animal's own reference history beat population models at detecting change?

B.Tech final-year major project. One claim: **an animal's own reference history improves
detection of meaningful change on later, unseen observations, beyond what population models
and simple normalization can explain.** Falsification battery (32 dairy cows): own-reference
0.849 vs matched wrong-animal 0.747 vs population 0.829; shrinkage best at 0.868.

## Results (seed 42, all in `reports/eval_report.json`; corrected 2026-09-26 review pass)

| Question | Result |
|---|---|
| RQ1 — per-animal baseline (beef) | Injected-shift AUROC **0.917 vs 0.825** (subtle), 0.961 vs 0.892 (strong) |
| RQ1 — dairy scale (161 Holstein) | Cow-LDA **0.72 → 0.03** (chance 0.006) after per-cow-z; LOIO 0.848 vs 0.796 |
| RQ2 — pooling (matched, clean split) | Mean pooling **0.883** ≈ attention 0.875; pooling not the driver |
| RQ3 — identity leakage | Linear readout suppressed (LDA 0.29/0.03) but **nonlinear leakage persists** (5NN 0.72/0.41) |
| RQ4 — quality shift | Worst blur group B0 0.796 vs MIL 0.777 (matched models) |
| Early warning | Chronological CUSUM, zero misses; FAR-0.01 target unachievable, delays at achieved FAR |
| Landmarks | 13-pt KRCNN: held-out success@0.50 **0.24** (corrected area; 1 cow each) |

Key corrections from review: landmark area bug fixed (0.66 was inflated); balanced sheep `test`
found contaminated (60/74 dhash overlap) — primary eval is the dhash-clean `test_raw`;
audit/CUSUM made leakage-safe; Procrustes rotation fixed. See `reports/limitations.md`.

## Architecture

```
                ┌──────────────────── FRONT-END ────────────────────┐
                │  CattleFace-RGBT 2560x1440                        │
                │  Keypoint R-CNN R50-FPN → 13 kpts (frozen body)   │
                │  Procrustes align → 12-D ROI descriptors          │
                │  (eye/ear/muzzle geometry, scale-normalized)      │
                └───────────────────────┬───────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
   PERSONALISATION              PAIN (sheep proxy)            EARLY WARNING
   per-cow MVN (Ledoit-Wolf)    frozen ResNet-50 → 25         upper CUSUM (k=0.5)
   Mahalanobis deviation,       patch instances →             on z-deviation stream;
   per-cow z-score;             mean-pool MLP (default)       fit/calibrate/eval split
   cold-start → population      attention MIL (tested)        chronological per cow
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        ▼
                          AUDIT + ROBUSTNESS (over everything)
                          LDA/5NN identity probes (ref-fit stats);
                          blur-quartile worst-group; LOIO; falsification
                          controls (wrong-animal, shrinkage, calibration)
```

Module map: `src/front_end/` (ingest, geometry, train/retrain/eval_landmarks) ·
`src/baseline/` (per-cow Gaussian + cold-start) · `src/pain/` (sheep embeddings, MIL,
fair 4-cell comparison) · `src/cusum/` · `src/audit/` (identity leakage) ·
`src/eval/` (variance, dairy, session-order, cold-start, falsify, lodo, report builder) ·
`src/datasets/` (loaders + video interface stub). Reference/probe discipline everywhere:
nothing fit on reference ever sees probe frames; all seeds fixed at 42.

## Repro (data download excluded)
```powershell
pip install -r requirements.txt
$env:PYTHONHASHSEED=42
python -m src.front_end.ingest            # manifests + GT features
python -m src.front_end.train_landmarks   # Keypoint R-CNN fine-tune (GPU, ~40 min)
python -m src.front_end.eval_landmarks    # OKS success-rate + reports/figures/landmark_qc.png
python -m src.eval.variance_decomp        # Phase 2: reports/variance_decomposition.md
python -m src.baseline.gaussian           # Phase 3: per-animal MVN, LOIO
python -m src.pain.train_sheep            # Phase 4: cached embeddings + first MIL
python -m src.pain.fair_compare           # matched 4-cell comparison -> pain_fair.json
python -m src.cusum.cusum                 # Phase 5: chronological delay-vs-FAR
python -m src.audit.leakage               # Phase 6: LDA+5NN identity audit
python -m src.eval.lodo                   # Phase 7: matched ablations + worst-group + ROC
python -m src.eval.falsify                # kill-test: 6 controls, matched budgets
python -m src.eval.session_order          # cross-session reference/probe
python -m src.eval.coldstart               # reference-budget curve
python -m src.datasets.recowgnition       # dairy manifest (local gated copy)
python -m src.eval.dairy                  # dairy-scale validation -> dairy.json
python -m src.eval.build_report           # reports/eval_report.json
python -m unittest discover -s tests      # 20 math/count/integrity checks
```

## Data (see `data/README.md` for URLs, licenses, checksums)
- **CattleFace-RGBT** (CC BY 4.0): 1890 RGB + 2611 thermal annotated frames, 13 keypoints, ~5 cows
  with per-animal streams (folder=sequence mapping unverified). Geometry, baselines, audit.
- **Sheep pain mirror** (CC BY 4.0, DOI 10.17632/y5sm4smnfr.5): 1123 images, expression-proxy
  labels (not diagnoses); balanced test contaminated — use test_raw.
- **ReCowGnition** (local gated copy, CC BY-NC-SA 4.0): 6838 dairy faces / 161 cows. Identity/scale
  validation only (no pain labels).
- **UU Equine** (request-only) + **cattle-pain videos** (on request): still missing — unlock
  species LODO and real cattle-pain eval. `src/datasets/base.py` is an interface stub.

## What we've achieved to date

- **Public-data pipeline end-to-end** (Phases 0–8): 4 datasets verified/logged, 20/20 tests green,
  one-command repro chain, every number in `reports/eval_report.json` with config hash + seed.
- **Central claim tested, not just stated**: 6-control falsification battery — own-reference beats
  matched wrong-animal reference (0.849 vs 0.747); shrinkage best (0.868); calibration-only
  explains nothing (0.829 = 0.829). Mechanism identified: centering, not covariance.
- **Dairy scale**: 161 Holsteins — linear identity readout 0.72→0.03, LOIO 0.848 vs 0.796,
  cross-session ordering same<cross<different-cow, cold-start crossing at ~10–20 frames.
- **Honest negatives published**: attention ≯ pooling (0.875 vs 0.883 matched); nonlinear identity
  leakage persists (5NN 0.72/0.41); balanced sheep test withdrawn as contaminated; landmark val
  weak (0.24); beef n=5 not significant (p=0.24) — supporting evidence only.
- **Two independent review passes applied**: area bug, rotation fix, leakage-safe protocols,
  chronological CUSUM with saturation/miss reporting, cluster CIs, AUPRC, LICENSE/CITATION.
- **Ready for the held-out test**: 6 data-request email drafts in `reports/data_request_emails.md`;
  video interface stub + predeclared protocol waiting for cattle-pain videos.

## Honest scope
Proof-of-concept with trustworthy numbers — **not** a validated cattle pain detector.
Full limits in `reports/limitations.md`; paper outline in `reports/paper_outline.md`
(every claim traces to `eval_report.json`).
