# PB-FIRL — Personalised, Weakly-Supervised, Identity-Audited Livestock Pain Recognition

B.Tech final-year major project. Tests whether comparing each animal against **its own
reference appearance** (not a population average) gives more defensible pain representations,
with weak-label learning, identity-leakage auditing, and robustness checks. Public-data phase:
personalisation validated cattle-side (geometry) + dairy-scale (embeddings); pain classifier
proven only on sheep expression proxies.

## Results (seed 42, all in `reports/eval_report.json`; corrected 2026-09-26 review pass)

| Question | Result |
|---|---|
| RQ1 — per-animal baseline (beef) | Injected-shift AUROC **0.917 vs 0.825** (subtle), 0.961 vs 0.892 (strong) |
| RQ1 — dairy scale (161 Holstein) | Cow-LDA **0.72 → 0.03** (chance 0.006) after per-cow-z; LOIO 0.848 vs 0.796 |
| RQ2 — pooling (matched, clean split) | Mean pooling **0.897** >= attention 0.864 (BCE); attention win withdrawn |
| RQ3 — identity leakage | Linear readout suppressed (LDA 0.29/0.03) but **nonlinear leakage persists** (5NN 0.72/0.41) |
| RQ4 — quality shift | Worst blur group B0 0.844 vs MIL 0.761 (matched models) |
| Early warning | Chronological CUSUM delay 25→12 frames (subtle) / 11→7 (strong) across FAR |
| Landmarks | 13-pt KRCNN: held-out success@0.50 **0.24** (corrected area; 1 cow each) |

Key corrections from review: landmark area bug fixed (0.66 was inflated); balanced sheep `test`
found contaminated (60/74 dhash overlap) — primary eval is the dhash-clean `test_raw`;
audit/CUSUM made leakage-safe; Procrustes rotation fixed. See `reports/limitations.md`.

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

## Honest scope
Proof-of-concept with trustworthy numbers — **not** a validated cattle pain detector.
Full limits in `reports/limitations.md`; paper outline in `reports/paper_outline.md`
(every claim traces to `eval_report.json`).
