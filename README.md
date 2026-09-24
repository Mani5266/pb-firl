# PB-FIRL — Personalised, Weakly-Supervised, Identity-Audited Livestock Pain Recognition

B.Tech final-year major project. Detects pain-related facial change in cattle by comparing each
animal against **its own healthy baseline** (not a population average), learning from weak labels
with **attention MIL**, auditing that features don't just memorize cow identity, and measuring
**cross-domain robustness**. Public-data phase: personalisation validated cattle-side,
pain detector proven cross-species.

## Results (seed 42, all in `reports/eval_report.json`)

| Question | Result |
|---|---|
| RQ1 — per-animal baseline | Injected-shift AUROC **0.917 vs 0.825** (subtle), 0.961 vs 0.892 (strong), per-cow vs population |
| RQ2 — MIL vs hard labels | Pain AUROC **0.959 vs 0.885** (balanced), **0.898 vs 0.874** (natural); F1 0.904 vs 0.809 |
| RQ3 — identity leakage | Cow-ID decodability **0.566 → 0.125** (chance 0.20) on deviation features, signal retained |
| RQ4 — domain shift | Worst quality-shift group AUROC **0.856** (MIL) vs 0.809 (baseline) |
| Early warning | CUSUM delay 21→10 frames (subtle) / 6→1 (strong) across FAR 0.01→0.05 |
| Landmarks | Keypoint R-CNN R50-FPN, 13 pts: test mAP ~1.0, val AP50 0.66 (1 held-out cow each, honestly reported) |

## Repro (data download excluded)
```powershell
pip install -r requirements.txt
$env:PYTHONHASHSEED=42
python -m src.front_end.ingest            # manifests + GT features
python -m src.front_end.train_landmarks   # Keypoint R-CNN fine-tune (GPU, ~40 min)
python -m src.front_end.eval_landmarks    # OKS AP + reports/figures/landmark_qc.png
python -m src.eval.variance_decomp        # Phase 2: reports/variance_decomposition.md
python -m src.baseline.gaussian           # Phase 3: per-animal MVN, LOIO
python -m src.pain.train_sheep            # Phase 4: B0 vs MIL embeddings
python -m src.pain.improve_mil            # Round-1: focal-loss MIL sweep -> sheep_mil_best.pth
python -m src.front_end.retrain_landmarks # Round-2: scale-jitter retrain
python -m src.cusum.cusum                 # Phase 5: delay-vs-FAR
python -m src.audit.leakage               # Phase 6: identity audit
python -m src.eval.lodo                   # Phase 7: ablations + worst-group + ROC
python -m src.eval.build_report           # reports/eval_report.json
python -m unittest discover -s tests      # count + split-integrity checks
```

## Data (see `data/README.md` for URLs, licenses, checksums)
- **CattleFace-RGBT** (CC BY 4.0): 1890 RGB + 2611 thermal annotated frames, 13 keypoints, ~5 cows
  with per-animal streams. Geometry, baselines, audit.
- **Sheep pain mirror** (CC BY 4.0, DOI 10.17632/y5sm4smnfr.5): 1123 images for the pain module.
- **ReCowGnition** (gated) + **UU Equine** (request-only): pending — unlock dairy identities, B4,
  and species-level LODO. Pipeline loads them via `src/datasets/base.py` with zero changes.

## Honest scope
Cross-species proof-of-concept + cattle personalisation validation — **not** a validated cattle
pain detector yet. Full limits in `reports/limitations.md`; paper outline in
`reports/paper_outline.md` (every claim traces to `eval_report.json`).
