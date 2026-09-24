# Variance decomposition (Phase 2, GT keypoints, RGB, 5 cows × 1890 imgs)

Chance cow accuracy: 0.20. LDA 5-fold CV.

| representation | between | within | cow-LDA acc |
|---|---|---|---|
| raw-centered 26D | 0.235 | 0.765 | 0.834 |
| procrustes 26D | 0.615 | 0.385 | 0.862 |
| ROI-12D raw | 0.112 | 0.888 | 0.566 |
| ROI-12D per-cow-z | 0.000 | 1.000 | 0.125 |

Interpretation: Procrustes alignment removes pose/scale nuisance; per-cow z-scoring removes the between-cow morphology share that remains. The drop in cow-decodability from raw to deviation features is the empirical case for personalisation (RQ1/RQ3). Identity labels assume folders = 02_13 sequences (see data/README).
