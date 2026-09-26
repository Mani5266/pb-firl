# Variance decomposition (Phase 2, GT keypoints, RGB, 5 cows x 1890 imgs)

Protocol: per-cow reference/probe split; stats fit on reference, metrics on probe; LDA trained on reference, scored on probe. Chance cow accuracy: 0.20.

| representation | between | within | cow-LDA acc |
|---|---|---|---|
| raw-centered 26D | 0.239 | 0.761 | 0.866 |
| procrustes 26D | 0.631 | 0.369 | 0.958 |
| ROI-12D raw | 0.101 | 0.899 | 0.590 |
| ROI-12D per-cow-z (ref-fit) | 0.010 | 0.990 | 0.290 |

Interpretation: Procrustes alignment concentrates identity (pose nuisance removed); per-cow z-scoring with reference-fit stats cuts linear cow-decodability on unseen frames. Caveat: the per-cow-z between-share is near zero partly by construction (within-cow centring removes mean differences); rely on the LDA/5NN audit (identity_audit.json) for the decodability claim. Identity labels assume folders = 02_13 sequences (see data/README).
