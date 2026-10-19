# Step 3, Factor 1: All‑Sessions / All‑Dates Training Data Expansion

## Motivation (from Muse's diagnosis)

The deviation battery survives on LOCO checkpoints (Step 4b). The suspected bottleneck is identity coverage: only 3 of 5 cows appear in each fold's training set. The current LOCO trains on 3 cows (fold‑specific subsets: {1,25,50}, {17,1059,1090}, etc.). The hypothesis is that adding the remaining non‑held‑out cow per fold — i.e. training on **4 cows** instead of 3 — will increase worst‑cow (cow 1063) success@0.50 and thereby improve cross‑animal generalisation.

## Data reality check (pre‑condition)

- The entire annotated RGB dataset contains exactly **1890 images in 5 folders**, all from one recording session (02_13). There is no additional dated RGB data on disk. 
- The 5 cows are 1036, 1063, 1059, 1087, 1090. 
- In the current LOCO each fold uses **3 of 5 cows** as training (the other 2 are val + held‑out). 
- Factor 1 expands training to **4 cows** per fold (the held‑out cow becomes the single excluded cow; the former val‑cow is absorbed into training).  No new dated data exists; all annotated RGB is from one session (02_13).  The only “additional” identities come from absorbing the former val‑cow into training.

## Protocol

| Item | Description |
|------|-------------|
| **Membership change** | Per fold: train on 4 cows (all non‑held‑out), previously 3. The former val‑cow is now a training cow. |
| **Checkpoint selection** | Fixed epoch 7 (the final epoch of the 8‑epoch schedule). Never uses a held‑out cow for model selection. |
| **Training hyper‑parameters** | Identical to baseline: 8 epochs, lr = 1e‑4, batch = 2, min‑size = 512, frozen backbone, seed = 42. |
| **Evaluation** | After training, evaluate the final‑epoch checkpoint on the held‑out cow using the same `eval_oks` procedure as baseline LOCO. Record success@0.50, PCK@0.1. |
| **Per‑cow reporting** | Always report cow 1063 first; then the other 4 cows in ascending order. |
| **Stop rule** | If cow 1063’s success@0.50 does **not** improve over the baseline value of 0.156, cease the ablation and report honestly — do not continue to factors 2 or 3. |
| **Wall‑clock** | 5 folds × 8 epochs × ~7 min/epoch ≈ **2 h 40 min**. Report actual wall‑clock time. |
| **Outputs** | `runs/checkpoints/loco_f1/fold{1‑5}_heldout*.pth` (final‑epoch checkpoints); `runs/features_cache/loco_factor1.json` (5‑fold results table); updated `loco_results.md` with an “Factor 1” section. |

## Expected outcome

- If the extra identity (the former val‑cow) helps, cow 1063’s success@0.50 should rise above 0.156.
- If it does **not** improve, the ablation stops — the gap is due to something other than the number of training cows (e.g., scale variance / pose bias as the diagnosis already suggested).
- If it does improve, the ablation continues in principle (but per Muse’s rules, we stop as soon as the worst cow fails to improve).

## Files to create

- `src/front_end/loco_factor1.py` — new script running the 5‑fold expansion.
- `reports/ablation_factor1.md` — this protocol + final results.
- Updated `reports/loco_results.md` with a “Factor 1” section appended.

## Non‑goals

- Do **not** add thermal data, data augmentation, schedule changes, or learning‑rate tweaks. This factor is **solely** about training‑membership expansion from 3 → 4 cows.

---

*Begin implementation once this protocol is approved.*