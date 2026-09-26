# Data-request emails (DRAFTS — send from your INSTITUTIONAL email, all 6 in parallel)

Replace: [YOUR NAME], [INSTITUTION], [INSTITUTION EMAIL], [SUPERVISOR]. Attach/link repo:
https://github.com/Mani5266/pb-firl. Ask SMALL (subset first). Follow up once after 10 days.

## 1. CattleFace-RGBT authors (folder mapping + thermal_raw videos)
To: corresponding author of Pham et al. 2025 (Smart Agric. Technol. 12:101434) + GitHub issue on
UARK-AICV/CattleFace-RGBT-benchmark.
Subject: Data question — CattleFace-RGBT sequence mapping + thermal_raw videos (B.Tech project)

> I am [YOUR NAME], a B.Tech final-year student at [INSTITUTION] ([SUPERVISOR]'s group), working
> on per-animal facial baselines for cattle pain recognition (repo link above, MIT-licensed code).
> Two questions on CattleFace-RGBT: (1) do the annotated-image folders {1,2,17,25,50,64}
> correspond to 02_13 session sequences (my folder→cow mapping via cow_mapping.json assumes so —
> please confirm or correct); (2) may I access the 51 thermal_raw videos (.mp4) for true temporal
> streams? Even 5–10 sequences would suffice for a start. I will cite Pham et al. 2025 + the
> benchmark paper and comply with CC BY 4.0 attribution.

## 2. UU Equine Pain Face (Hummel/Pessanha/van Loon/Veltkamp)
To: corresponding author of Hummel et al. FG2020 (paper states data "obtained by request").
Subject: Data request — UU Equine Pain Face Dataset — B.Tech final-year project

> [Same intro.] Your per-ROI pain scores + landmarks are the closest public match to my
> geometry-first pipeline (13 cattle landmarks → ROI descriptors). May I access the horse/donkey
> images with ROI pain annotations for a leave-one-species evaluation? A subset (e.g. frontal +
> tilted horse faces) is enough to start. Full citation + license compliance guaranteed.

## 3. Zhang / Sailunaz / Neethirajan 2025 (dairy-cattle pain videos)
To: corresponding author (Neethirajan active in this space), DOI 10.3390/ai6090199.
Subject: Data request — dairy-cattle pain videos — per-animal baseline study

> [Same intro.] My core hypothesis is that comparing each animal to its own healthy baseline
> beats population averages, but no public set has cattle identity + pain jointly. May I access
> your dairy-cattle pain videos with animal IDs and pain labels — or at minimum per-animal
> evaluation splits? Even ~10 animals × baseline + pain sequences would let me test this. I will
> report my benchmark numbers back to you under your protocol.

## 4. Patel & Neethirajan 2026 — CowPainCheck (JAST 68(4):959–1001)
To: corresponding author via JAST page, DOI 10.5187/jast.2500286.
Subject: Data request — CowPainCheck dairy-cow pain frames — B.Tech project

> [Same intro.] Your CowPainCheck dataset is the most recent dairy-cow facial pain data I know.
> Could you share frames + labels (or the evaluation protocol) for an independent benchmark? Happy
> to sign a data-use agreement and return my results to you.

## 5. Noor / Zhao sheep full archive (Mendeley 10.17632/y5sm4smnfr.5)
To: dataset owners via Mendeley + paper authors (Comp. Electron. Agric. 10.1016/j.compag.2020.105528).
Subject: Full sheep facial-expression archive + subset question

> [Same intro.] I work from the public HF mirror (1123 images) and found the balanced test split
> overlaps train (60/74 shared hashes) — could you share a direct zip of the full 2350-image
> archive and confirm which subset your reported results used, plus any dedup guarantees?

## 6. Cattle castration / longitudinal pain studies (via ABW Cases 2024 review, 10.1079/abwcases.2024.0008)
To: corresponding authors of primary pre/post-treatment video studies cited in the review.
Subject: Pre/post-treatment cattle video sequences — early-warning validation

> [Same intro.] My early-warning module (CUSUM on deviation streams) is currently validated only
> on synthetic shifts. Your pre/post-treatment sequences per animal are exactly the temporal
> structure I need. May I access video or frame sequences for a few animals (baseline +
> pain-onset), with timestamps if available? Citation + compliance guaranteed.
