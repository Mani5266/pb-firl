# PB-FIRL data log — access date 2026-09-24, seed 42

## 6.1 CattleFace-RGBT (cattle geometry + identity, NO pain labels)
- HF: `phamtrongthang/CattleFace-RGBT` (repo_type=dataset), files: README.md,
  annotations/{rgb_keypoints.json, thermal_keypoints.json, metadata.csv, cow_mapping.json},
  rgb.zip (1534841531 B), thermal.zip (689723531 B), thermal_raw.zip (4210764919 B, OPTIONAL — not downloaded).
- Local: `data/cattleface_raw/annotations/` (+ rgb/thermal zips when downloaded).
- License: CC BY 4.0 (from dataset README front-matter). Attribute in README/paper:
  Pham et al. 2025 (Smart Agric. Technol. 12:101434); landmark benchmark Coffman et al. arXiv:2406.03431,
  ref code https://github.com/UARK-AICV/CattleFace-RGBT-benchmark
- VERIFIED on annotations: 1890 RGB images/ann, 2611 thermal images/ann, keypoints len 39 (=13 pts),
  3 dates {02_01, 02_06, 02_13}. COCO format, visibility flags {0,2}.
- MISMATCHES vs claimed (recorded, not fabricated): cow_mapping.json + metadata.csv hold
  108 sequence entries but only **96 unique cow_tag IDs** (claim: 108 unique cattle).
  metadata.csv has 29 rows / **25 unique cows** with temperature_f (claim: 21 cattle with rectal temp).
  Raw-thermal/video counts (30954 TIFF / 51 mp4) live inside not-yet-downloaded zips — unverified.
- Keypoint order: left_ear_base, left_ear_middle, left_ear_tip, poll, right_ear_base,
  right_ear_middle, right_ear_tip, left_eye, right_eye, muzzle, left_nostril, right_nostril, mouth.
- Identity assumption (load-bearing, flagged): image folders {1,2,17,25,50,64} match ONLY the
  02_13 session's sequence set (02_01 lacks 17/25/50/64; 02_06 has just 1,2), so folder is read as
  02_13 sequence -> cow_tag via cow_mapping.json: 1=1090, 2=1115 (thermal only), 17=1063, 25=1087,
  50=1036, 64=1059. If folders are chute/camera IDs instead, per-cow results are invalid -
  re-map when thermal_raw (dated) videos arrive.
- Disk extras beyond annotations: 24 RGB + 183 thermal JPGs unannotated (1914/2794 on disk).

## 6.2 ReCowGnition (dairy-cow identity, NO pain labels) — ON DISK 2026-09-24
- Official repo: https://github.com/marcohuber/recowgnition/ @ 37cd2b6 (HEAD 2026-09-24).
- License: CC BY-NC-SA 4.0 (repo README §License; Fraunhofer IGD 2026) — NON-COMMERCIAL, respect it.
  Local copy under `data/ReCowGnition - Dataset/` (gated request fulfilled manually; NOT in git).
  Includes CowDetect.pt (YOLO face/muzzle detector, unused this phase).
- VERIFIED local: 6838 JPGs, 161 unique cow IDs, 5 sessions
  {GX014028, GX014040, GX014041, GX024040, GX024041}, 112x112 px, per-cow 1–219 (median 31).
  Filename: [session]_[cowID]_[a]_[b].jpg. Matches paper (arXiv:2607.22071) exactly.
- Role: dairy-scale personalisation validation (embeddings; §dairy in eval_report).
  No pain labels -> B4 adversarial still deferred.

## 6.3 Sheep pain (Mendeley) — mirror on disk, full archive unverified
- Source DOI: 10.17632/y5sm4smnfr.5 (v5, 2020-03-15, Noor/Zhao, Harbin Inst. Tech.), CC BY 4.0.
  https://data.mendeley.com/datasets/y5sm4smnfr/5 — file listing needs JS/login; full 2350-img
  archive (1407 normal / 943 abnormal per Noor et al.) NOT directly verified.
- Usable mirror (dhash-deduped subset) on disk: HF `oliveirabruno01/sheep-facial-expression-benchmark`,
  `data/sheep_raw/data/*.parquet` — train 172 (86/86), test 74 (37/37),
  train_raw 898 (783/115), test_raw 225 (196/29); total raw 1123. Columns: image, label,
  label_name (no_pain/pain), blur_score, dhash, source_filename/url/doi, license (cc-by-4.0).
- Related paper DOI: 10.1016/j.compag.2020.105528.

## 6.4 UU Equine Pain Face (horse/donkey, ROI pain 0–2 + landmarks) — REQUEST-ONLY, not on disk
- 1855 horse + 531 donkey imgs; landmarks 54/44/45 pts (frontal/tilted/profile); per-ROI pain scores.
  Papers: Hummel et al. FG2020 (doi:10.1109/fg47880.2020.00114); Pessanha et al. TAFFC 2022/2023
  (doi:10.1109/TAFFC.2022.3177639); thesis https://doi.org/10.34626/q9cs-kg38 ;
  repo https://dspace.library.uu.nl/handle/1874/431590
- NO direct public download found: FG2020 paper states "obtained by request from the authors".
  Later papers say "publicly available" but give no URL. Status: UNVERIFIED — email authors before depending on it.
