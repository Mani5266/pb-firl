# Datasets Used in PB-FIRL (All Links)

## Public Datasets (directly accessible)

### 1. CattleFace-RGBT (Primary cattle geometry + identity)
- **Hugging Face:** https://huggingface.co/datasets/phamtrongthang/CattleFace-RGBT
- **Paper:** Pham et al. 2025, "CattleFever: An automated cattle fever estimation system", Smart Agricultural Technology 12:101434
- **Landmark benchmark:** Coffman et al., arXiv:2406.03431
- **Benchmark code:** https://github.com/UARK-AICV/CattleFace-RGBT-benchmark
- **License:** CC BY 4.0
- **Contents:** 1,890 RGB (2560×1440) + 2,611 thermal JPG images with 13 keypoints, 30,954 raw thermal TIFF frames, 51 thermal videos, 108 unique cattle, 21 with rectal temperature readings

### 2. Sheep Facial Expression Dataset (Pain proxy labels)
- **Mendeley Data:** https://data.mendeley.com/datasets/y5sm4smnfr/5
- **DOI:** 10.17632/y5sm4smnfr.5
- **Hugging Face mirror:** https://huggingface.co/datasets/oliveirabruno01/sheep-facial-expression-benchmark
- **Paper:** Noor et al., Computers and Electronics in Agriculture, DOI: 10.1016/j.compag.2020.105528
- **License:** CC BY 4.0
- **Contents:** 2,350 images (1,407 normal / 943 abnormal SPFES scores) — mirror has 1,123 deduped images (898 train_raw, 225 test_raw)

### 3. ReCowGnition (Dairy cow identity, 161 cows)
- **GitHub Repository:** https://github.com/marcohuber/recowgnition
- **Paper:** arXiv:2607.22071 (ICPR 2026 workshop)
- **Download:** Google Drive (request via repo README) — https://drive.google.com/drive/folders/1ywFj3_pew7uOG0FUnbDbkRiz6JPOpAaB
- **License:** CC BY-NC-SA 4.0 (NON-COMMERCIAL)
- **Contents:** 6,838 aligned face images (112×112), 161 Holstein cows, 5 recording sessions

---

## Request-Only Datasets (Not yet obtained)

### 4. UU Equine Pain Face Dataset
- **Description:** 1,855 horse + 531 donkey images with per-ROI pain scores (0-2) and manual landmarks
- **Papers:** Hummel et al., FG 2020 (DOI: 10.1109/fg47880.2020.00114); Pessanha et al., TAFFC 2022/2023
- **Repository:** https://dspace.library.uu.nl/handle/1874/431590
- **Contact:** Authors (Hummel/Pessanha/van Loon/Veltkamp at Utrecht University)
- **Status:** Request-only, no direct download found

### 5. Zhang et al. 2025 Dairy Cattle Pain Videos
- **Paper:** Zhang, Sailunaz & Neethirajan (2025), "Micro-Expression-Based Facial Analysis for Automated Pain Recognition in Dairy Cattle", AI 6(9):199
- **DOI:** 10.3390/ai6090199
- **Contact:** Corresponding author Neethirajan (active publisher in this space)
- **Status:** On request — dairy cattle pain videos with identity labels

### 6. CowPainCheck (Patel & Neethirajan 2026)
- **Paper:** JAST 68(4):959–1001, DOI: 10.5187/jast.2500286
- **Contact:** Corresponding author via JAST
- **Status:** On request — dairy cow facial pain expression dataset

### 6. Castration/Pain Longitudinal Studies
- **Review:** ABW Cases 2024 (DOI: 10.1079/abwcases.2024.0008)
- **Contact:** Primary study authors cited in review
- **Status:** Request-only — pre/post-treatment video sequences per animal

---

## Local Data Structure (after download)

```
data/
├── cattleface_raw/          # CattleFace-RGBT annotations + RGB/thermal JPGs
│   ├── annotations/
│   │   ├── rgb_keypoints.json
│   │   ├── thermal_keypoints.json
│   │   ├── metadata.csv
│   │   └── cow_mapping.json
│   ├── rgb/                 # 1,890 JPGs (5 folders)
│   └── thermal/             # 2,611 JPGs (6 folders)
├── sheep_raw/               # Sheep mirror (HF)
│   └── data/
│       ├── train-00000-of-00001.parquet
│       ├── test-00000-of-00001.parquet
│       ├── train_raw-00000-of-00001.parquet
│       └── test_raw-00000-of-00001.parquet
├── ReCowGnition - Dataset/  # Local copy (not in git)
│   ├── ReCowGnition/        # 6,838 JPGs
│   ├── ReCowGnition.zip
│   └── CowDetect.pt
└── README.md                # Full verification log
```

---

## License Summary

| Dataset | License | Commercial Use |
|---------|---------|----------------|
| CattleFace-RGBT | CC BY 4.0 | ✅ Yes |
| Sheep Expression Mirror | CC BY 4.0 | ✅ Yes |
| ReCowGnition | CC BY-NC-SA 4.0 | ❌ Non-commercial only |
| UU Equine | Request terms | Ask authors |
| Cattle-pain videos | Request terms | Ask authors |

---

## Access Status (as of 2026-10-03)

| Dataset | Status |
|---------|--------|
| CattleFace-RGBT | ✅ Downloaded, verified |
| Sheep Expression (mirror) | ✅ Downloaded, verified |
| ReCowGnition | ✅ Local copy (gated, not in git) |
| UU Equine | ❌ Request sent / pending |
| Cattle-pain videos | ❌ Request sent / pending |

---

## Notes for Paper Writing

- **CattleFace-RGBT:** Primary source for geometry experiments. Folder→cow mapping uses 02_13 session sequences (unverified assumption — documented in limitations).
- **Sheep data:** Labels are *expression proxies*, not clinical pain diagnoses. Balanced test split is contaminated (60/74 dhash overlap with train) — primary evaluation uses `test_raw` (dhash-disjoint).
- **ReCowGnition:** Used only for embedding-space validation (identity/session). No pain labels. CC BY-NC-SA — kept out of git.
- **Missing data:** All cattle-pain validation requires author datasets. The `BasePainVideoDataset` interface in `src/datasets/base.py` is ready for when they arrive.