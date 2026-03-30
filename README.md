# Real-Time ASL Translation
### Applied Deep Learning — Final Project
**Northeastern University · Spring 2026**

> Keypoint-based temporal transformer for real-time American Sign Language word recognition on CPU.

---

## Group Members
| Name | GitHub | Role |
|------|--------|------|
| Gyula Planky | [@gyuszix](https://github.com/gyuszix) | Data pipeline + augmentation |
| Hrishikesh Pradhan | [@hspgit](https://github.com/hspgit) | Model architecture + training |
| Jian Gao | [@iamjaygao](https://github.com/iamjaygao) | CTC decoder + distillation |
| Radhika Khurana | [@khuranaradhika](https://github.com/khuranaradhika) | ONNX export + evaluation + demo |

---

## Project Overview

We build a lightweight transformer that takes hand keypoints extracted from a webcam stream and outputs a real-time ASL word transcript. The core contribution is making this work on **CPU in real time** — most existing models require GPU inference.

**Pipeline:**
```
Webcam → MediaPipe HandLandmarker → Temporal Transformer → CTC Decoder → Text
          (keypoints, ~8ms/frame)    (~1.2M params)          (greedy)
```

---

## Repository Structure

```
asl-realtime/
│
├── data/
│   ├── raw/
│   │   ├── wlasl/
│   │   │   ├── WLASL_v0.3.json       # Annotation file (2000 signs, 21k instances)
│   │   │   └── videos/               # Downloaded .mp4 files (gitignored — run downloader)
│   │   └── msasl/
│   │       ├── MSASL_train.json       # MS-ASL train split annotations
│   │       ├── MSASL_val.json         # MS-ASL val split annotations
│   │       ├── MSASL_test.json        # MS-ASL test split annotations
│   │       ├── MSASL_classes.json     # Class label mapping
│   │       ├── MSASL_synonym.json     # Sign synonyms
│   │       └── videos/               # Downloaded + trimmed clips (gitignored — run downloader)
│   └── processed/
│       ├── train/                    # Pre-extracted .npy keypoint files (gitignored)
│       └── vocab.json                # Sign → index mapping
│
├── src/
│   ├── dataloader.py         # WLASLDataset, DataLoader, augmentations
│   ├── model.py              # SignTransformer, PositionalEncoding
│   ├── train.py              # Training loop, checkpointing, logging
│   ├── export.py             # ONNX export + latency benchmark
│   └── demo.py               # Real-time webcam demo
│
├── scripts/
│   ├── download_wlasl.py     # WLASL video downloader (direct links + yt-dlp)
│   └── download_msasl.py     # MS-ASL video downloader (YouTube via yt-dlp + ffmpeg trim)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory analysis on WLASL
│   ├── 02_baseline.ipynb     # Baseline training experiments
│   ├── 03_ablations.ipynb    # Ablation study results + plots
│   └── 04_demo_test.ipynb    # Demo prototype
│
├── models/
│   ├── checkpoints/          # .pt checkpoint files (gitignored)
│   └── sign_model.onnx       # Exported deployment model
│
├── docs/
│   ├── experiments.md        # Running experiment log
│   └── project_outline.pdf   # Full project proposal
│
├── results/
│   ├── figures/              # Pareto curves, confusion matrices
│   └── metrics/              # JSON/CSV evaluation results
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install
```bash
git clone https://github.com/khuranaradhika/asl-realtime.git
cd asl-realtime
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

---

## Datasets

### WLASL (primary — train + eval)

| Classes | Instances | Signers | Access |
|---------|-----------|---------|--------|
| 2,000 | 21,083 | 119 | [Free](https://github.com/dxli94/WLASL) |

The annotation file (`WLASL_v0.3.json`) is included in this repo. Videos must be downloaded separately.

```bash
# Download all videos (direct links + YouTube via yt-dlp)
python3 scripts/download_wlasl.py

# Skip YouTube if you don't have yt-dlp or want a faster run
python3 scripts/download_wlasl.py --skip-youtube
```

**Actual yield (March 2026):** 6,417 usable clips out of 21,083 instances (~30%). High failure rate is expected — the dataset was compiled in 2020 and most direct-link hosts are now dead or blocked.

| Failure reason | Count |
|---|---|
| HTTP 403 (access denied) | 2,323 |
| HTTP 404 (dead link) | 2,181 |
| HTML served instead of video | 1,875 |
| Dead: www.aslpro.com | 1,728 |
| Dead: aslsignbank.haskins.yale.edu | 1,070 |
| Dead: www.signingsavvy.com | 344 |
| YouTube bot detection | ~832 |

> **Recovering YouTube clips:** Export cookies from your browser using a browser extension
> (e.g. "Get cookies.txt LOCALLY" for Chrome), save as `data/raw/wlasl/cookies.txt`
> (gitignored), then re-run the downloader — it picks up the cookie file automatically.

---

### MS-ASL (cross-dataset evaluation)

| Classes | Instances | Signers | Access |
|---------|-----------|---------|--------|
| 1,000 | 25,513 | 222 | [Request form (free)](https://www.microsoft.com/en-us/research/project/ms-asl/) |

MS-ASL requires a short access form (name, institution, email). The download link arrives by email within a few minutes. The annotation JSONs (`MSASL_*.json`) are included in this repo once access is granted.

MS-ASL videos are hosted on YouTube as long-form recordings. Clips are defined by `start_time` / `end_time` timestamps — they are **not** pre-trimmed. The downloader handles trimming automatically via ffmpeg.

```bash
pip3 install yt-dlp
brew install ffmpeg
python3 scripts/download_msasl.py
```

> **YouTube bot detection:** Export browser cookies and save as `data/raw/msasl/cookies.txt`
> (gitignored) — the script picks it up automatically.

**Important notes:**
- MS-ASL and WLASL labels are both lowercase — no normalization needed
- Only the overlapping sign classes between WLASL and MS-ASL are used for cross-dataset eval
- Download takes several hours — each clip requires downloading the full YouTube video then trimming to the annotated segment
- **Don't block on this.** Get WLASL training running first. MS-ASL eval is a one-day add-on.

---

## Keypoint Extraction

Run after downloading videos. Uses MediaPipe HandLandmarker (Tasks API, compatible with mediapipe 0.10.30+). Downloads a ~25MB model file on first run automatically.

```bash
python3 src/dataloader.py --extract --split train --vocab 100
python3 src/dataloader.py --extract --split val   --vocab 100
python3 src/dataloader.py --extract --split test  --vocab 100
```

---

## Training

```bash
# Student model (CPU-deployable, ~1.2M params)
python3 src/train.py --vocab 100 --epochs 50 --d_model 128 --n_layers 3

# Teacher model (GPU recommended, ~18M params)
python3 src/train.py --vocab 100 --epochs 100 --teacher
```

Checkpoints save to `models/checkpoints/` automatically.

---

## Export + Demo

```bash
# Export trained model to ONNX and benchmark CPU latency
python3 src/export.py --checkpoint models/checkpoints/student_d128_l3_v100_best.pt

# Real-time webcam demo (requires trained ONNX model)
python3 src/demo.py --model models/sign_model.onnx --vocab 100
```

---

## Results

| Model | WLASL100 Top-1 | WLASL100 Top-5 | CPU Latency | Size |
|-------|---------------|---------------|-------------|------|
| Baseline (no aug) | — | — | — | — |
| + Augmentation | — | — | — | — |
| + Distillation | — | — | — | — |
| **Full model** | — | — | — | — |
| Teacher (reference) | — | — | N/A (GPU) | — |

*Results will be updated as experiments complete.*

---

## Git Workflow

- `main` — stable, working code only
- Feature branches: `feature/your-name-description`
  - e.g. `feature/gyula-mediapipe-pipeline`
  - e.g. `feature/radhika-onnx-export`
- Open a PR → at least **one review** → merge
- Clear notebook outputs before committing:
  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
  ```
- **Never commit** `cookies.txt`, `failed_downloads.jsonl`, or anything in `data/raw/*/videos/`

---

## Key References

- [WLASL Dataset](https://github.com/dxli94/WLASL) — Li et al., WACV 2020
- [MS-ASL Dataset](https://www.microsoft.com/en-us/research/project/ms-asl/) — Joze & Koller, BMVC 2019
- [SPOTER](https://github.com/matyasbohacek/spoter) — Bohácek & Hrúz, WACV 2022
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — Google, 2023
- [PyTorch DataLoader tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [sign.mt](https://sign.mt) — Moryossef, EMNLP 2024
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloader used for video acquisition
