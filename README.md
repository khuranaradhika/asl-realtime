# Real-Time ASL Translation
### Applied Deep Learning — Final Project
**Northeastern University · Spring 2026**

> Keypoint-based temporal transformer for real-time American Sign Language word recognition on CPU.

## How To Get Set Up

### 1. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
brew install ffmpeg              # Mac — needed for keypoint extraction and MS-ASL trimming
```

### 3. Download WLASL videos
```bash
python3 scripts/download_wlasl.py
```
This downloads ~8,500 videos to `data/raw/wlasl/videos/`. If YouTube downloads fail due to bot detection, export your browser cookies and save as `data/raw/wlasl/cookies.txt` — the script picks it up automatically.

### 4. Extract keypoints
```bash
python3 scripts/preprocess.py --split train --vocab 2000
python3 scripts/preprocess.py --split val   --vocab 2000
python3 scripts/preprocess.py --split test  --vocab 2000
```
Saves `.npy` keypoint files and split manifests to `data/processed/`. Downloads a ~25MB MediaPipe model on first run.


---

## Group Members
| Name | GitHub | Role |
|------|--------|------|
| Jian Gao | [@iamjaygao](https://github.com/iamjaygao) | Data pipeline + augmentation |
| Hrishikesh Pradhan | [@hspgit](https://github.com/hspgit) | Model architecture + training |
| Radhika Khurana | [@khuranaradhika](https://github.com/khuranaradhika) | CTC decoder + distillation |
| Gyula Planky | [@gyuszix](https://github.com/gyuszix) | ONNX export + evaluation + demo |
---

## Project Overview

We build a lightweight transformer that takes hand keypoints extracted from a webcam stream and outputs a real-time ASL word transcript. The core contribution is making this work on **CPU in real time** — most existing models require GPU inference.

**Pipeline:**
```
Webcam → MediaPipe HandLandmarker → Temporal Transformer → CTC Decoder → Text
          (keypoints, ~8ms/frame)    (~672K params, 2000 classes)  (greedy)
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
│   ├── config.py             # Shared constants and paths
│   ├── augmentations.py      # Keypoint transforms (flip, jitter, noise, normalize)
│   ├── keypoints.py          # MediaPipe extraction utilities (shared by preprocess + demo)
│   ├── dataloader.py         # WLASLDataset + get_dataloader
│   ├── model.py              # SignTransformer, PositionalEncoding
│   ├── decode.py             # Greedy CTC decode (+ beam search, Person 3)
│   ├── evaluate.py           # Top-1/Top-5 evaluation (+ WER, Person 3)
│   ├── train.py              # Training loop, checkpointing, logging
│   ├── export.py             # ONNX export + latency benchmark
│   └── demo.py               # Real-time webcam demo
│
├── scripts/
│   ├── preprocess.py         # One-time keypoint extraction from raw videos
│   ├── download_wlasl.py     # WLASL video downloader (direct links + yt-dlp)
│   └── download_msasl.py     # MS-ASL video downloader (YouTube via yt-dlp + ffmpeg trim)
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

## Training

```bash
# Student model (CPU-deployable, ~672K params)
python3 src/train.py --vocab 2000 --epochs 50 --d_model 128 --n_layers 3

# Teacher model (GPU recommended)
python3 src/train.py --vocab 2000 --epochs 100 --teacher
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
