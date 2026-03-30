# Real-Time ASL Translation
### Applied Deep Learning — Final Project
**Northeastern University · Spring 2026**

> Keypoint-based temporal transformer for real-time American Sign Language word recognition on CPU.

---

## Group Members
| Name | GitHub | Role |
|------|--------|------|
| Gyula Planky | [@gyuszix](https://github.com/gyuszix) |  |
| Hrishikesh Pradhan | [@hspgit](https://github.com/hspgit) |  |
| Jian Gao | [@iamjaygao](https://github.com/iamjaygao) |  |
| Radhika Khurana | [@khuranaradhika](https://github.com/khuranaradhika) |  |

Possible roles: 
Data pipeline + augmentation, Model architecture + training, CTC decoder + distillation, ONNX export + evaluation + demo
---

## Project Overview

We build a lightweight transformer that takes hand keypoints extracted from a webcam stream and outputs a real-time ASL word transcript. The core contribution is making this work on **CPU in real time** — most existing models require GPU inference.

**Pipeline:**
```
Webcam → MediaPipe Holistic → Temporal Transformer → CTC Decoder → Text
          (keypoints, free)    (~1.2M params)          (greedy)
```

---

## Repository Structure

```
asl-realtime/
│
├── data/
│   ├── raw/
│   │   └── wlasl/
│   │       ├── WLASL_v0.3.json   # Annotation file (2000 signs, 21k instances)
│   │       └── videos/           # Downloaded .mp4 files (gitignored)
│   └── processed/
│       ├── train/                # Pre-extracted .npy keypoint files
│       └── vocab.json            # Sign label mapping
│
├── src/
│   ├── dataloader.py         # WLASLDataset, DataLoader, augmentations
│   ├── model.py              # SignTransformer, PositionalEncoding
│   ├── train.py              # Training loop, checkpointing, logging
│   ├── export.py             # ONNX export + quantization
│   └── demo.py               # Real-time webcam demo
│
├── scripts/
│   └── download_wlasl.py     # Parallel downloader for all WLASL videos
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
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Download WLASL
```bash
# Downloads all 2000 signs (direct links + YouTube via yt-dlp)
python scripts/download_wlasl.py

# Skip YouTube if you don't have yt-dlp or want a faster run
python scripts/download_wlasl.py --skip-youtube
```

**Actual yield (run March 2026):** 6,417 usable clips out of 21,083 instances (~30%).
High failure rate is expected — the dataset is from 2020 and most direct-link hosts are dead or blocked:

| Failure reason | Count |
|---|---|
| HTTP 403 (access denied) | 2,323 |
| HTTP 404 (dead link) | 2,181 |
| HTML served instead of video | 1,875 |
| Dead: www.aslpro.com | 1,728 |
| Dead: aslsignbank.haskins.yale.edu | 1,070 |
| Dead: www.signingsavvy.com | 344 |
| YouTube bot detection (need `--cookies`) | ~832 |
| Other | ~313 |

YouTube bot-detection errors can be partially resolved by passing browser cookies to yt-dlp — see [yt-dlp cookie docs](https://github.com/yt-dlp/yt-dlp#how-do-i-pass-cookies-to-yt-dlp).

### 3. Extract keypoints
```bash
python src/dataloader.py --extract --split train
python src/dataloader.py --extract --split val
python src/dataloader.py --extract --split test
```

### 4. Train baseline
```bash
python src/train.py --vocab 100 --epochs 50 --d_model 128 --n_layers 3
```

### 5. Run demo
```bash
python src/demo.py --model models/sign_model.onnx
```

---

## Datasets

| Dataset | Classes | Instances | Signers | Role | Access |
|---------|---------|-----------|---------|------|--------|
| WLASL2000 | 2,000 | 21,083 | 119 | Train + eval | [Free](https://github.com/dxli94/WLASL) |
| MS-ASL | 1,000 | 25,513 | 222 | Cross-dataset eval | [Request form](https://www.microsoft.com/en-us/research/project/ms-asl/) |

**MS-ASL notes:**
- Access requires a short form (name, institution, email) — download link arrives by email within minutes
- Annotations use `start_time` / `end_time` fields (clips are segments of longer YouTube videos, not pre-trimmed) — needs a trim step before keypoint extraction
- Label strings differ from WLASL (e.g. `"BOOK"` vs `"book"`) — normalize to lowercase and intersect before cross-dataset eval
- **Don't block on this.** Get WLASL training running first; MS-ASL eval is a clean one-day add-on after that.

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
- Open a PR → at least **one review** → merge
- Clear notebook outputs before committing:
  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
  ```

---

## Key References

- [WLASL Dataset](https://github.com/dxli94/WLASL) — Li et al., WACV 2020
- [SPOTER](https://github.com/matyasbohacek/spoter) — Bohácek & Hrúz, WACV 2022
- [MediaPipe](https://google.github.io/mediapipe/) — Lugaresi et al., 2019
- [PyTorch DataLoader tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [sign.mt](https://sign.mt) — Moryossef, EMNLP 2024
