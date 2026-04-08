# Real-Time ASL Translation
### Applied Deep Learning — Final Project
**Northeastern University · Spring 2026**

> Keypoint-based temporal transformer for real-time American Sign Language word recognition, deployable as a web application.

---

## Group Members
| Name | GitHub | Role |
|------|--------|------|
| Radhika Khurana | [@khuranaradhika](https://github.com/khuranaradhika) | Data pipeline, model architecture + training, web server |
| Jian Gao | [@iamjaygao](https://github.com/iamjaygao) | Dataset sourcing, keypoint extraction + augmentation |
| Hrishikesh Pradhan | [@hspgit](https://github.com/hspgit) | Baseline experiments + knowledge distillation |
| Gyula Planky | [@gyuszix](https://github.com/gyuszix) | ONNX export, evaluation + demo |

---

## Project Overview

Given a continuous video stream of a person signing, our system identifies which ASL word is being signed and produces a text transcript in real time. We target the top 300 most frequent signs across our combined training corpus.

Existing models achieving strong benchmark accuracy (I3D, SlowFast, VideoMAE) rely on GPU inference over raw video frames and cannot run on a standard laptop. Our approach avoids raw video at inference time entirely — MediaPipe reduces each frame to a 126-dimensional hand keypoint vector, which a compact transformer classifies. The result is a system fast enough to run in a browser, on CPU, with no install required for the end user.

**Pipeline:**
```
Webcam → MediaPipe (browser) → keypoints → WebSocket → Transformer → Text
          126 floats/frame, ~8ms           ~5ms inference
```

**Success metrics:**
- ≥ 60% Top-1 accuracy on WLASL val set (vocab=300)
- ≤ 50ms end-to-end inference latency
- Runs in browser on standard laptop (no GPU required)

---

## Architecture

We compare three temporal modeling approaches on the same data and training setup:

| Model | How it sees time | Params | Loss |
|-------|-----------------|--------|------|
| **1D CNN** | Local 3-frame windows | ~450K | Cross-entropy |
| **BiLSTM** | Sequential hidden state | ~735K | Cross-entropy |
| **Transformer** | Full self-attention across all frames | ~452K | Cross-entropy |

All three use **cross-entropy loss** with global average pooling — the correct setup for isolated word classification. (CTC loss is retained in the codebase for future continuous/streaming signing work.)

If Transformer wins, it validates that long-range temporal context matters for ASL word recognition. If not, local/sequential patterns are sufficient and data is the bottleneck — both are defensible findings.

---

## Repository Structure

```
asl-realtime/
│
├── data/
│   ├── raw/
│   │   ├── wlasl/
│   │   │   ├── WLASL_v0.3.json         # Annotations (2000 signs, 21k instances)
│   │   │   └── videos/                 # Downloaded .mp4 files (gitignored)
│   │   └── msasl/
│   │       ├── MSASL_{train,val,test}.json
│   │       ├── MSASL_classes.json
│   │       └── videos/                 # Downloaded + trimmed clips (gitignored)
│   └── processed/
│       ├── train/                      # Extracted .npy keypoint files (gitignored)
│       ├── val/
│       ├── test/
│       ├── aslcitizen/
│       ├── vocab.json                  # 300-sign vocab (sign → index)
│       ├── train_manifest.json         # WLASL train — 1,048 samples
│       ├── val_manifest.json           # WLASL val — 229 samples
│       ├── test_manifest.json          # WLASL test — 206 samples
│       ├── aslcitizen_train_manifest.json  # ASL Citizen — 1,696 samples
│       └── combined_train_manifest.json    # WLASL + ASL Citizen — 2,726 samples
│
├── src/
│   ├── config.py             # Shared constants and paths
│   ├── augmentations.py      # Flip, speed perturbation, jitter, noise, normalization
│   ├── keypoints.py          # MediaPipe extraction (hands + holistic)
│   ├── dataloader.py         # WLASLDataset + get_dataloader
│   ├── model.py              # SignClassifier (CE), SignTransformer (CTC), CNN, BiLSTM
│   ├── decode.py             # Greedy CTC decode (+ beam search, Jian)
│   ├── evaluate.py           # Top-1/Top-5 accuracy, per-class breakdown
│   ├── train.py              # Training loop, checkpointing, per-class breakdown
│   ├── server.py             # FastAPI WebSocket inference server
│   ├── export.py             # ONNX export + CPU latency benchmark
│   └── demo.py               # Real-time webcam demo (local, OpenCV)
│
├── frontend/
│   └── index.html            # Browser demo — MediaPipe.js + WebSocket
│
├── scripts/
│   ├── preprocess.py         # Keypoint extraction → .npy + manifests
│   ├── download_wlasl.py     # WLASL downloader (direct links + yt-dlp)
│   └── download_msasl.py     # MS-ASL downloader (YouTube + ffmpeg trim)
│
├── models/
│   ├── checkpoints/          # .pt checkpoint files (gitignored)
│   └── sign_model.onnx       # Exported deployment model
│
├── docs/
│   ├── step2.md              # Step 2 changes — model, training, web server (Radhika)
│   ├── experiments.md        # Running experiment log
│   └── project_outline.pdf   # Full project proposal
│
├── results/
│   ├── figures/              # Pareto curves, confusion matrices
│   └── metrics/              # JSON per-run results
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/khuranaradhika/asl-realtime.git
cd asl-realtime
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg             # Mac — needed for MS-ASL trimming
```

---

## Datasets

We train on a combined corpus of WLASL + ASL Citizen, filtered to vocab=300.

| Dataset | Role | Clips | Access |
|---------|------|-------|--------|
| WLASL | Train + in-distribution eval | 1,048 train / 229 val / 206 test | [Free](https://github.com/dxli94/WLASL) |
| ASL Citizen | Train only | 1,696 | — |
| **Combined** | **Training corpus** | **2,726** | — |
| MS-ASL | Cross-dataset generalization eval only | ~13,375 usable | [Request form](https://www.microsoft.com/en-us/research/project/ms-asl/) |

MS-ASL is **not used for training** — held out entirely as a cross-dataset generalization test.

### Download WLASL videos

```bash
python3 scripts/download_wlasl.py              # direct links + YouTube
python3 scripts/download_wlasl.py --skip-youtube  # faster, fewer clips
```

**Yield (March 2026):** 6,417 clips out of 21,083 (~30%). Most losses are dead hosting links from 2020. Export browser cookies via "Get cookies.txt LOCALLY" (Chrome), save as `data/raw/wlasl/cookies.txt` (gitignored).

### Download MS-ASL videos

```bash
python3 scripts/download_msasl.py
```

**Yield (March 2026):** 13,375 clips out of 25,513 (~52%). Save cookies as `data/raw/msasl/cookies.txt`. Keep `MAX_WORKERS=2` to avoid rate limiting.

---

## Keypoint Extraction

Run after downloading videos. MediaPipe HandLandmarker model (~25MB) downloads automatically on first run.

```bash
python3 scripts/preprocess.py --split train --vocab 300
python3 scripts/preprocess.py --split val   --vocab 300
python3 scripts/preprocess.py --split test  --vocab 300
```

Always run train split first — it builds `vocab.json` which val/test depend on.

**Output format:** Each `.npy` file is `(T, 126)` — T frames × 126 features (21 left-hand landmarks × 3 + 21 right-hand landmarks × 3, xyz).

**Normalization** is applied at training time in `augmentations.py`:
1. Wrist-relative translation — subtracts dominant wrist position, invariant to signer position in frame
2. Hand-span scale normalization — divides by wrist-to-middle-fingertip distance, invariant to hand size and camera distance

**Holistic option (225-dim, for next model version):**
```bash
python3 scripts/preprocess.py --split train --vocab 300 --holistic
```
Saves to `data/processed/train_holistic/`, does not overwrite existing files.

---

## Augmentations

Applied during training only:

| Augmentation | Description | Effect |
|---|---|---|
| Horizontal flip | Swap left/right hands, mirror x | Doubles effective dataset size |
| Speed perturbation | Resample at 0.8×–1.2× speed | Handles fast/slow signers |
| Temporal jitter | Randomly drop or repeat frames | Robustness to dropped frames |
| Gaussian noise | σ=0.01 on all coordinates | Simulates MediaPipe detection noise |

---

## Training

All experiments run from the project root with `python3 -m`. The `--combined` flag uses the 2,726-sample combined manifest.

```bash
# EXP-006 — Transformer + augmentation (main model)
python3 -m src.train --model transformer --vocab 300 --epochs 50 --combined

# EXP-005 — Transformer, no augmentation (ablation)
python3 -m src.train --model transformer --vocab 300 --epochs 50 --no-augment --combined

# EXP-004 — BiLSTM baseline
python3 -m src.train --model lstm --vocab 300 --epochs 50 --combined

# EXP-003 — 1D CNN baseline
python3 -m src.train --model cnn --vocab 300 --epochs 50 --combined
```

Each run saves independently:
- `models/checkpoints/{run_name}_best.pt` — best val checkpoint
- `results/metrics/{run_name}_history.json` — loss/Top-1/Top-5 per epoch
- `results/metrics/{run_name}_per_class.json` — per-class accuracy breakdown

See `docs/experiments.md` for the full experiment log and `docs/step2.md` for details on the training pipeline changes.

---

## Export + Demo

### Local webcam demo (OpenCV)

```bash
python3 -m src.export \
  --checkpoint models/checkpoints/transformer_d128_l3_v300_combined_best.pt \
  --vocab 300

python3 -m src.demo --model models/sign_model.onnx --vocab 300
```

### Web demo (browser)

```bash
# Install server dependencies (once)
pip install fastapi "uvicorn[standard]"

# Export model first (see above), then start server
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. MediaPipe runs in the browser — only 126 keypoint floats per frame are sent to the server, no video.

---

## Results

| Model | Top-1 (WLASL val) | Top-5 (WLASL val) | Top-1 (MS-ASL) | CPU Latency | Params |
|-------|------------------|------------------|----------------|-------------|--------|
| 1D CNN | — | — | — | — | ~450K |
| BiLSTM | — | — | — | — | ~735K |
| Transformer (no aug) | — | — | — | — | ~452K |
| Transformer + aug | — | — | — | — | ~452K |
| Transformer (CTC, step 1) | 6.7% | 7.1% | — | — | ~452K |

*Results will be updated as experiments complete. See `docs/experiments.md` for per-run details.*

---

## Git Workflow

- `main` — stable, working code only
- `dev` — integration branch
- Feature branches: `feature/your-name-description`
  - e.g. `feature/jian-beam-search`
  - e.g. `feature/gyula-onnx-export`
- Open a PR → at least **one review** → merge to `dev`, then `main`
- **Never commit** `cookies.txt`, `failed_downloads.jsonl`, or anything in `data/raw/*/videos/`

---

## Key References

- [WLASL Dataset](https://github.com/dxli94/WLASL) — Li et al., WACV 2020
- [MS-ASL Dataset](https://www.microsoft.com/en-us/research/project/ms-asl/) — Joze & Koller, BMVC 2019
- [SPOTER](https://github.com/matyasbohacek/spoter) — Bohácek & Hrúz, WACV 2022
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — Google, 2023
- [YouTube-ASL](https://github.com/google-research/google-research/tree/master/youtube_asl) — Google, 2023
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloader used for video acquisition
