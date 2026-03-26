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
applied_deep_learning_final/
│
├── data/
│   ├── raw/                  # Downloaded WLASL/MS-ASL videos (gitignored)
│   └── processed/            # Pre-extracted .npy keypoint files + manifests
│
├── src/
│   ├── dataloader.py         # WLASLDataset, DataLoader, augmentations
│   ├── model.py              # SignTransformer, PositionalEncoding
│   ├── train.py              # Training loop, checkpointing, logging
│   ├── evaluate.py           # Top-1/5 accuracy, WER, latency benchmarking
│   ├── distill.py            # Knowledge distillation (stretch goal)
│   ├── export.py             # ONNX export + quantization
│   └── demo.py               # Real-time webcam demo
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
# Clone the WLASL download scripts
git clone https://github.com/dxli94/WLASL.git /tmp/wlasl
cd /tmp/wlasl
pip install -r requirements.txt
python start_kit/downloader.py
# Move videos to our data directory
mv videos/ ../applied_deep_learning_final/data/raw/wlasl/
```
> ⚠️ Expect ~15–20% of clips to be unavailable (YouTube takedowns). This is normal.

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

| Dataset | Classes | Clips | Signers | Access |
|---------|---------|-------|---------|--------|
| WLASL100 | 100 | ~2,000 | 119 | [Free](https://github.com/dxli94/WLASL) |
| WLASL1000 | 1,000 | ~13,000 | 119 | Same |
| MS-ASL | 1,000 | 25,513 | 222 | [Free (Microsoft)](https://www.microsoft.com/en-us/research/project/ms-asl/) |

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
