# Experiment Log

Running log of all training runs, configs, and results.
Update this after every experiment — it's the team's shared memory.

---

## Format

Each entry should include:
- Date, who ran it
- Full command used
- Key hyperparameters
- Results (Top-1, Top-5, latency if measured)
- Notes / observations

---

## Experiments

### EXP-001 — Baseline (no augmentation)
- **Date:** TBD
- **Run by:** 
- **Command:**
  ```bash
  python src/train.py --vocab 100 --epochs 50 --d_model 128 --n_layers 3
  ```
- **Config:** d_model=128, nhead=4, n_layers=3, ffn=256, dropout=0.1, lr=3e-4, batch=32
- **Results:**
  - Val Top-1: —
  - Val Top-5: —
  - Training time: —
- **Notes:** Baseline — no augmentation, no distillation

---

### EXP-002 — + Augmentation
- **Date:** TBD
- **Run by:** 
- **Command:**
  ```bash
  python src/train.py --vocab 100 --epochs 50 --d_model 128 --n_layers 3
  # (augmentation is on by default for train split)
  ```
- **Config:** Same as EXP-001 + flip, jitter, noise, coord norm
- **Results:**
  - Val Top-1: —
  - Val Top-5: —
- **Notes:** Expected +3–8% over baseline

---

### EXP-003 — Architecture search: d_model=64
- **Date:** TBD
- **Run by:** 
- **Command:**
  ```bash
  python src/train.py --vocab 100 --epochs 50 --d_model 64 --n_layers 3
  ```
- **Results:**
  - Val Top-1: —
  - CPU Latency: —
- **Notes:** Faster but likely lower accuracy — add to Pareto curve

---

### EXP-004 — Architecture search: d_model=256
- **Date:** TBD
- **Run by:** 
- **Command:**
  ```bash
  python src/train.py --vocab 100 --epochs 50 --d_model 256 --n_layers 4
  ```
- **Results:**
  - Val Top-1: —
  - CPU Latency: —
- **Notes:** Heavier model — check if accuracy justifies latency cost

---

### EXP-005 — Teacher model (GPU)
- **Date:** TBD
- **Run by:** 
- **Command:**
  ```bash
  python src/train.py --vocab 100 --epochs 100 --teacher
  ```
- **Config:** d_model=512, nhead=8, n_layers=6, ffn=1024
- **Results:**
  - Val Top-1: —
  - Val Top-5: —
- **Notes:** Reference model — not for deployment, used as distillation teacher

---

### EXP-006 — Knowledge distillation (stretch goal)
- **Date:** TBD
- **Run by:** 
- **Command:**
  ```bash
  python src/distill.py --teacher models/checkpoints/teacher_best.pt --alpha 0.3 --tau 4
  ```
- **Results:**
  - Val Top-1: —
- **Notes:** Only start after EXP-005 is complete and EXP-002 >= 65%

---

## ONNX Latency Benchmarks

| Model config | Top-1 | Latency (mean) | Latency (p95) | Size |
|-------------|-------|----------------|---------------|------|
| d=64, l=3   | —     | —              | —             | —    |
| d=128, l=3  | —     | —              | —             | —    |
| d=256, l=4  | —     | —              | —             | —    |
| Teacher     | —     | GPU only       | GPU only      | —    |

*This table feeds directly into the Pareto curve figure.*
