# Models Compared

Three temporal modeling approaches on the same data and training setup.

| Model | How it sees time | Params | Loss |
|-------|-----------------|--------|------|
| 1D CNN | Local 3-frame windows | ~450K | Cross-entropy |
| BiLSTM | Sequential hidden state | ~735K | Cross-entropy |
| **Transformer** | Full self-attention across all frames | ~452K | Cross-entropy |

All use **label smoothing (0.1)** + global average pooling for isolated word classification.

---

# Transformer Architecture

```
Input: (B, T, 126) keypoint sequences
  → Linear projection:  126 → 128    (d_model)
  → Sinusoidal positional encoding
  → 3 × TransformerEncoderLayer
      - 4 attention heads
      - FFN dim: 256
      - Pre-norm (norm_first=True)
      - Dropout: 0.1
  → Global mean pool over non-padded frames
  → Linear classifier: 128 → 1,896
  → Cross-entropy loss
```

**AdamW** · lr=3e-4 · cosine annealing · gradient clipping

---

# Knowledge Distillation

Student model (CPU-deployable) trained to mimic a larger teacher.

<div class="grid grid-cols-2 gap-8">
<div>

### Teacher
- d_model = 512, 6 layers, 8 heads
- ~18M parameters
- GPU only

</div>
<div>

### Student
- d_model = 128, 3 layers, 4 heads
- ~672K parameters
- Runs on CPU in <15ms

</div>
</div>

Student loss = α × CE(student, labels) + (1−α) × KL(student ∥ teacher)

---

# Results

| Model | Top-1 | Top-5 | Params | Notes |
|-------|-------|-------|--------|-------|
| 1D CNN | — | — | ~450K | baseline |
| BiLSTM | — | — | ~735K | baseline |
| Transformer (no aug) | — | — | ~452K | ablation |
| Transformer | **40.8%** | **56.4%** | ~452K | 150 epochs, MPS |
| **Distilled student** | — | — | ~672K | **← this model** |

Random baseline: 0.33% (1/1896). Best result = **124× above random**.

State-of-the-art (I3D, VideoMAE) reaches ~60–65% using raw video on GPU.