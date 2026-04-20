# Pipeline

```
Webcam / Video
      ↓
MediaPipe HandLandmarker  (~8 ms/frame)
      ↓
  (T, 126) keypoints       21 landmarks × 3 (xyz) × 2 hands
      ↓
Temporal Transformer       ~5 ms inference on CPU
      ↓
   ASL Word
```

MediaPipe extracts hand skeleton — **no raw video reaches the model**.

---

# Keypoint Extraction

Each video frame → **126-dimensional vector**

```
Left hand:   21 landmarks × (x, y, z) = 63 floats
Right hand:  21 landmarks × (x, y, z) = 63 floats
─────────────────────────────────────────────────
Total:                                  126 floats/frame
```

Stored as `.npy` files of shape `(T, 126)` — one per video clip.

---

# Augmentations

Applied during training only.

| Augmentation | Description | Effect |
|---|---|---|
| Horizontal flip | Swap left/right hands, mirror x | Doubles effective dataset size |
| Temporal jitter | Randomly drop or repeat frames | Robustness to dropped frames |
| Gaussian noise | σ=0.01 on all coordinates | Simulates MediaPipe detection noise |
| Wrist normalization | Subtract dominant wrist position | Translation invariance |

---

# CNN Baseline: Local Temporal Patterns

A lightweight convolutional baseline to establish a lower bound on model complexity.

<div class="grid grid-cols-2 gap-6 mt-6">
<div>

**Architecture**

```
Input (B, T, 126)
    ↓
Linear projection → d=128
    ↓
4 × Conv1d residual blocks
  (kernel=3, causal padding)
    ↓
Linear classifier → C
    ↓
Output (B, C) or (T, B, C+1)
```

</div>
<div>

**Key Properties**

- **Receptive field**: 3×7 = ~21 frames max
- **Parameters**: ~560K
- **CPU inference**: ~2–3ms (fastest)
- **No long-range memory**: Cannot model sign endpoints 2+ seconds away

</div>
</div>

---

# CNN Baseline: Trade-offs

<div class="grid grid-cols-3 gap-4 mt-6">
<div class="card">

**✓ Strengths**

- Fastest CPU inference
- Captures local motion
- Simplest to debug

</div>
<div class="card">

**✗ Weaknesses**

- **No long-range context**
- Brittle to variable signing speed
- ~10–15% lower accuracy than Transformer

</div>
<div class="card">

**Empirical Result**

Top-1 accuracy: **~62%** on 1,896-class vocab

(Transformer: ~72%)

</div>
</div>

<div class="quote mt-6">
**Lesson**: Convolutional locality is fundamentally limiting for sign language. Signs are defined by temporal *dynamics* — the relationship between frame 1 and frame 80 matters.
</div>

---

# LSTM Baseline: Sequential Hidden State

Bidirectional LSTM: maintains a hidden state across all frames, no positional encoding needed.

<div class="grid grid-cols-2 gap-6 mt-6">
<div>

**Architecture**

```
Input (B, T, 126)
    ↓
2 × stacked BiLSTM
  (hidden=128 → 256 bidirectional)
    ↓
Dropout
    ↓
Linear classifier → C
    ↓
Output (B, C) or (T, B, C+1)
```

</div>
<div>

**Key Properties**

- **Memory span**: All frames — implicit gate weighting
- **Parameters**: ~560K
- **CPU inference**: ~3–5ms (middle ground)
- **Bidirectional**: Forward + backward context

</div>
</div>

---

# LSTM Baseline: Trade-offs

<div class="grid grid-cols-3 gap-4 mt-6">
<div class="card">

**✓ Strengths**

- **Full temporal span**
- No positional encoding
- 3× faster than Transformer
- Implicit gate weighting

</div>
<div class="card">

**✗ Weaknesses**

- Hidden state bottleneck
- Weaker long-range modeling
- Sequential computation
- ~5–8% lower accuracy

</div>
<div class="card">

**Empirical Result**

Top-1 accuracy: **~67–68%** on 1,896-class vocab

(CNN: ~62%, Transformer: ~72%)

</div>
</div>

<div class="quote mt-6">
**Insight**: BiLSTM bridges the gap—full temporal coverage but weaker attention than Transformer. Good efficiency-accuracy trade-off.
</div>
