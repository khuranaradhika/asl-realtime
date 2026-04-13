# Datasets

We train on a combined corpus of three datasets filtered to **vocab = 1,896**.

| Dataset | Role | Clips | Classes |
|---------|------|-------|---------|
| WLASL | Train + in-distribution eval | ~4,143 train / 1,208 val | 2,000 |
| ASL Citizen | Train only | 1,542 | ~300 |
| Aslense | Train only | 53,933 | 2,208 |
| **Combined** | **Training corpus** | **52,998 train / 5,567 val** | **1,896** |

~28 samples per class on average.

---

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
