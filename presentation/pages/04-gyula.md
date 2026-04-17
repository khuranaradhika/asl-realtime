# From PyTorch to CPU: Why ONNX? <span style="font-size:0.5rem;opacity:0.4;vertical-align:super;">G</span>

Training produces a `.pt` file that requires the full PyTorch stack. ONNX decouples the model from the framework entirely.

<div class="grid grid-cols-2 gap-6 mt-8">
<div class="card">

**`.pt` checkpoint**
- Requires PyTorch (~500MB)
- GPU assumed
- Research use only

</div>
<div class="card">

**`.onnx` model**
- Only needs `onnxruntime` (~10MB)
- CPU-optimized by default
- Runs on any laptop, kiosk, Raspberry Pi
- Zero internet dependency at runtime

</div>
</div>

<div class="quote mt-8">
The entire model file is 3MB. It runs offline. No GPU, no PyTorch, no cloud — just a binary and a webcam.
</div>

---

# Export Pipeline: Checkpoint → ONNX

`src/export.py` converts the trained model and benchmarks it automatically.

<div class="grid grid-cols-3 gap-4 mt-2">
<div class="card">

**1. Load**

Read `.pt` checkpoint, auto-detect architecture (`d_model`, `n_layers`) and loss type from saved args.

</div>
<div class="card">

**2. Export**

`torch.onnx.export` with dynamic axes for batch + sequence length. Opset 17/18, constant folding enabled.

</div>
<div class="card">

**3. Benchmark**

200 inference runs on CPU. Reports mean, median, p95, p99 latency before deployment.

</div>
</div>

<div class="grid grid-cols-3 gap-4 mt-8 text-center">
<div>
<div class="stat-number">0.5ms</div>
<div class="stat-label">Median inference latency</div>
</div>
<div>
<div class="stat-number">3MB</div>
<div class="stat-label">ONNX model file size</div>
</div>
<div>
<div class="stat-number">&lt;25ms</div>
<div class="stat-label">End-to-end pipeline</div>
</div>
</div>

---

# Results: Full Ablation

All models trained on the same 1,896-class combined corpus. Best checkpoint selected by validation Top-1.

| Model | Top-1 | Top-5 | Params | Notes |
|-------|-------|-------|--------|-------|
| 1D CNN | 38.4% | 61.1% | ~450K | local 3-frame windows |
| BiLSTM | 31.5% | 53.1% | ~735K | sequential hidden state |
| Transformer (d=128, aug) | 40.1% | 58.9% | ~452K | augmentation hurt — likely underfit |
| Transformer (d=256, aug) | 44.0% | 59.8% | ~452K | larger model, same issue |
| **Transformer (d=128, no aug)** | **67.0%** | **88.6%** | **~452K** | **← demo model** |

<div class="callout callout-blue mt-4">
Augmentation consistently degraded performance at 50 epochs — the model needs significantly more training to benefit from it. No-aug converges faster and stronger.
</div>

---

# What the Numbers Actually Look Like

The 67% headline is averaged across all 1,896 classes — including rare signs with only 1–2 training examples. The model's accuracy is remarkably stable across vocabulary sizes.

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

| Vocabulary Size | Top-1 Accuracy |
|----------------|---------------|
| Top 100 | 68.3% |
| Top 200 | 68.5% |
| Top 300 | 69.8% |
| Top 500 | 69.9% |
| Top 1,000 | 66.9% |
| Top 1,500 | 67.0% |
| All 1,896 | 67.0% |

</div>
<div>

**Accuracy distribution across all classes**

| Band | Classes | % of vocab |
|------|---------|------------|
| 100% | 653 | 34.4% |
| 60–99% | 702 | 37.0% |
| 20–59% | 405 | 21.4% |
| 0% | 136 | 7.2% |

<div class="callout callout-blue mt-4">
71.5% of classes score above 60%. Only 7.2% sit at 0% — nearly all low-frequency signs with fewer than 3 training examples.
</div>

</div>
</div>

---

# Results in Context

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

| Model | Top-1 | Hardware | Input |
|-------|-------|----------|-------|
| I3D | ~60% | GPU | Raw video |
| VideoMAE | ~65% | GPU | Raw video |
| SPOTER | ~60% | GPU | Keypoints |
| **Ours (no aug)** | **67.0%** | **CPU** | **Keypoints** |

</div>
<div>

### The key distinction

Every competitive model requires a GPU and operates on raw video frames. Ours is the only model in this range that:

- Runs entirely on CPU
- Uses keypoints only — no raw video stored
- Operates in real time at 30fps
- Fits in 3MB

</div>
</div>

<div class="quote mt-6">
67% on CPU with keypoints — matching or beating GPU video models while running on a laptop webcam.
</div>

---

# Real-Time Demo Architecture

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### The pipeline
1. Webcam captures frame at 30fps
2. MediaPipe extracts 126 keypoints (~8ms)
3. Keypoints pushed into a 60-frame rolling buffer
4. Every 0.5s — if buffer is full — ONNX model infers (~0.5ms)
5. Softmax confidence checked against threshold
6. Prediction added to 5-inference majority vote
7. Predicted word shown on screen

</div>
<div>

### Stability guards
- **Hand ratio gate** — ≥50% of buffer frames must have hands detected
- **Confidence threshold** — tunable at runtime via `--threshold`
- **Reset on absence** — 20 frames without hands clears history
- **Majority vote** — smooths over 5 consecutive inferences

</div>
</div>

---

# Live Demo

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### What to watch
- Hand skeleton overlay (navy = left, azure = right)
- Buffer counter filling to 60
- Hands % — must hit ≥50% to trigger inference
- Confidence bar — model's softmax certainty
- Predicted word centered on screen

</div>
<div>

<video controls style="width:100%;border-radius:8px;margin-bottom:0.75rem;">
  <source src="/videos/demo.mp4" type="video/mp4" />
</video>

**Model:** `transformer_d128_l3_v1896_noaug`  
**Top-1:** 67.0% · **Latency:** 0.5ms · **Vocab:** 1,896 signs

</div>
</div>

<div class="quote mt-6">
Press Q to quit. Move your hand out of frame for ~1 second to reset between signs.
</div>
