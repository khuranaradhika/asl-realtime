# From PyTorch to CPU: Why ONNX?

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

<img src="https://images.unsplash.com/photo-1495592822108-9e6261896da8?w=1600&auto=format&fit=crop&q=60" class="rounded-xl h-28 w-full object-cover mb-4" />

<div class="grid grid-cols-3 gap-4 mt-2">
<div class="card">

**1. Load**

Read `.pt` checkpoint, auto-detect `loss=ce` vs `loss=ctc` from saved args, build matching architecture.

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

# Evaluation

<div class="grid grid-cols-2 gap-8">
<div>

| Model | Top-1 | Top-5 | Params | Notes |
|-------|-------|-------|--------|-------|
| 1D CNN | — | — | ~450K | baseline |
| BiLSTM | — | — | ~735K | baseline |
| Transformer (no aug) | **66.9%** | — | ~452K | current demo model |
| Transformer (aug) | — | — | ~452K | in progress |

</div>
<div>
<img src="https://plus.unsplash.com/premium_photo-1681586126003-2a6d4ba943a2?w=1600&auto=format&fit=crop&q=60" class="rounded-xl h-52 w-full object-cover" />
</div>
</div>

<div class="callout callout-blue mt-4">
ℹ️ Full ablation results in progress — numbers will be updated before the final presentation.
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
<img src="https://images.unsplash.com/photo-1733370446176-cf060c668a28?w=1600&auto=format&fit=crop&q=60" class="rounded-xl h-40 w-full object-cover mb-4" />

### What to watch
- Hand skeleton overlay (purple = left, orange = right)
- Buffer counter filling to 60
- Hands % — must hit ≥50% to trigger inference
- Conf score — model's softmax certainty
- Predicted word in the bottom bar

</div>
<div>

### Run it

<video controls style="width:100%;border-radius:8px;margin-bottom:0.75rem;">
  <source src="/videos/demo.mp4" type="video/mp4" />
  Video coming soon.
</video>

**Model:** `transformer_d128_l3_v1896_noaug`  
**Top-1:** 66.9% · **Latency:** 0.5ms · **Vocab:** 1,896 signs

</div>
</div>

<div class="quote mt-6">
Press Q to quit. Move your hand out of frame for ~1 second to reset between signs.
</div>
