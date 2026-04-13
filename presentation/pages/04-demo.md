# Live Demo

Real-time webcam inference — no GPU required.

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### What you'll see
- Live webcam feed
- Hand skeleton overlay (left = purple, right = orange)
- Predicted ASL word at bottom of frame
- FPS counter + 60-frame buffer progress

</div>
<div>

### How it works
1. MediaPipe extracts keypoints per frame (~8ms)
2. 60-frame rolling buffer fills
3. Distilled transformer infers every 0.5s (~5ms)
4. Prediction smoothed over last 5 inferences

</div>
</div>

---

# Running the Demo

**Step 1 — Export to ONNX**

```bash
python -m src.export \
  --checkpoint models/checkpoints/distill_s128l3_a5_t4_v1896_combined_best.pt \
  --vocab 1896 \
  --output models/sign_model.onnx
```

**Step 2 — Launch**

```bash
python src/demo.py --model models/sign_model.onnx --vocab 1896
```

Press **Q** to quit.

---

# Takeaways

- Keypoint-based approach makes real-time CPU inference **viable**
- 126 floats/frame vs 1920×1080×3 pixels — 100,000× smaller input
- Transformer self-attention outperforms CNN and BiLSTM on temporal sign data
- Knowledge distillation preserves accuracy while shrinking the model for deployment
- **Next steps:** continuous signing (CTC), more signer diversity, language model post-processing
