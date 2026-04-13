---
theme: default
title: Real-Time ASL Translation
info: Applied Deep Learning — Final Project, Northeastern University Spring 2026
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Real-Time ASL Translation

Keypoint-based temporal transformer for real-time American Sign Language recognition on CPU

<div class="pt-12">
  <span class="text-gray-400">Applied Deep Learning · Northeastern University · Spring 2026</span>
</div>

---

# Team

| Name | Role |
|------|------|
| Radhika Khurana | Data pipeline, model architecture + training, web server |
| Jian Gao | Dataset sourcing, keypoint extraction + augmentation |
| Hrishikesh Pradhan | Baseline experiments + knowledge distillation |
| Gyula Planky | ONNX export, evaluation + demo |

---

# Problem

Existing ASL recognition models (I3D, SlowFast, VideoMAE) achieve strong accuracy but require **GPU inference** over raw video — unusable on a standard laptop.

<div class="grid grid-cols-2 gap-8 mt-8">
<div>

### Goal
Build a system that:
- Recognizes **1,896 ASL signs** from live webcam
- Runs in **real-time on CPU** (no GPU required)
- Achieves **≥ 60% Top-1** accuracy on held-out val

</div>
<div>

### Key insight
Skip raw video at inference time entirely.

```
Webcam → MediaPipe → 126 floats/frame
                     ↓
               Transformer → word
```

100,000× smaller input than raw video.

</div>
</div>

---
src: ./pages/02-data.md
---

---
src: ./pages/03-architecture.md
---

---
src: ./pages/04-demo.md
---
