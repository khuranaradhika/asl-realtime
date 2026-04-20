
# Team

| Name | Role |
|------|------|
| Radhika Khurana | Data pipeline, model architecture + training, web server |
| Jian Gao | Dataset sourcing, keypoint extraction + augmentation |
| Hrishikesh Pradhan | Baseline experiments + knowledge distillation |
| Gyula Planky | ONNX export, evaluation + demo |

---

# The Problem: 70 Million People. No Accessible AI.

<div class="grid grid-cols-2 gap-8">
<div>

Existing ASL recognition models (I3D, SlowFast, VideoMAE) achieve strong accuracy but require **GPU inference** over raw video — unusable on a standard laptop.

<div class="grid grid-cols-2 gap-4 mt-6">
<div class="card">

**GPU Dependency**

Top models require dedicated NVIDIA hardware

</div>
<div class="card">

**6.2M Pixels/Frame**

Computationally expensive and privacy-invasive

</div>
<div class="card">

**High Latency**

Processing delays destroy conversational flow

</div>
<div class="card">

**No Deployment Path**

Gap between research and real-world use

</div>
</div>

</div>
<div>
<img src="https://images.unsplash.com/photo-1651230868959-d982b7b4945c?w=1600&auto=format&fit=crop&q=60" class="rounded-xl h-80 w-full object-cover" />
</div>
</div>

---

# Our Insight: The Pixels Don't Matter. The Hands Do.

MediaPipe extracts **21 landmarks per hand × 3 coordinates = 126 numbers per frame** — versus 6,220,800 pixels. A **49,371× reduction** in data volume. No raw video stored at any point.

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="card">

**Raw Video**

<span class="text-red-500 font-semibold">6,220,800 pixels/frame</span>

GPU required · Privacy risk

</div>
<div class="card">

**Keypoints**

<span class="text-teal-600 font-semibold">126 floats/frame</span>

CPU-ready · Private by design

</div>
</div>

<div class="quote mt-6">
The result: same semantic information at a fraction of the computational cost. No background, no lighting, no camera angle — just hand geometry.
</div>

---

# System Pipeline: Four Steps, Under 25ms

<div class="grid grid-cols-4 gap-3 mt-6">
<div class="card text-center">
<div class="font-semibold">Webcam 30fps</div>
<div class="text-gray-400 text-xs mt-1">Raw video frames</div>
</div>
<div class="card text-center">
<div class="font-semibold">MediaPipe</div>
<div class="text-gray-400 text-xs mt-1">126 keypoints · ~8ms</div>
</div>
<div class="card text-center">
<div class="font-semibold">Buffer T×126</div>
<div class="text-gray-400 text-xs mt-1">Sequence · &lt;1ms</div>
</div>
<div class="card text-center">
<div class="font-semibold">Transformer</div>
<div class="text-gray-400 text-xs mt-1">ONNX · ~0.5ms</div>
</div>
</div>

<div class="grid grid-cols-3 gap-5 mt-6">
<div class="card">

**CPU Only**

No GPU required — runs on any modern consumer device

</div>
<div class="card">

**Offline Inference**

Zero network dependency at runtime — works anywhere

</div>
<div class="card">

**&lt;25ms Latency**

Real conversation speed — 8ms extraction + 0.5ms inference

</div>
</div>

---
theme: default
title: Real-Time ASL Translation
info: Applied Deep Learning — Final Project, Northeastern University Spring 2026
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Real-Time ASL Translation

<div class="grid grid-cols-2 gap-8">
<div class="pt-4">

Keypoint-based temporal transformer for real-time American Sign Language recognition on CPU

<div class="pt-8 text-gray-500">
Radhika Khurana · Jian Gao · Hrishikesh Pradhan · Gyula Planky
</div>
<div class="pt-2 text-gray-400 text-sm">
Applied Deep Learning · Northeastern University · Spring 2026
</div>

</div>
<div>
<img src="https://images.unsplash.com/photo-1592530392525-9d8469678dac?w=1600&auto=format&fit=crop&q=60" class="rounded-xl h-64 w-full object-cover" />
</div>
</div>

---


# Datasets

We train on a combined corpus of three datasets filtered to **vocab = 1,896**.

| Dataset | Role | Clips | Classes |
|---------|------|-------|---------|
| WLASL | Train + in-distribution eval | ~4,143 train / 1,208 val | 2,000 |
| ASL Citizen | Train only | 1,542 | ~300 |
| Aslense | Train only | 53,933 | 2,208 |
| **Combined** | **Training corpus** | **52,998 train / 5,567 val** | **1,896** |

~28 samples per class on average.
