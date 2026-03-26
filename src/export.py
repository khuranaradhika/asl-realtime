"""
src/export.py

Export a trained SignTransformer checkpoint to ONNX for CPU deployment.

Usage:
    python src/export.py --checkpoint models/checkpoints/student_best.pt
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path

from src.model import build_student_model, make_padding_mask


def export_to_onnx(checkpoint_path: str, output_path: str = "models/sign_model.onnx",
                   vocab_size: int = 100):
    device = torch.device("cpu")

    # Load model
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_student_model(n_classes=vocab_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: Top-1 = {ckpt.get('top1', 'N/A'):.3f}")

    # Dummy input
    dummy_kpts = torch.randn(1, 80, 126)
    dummy_mask = torch.zeros(1, 80, dtype=torch.bool)

    # Export
    torch.onnx.export(
        model,
        (dummy_kpts, dummy_mask),
        output_path,
        input_names=["keypoints", "padding_mask"],
        output_names=["log_probs"],
        dynamic_axes={
            "keypoints":    {0: "batch", 1: "seq_len"},
            "padding_mask": {0: "batch", 1: "seq_len"},
            "log_probs":    {0: "seq_len", 1: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"Exported to {output_path} ({size_mb:.1f} MB)")

    # Benchmark latency
    benchmark_latency(output_path, n_runs=200)


def benchmark_latency(onnx_path: str, n_runs: int = 200, seq_len: int = 80):
    """Measure end-to-end inference latency on CPU."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    kpts = np.random.randn(1, seq_len, 126).astype(np.float32)
    mask = np.zeros((1, seq_len), dtype=bool)

    # Warmup
    for _ in range(10):
        sess.run(["log_probs"], {"keypoints": kpts, "padding_mask": mask})

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(["log_probs"], {"keypoints": kpts, "padding_mask": mask})
        times.append((time.perf_counter() - t0) * 1000)

    times = sorted(times)
    print(f"\nLatency over {n_runs} runs (seq_len={seq_len}):")
    print(f"  Mean:   {np.mean(times):.1f} ms")
    print(f"  Median: {np.median(times):.1f} ms")
    print(f"  p95:    {np.percentile(times, 95):.1f} ms")
    print(f"  p99:    {np.percentile(times, 99):.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output",     type=str, default="models/sign_model.onnx")
    parser.add_argument("--vocab",      type=int, default=100)
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output, args.vocab)
