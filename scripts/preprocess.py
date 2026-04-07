"""
scripts/preprocess.py

One-time offline preprocessing: extract MediaPipe hand keypoints from raw
WLASL videos and save as .npy files with a split manifest. Run from project root:

    python scripts/preprocess.py --split train --vocab 2000
    python scripts/preprocess.py --split val   --vocab 2000
    python scripts/preprocess.py --split test  --vocab 2000
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import cv2

from src.config import DATA_RAW_DIR, DATA_PROC_DIR, KEYPOINT_DIM
from src.keypoints import get_hand_detector, extract_keypoints_from_frame


def extract_keypoints_from_video(video_path: str, detector) -> np.ndarray:
    """Run keypoint extraction on every frame of a video. Returns (T, 126) array."""
    cap         = cv2.VideoCapture(video_path)
    frames_kpts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kpts, _   = extract_keypoints_from_frame(rgb, detector)
        frames_kpts.append(kpts)

    cap.release()
    return np.stack(frames_kpts) if frames_kpts else np.zeros((1, KEYPOINT_DIM), dtype=np.float32)


def preprocess_dataset(split: str = "train", vocab_size: int = 100):
    """Extract keypoints for all videos in a split and save as .npy files."""
    proc_dir = DATA_PROC_DIR / split
    proc_dir.mkdir(parents=True, exist_ok=True)

    anno_path = DATA_RAW_DIR / "WLASL_v0.3.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"WLASL annotation not found at {anno_path}")

    with open(anno_path) as f:
        data = json.load(f)

    # Build vocabulary — top vocab_size most frequent signs across all splits
    sign_counts = {}
    for entry in data:
        for inst in entry["instances"]:
            sign_counts[entry["gloss"]] = sign_counts.get(entry["gloss"], 0) + 1

    top_signs = sorted(sign_counts, key=sign_counts.get, reverse=True)[:vocab_size]
    vocab     = {sign: idx for idx, sign in enumerate(top_signs)}

    with open(DATA_PROC_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary: {len(vocab)} signs")

    detector = get_hand_detector()
    manifest = []

    for entry in tqdm(data, desc=f"Extracting {split}"):
        gloss = entry["gloss"]
        if gloss not in vocab:
            continue
        label_idx = vocab[gloss]

        for inst in entry["instances"]:
            if inst["split"] != split:
                continue
            video_id   = inst["video_id"]
            video_path = str(DATA_RAW_DIR / "videos" / f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue

            save_path = proc_dir / f"{video_id}.npy"
            if not save_path.exists():
                kpts = extract_keypoints_from_video(video_path, detector)
                np.save(str(save_path), kpts)

            manifest.append({
                "path":      str(save_path),
                "label":     gloss,
                "label_idx": label_idx,
            })

    detector.close()

    with open(DATA_PROC_DIR / f"{split}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{split}: {len(manifest)} samples saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--vocab", type=int, default=2000)
    args = parser.parse_args()
    preprocess_dataset(split=args.split, vocab_size=args.vocab)
