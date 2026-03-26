"""
src/dataloader.py

WLASLDataset — PyTorch Dataset and DataLoader for WLASL keypoint sequences.
Handles MediaPipe keypoint extraction, augmentation, and batching.

Usage:
    # Extract keypoints from raw videos first:
    python src/dataloader.py --extract --split train

    # Then use in training:
    from src.dataloader import get_dataloader
    loader = get_dataloader(split='train', vocab_size=100, batch_size=32)
"""

import os
import json
import math
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────────

KEYPOINT_DIM   = 126   # 21 left hand pts × 3 + 21 right hand pts × 3
MAX_SEQ_LEN    = 150   # frames — clips longer than this are truncated
DATA_RAW_DIR   = Path("data/raw/wlasl")
DATA_PROC_DIR  = Path("data/processed")


# ─── Keypoint extraction (run once as preprocessing) ─────────────────────────

def extract_keypoints_from_video(video_path: str) -> np.ndarray:
    """
    Run MediaPipe Holistic on every frame of a video and return
    a (T, 126) array of hand keypoints.

    Missing hands are filled with zeros.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise ImportError("pip install mediapipe opencv-python")

    mp_holistic = mp.solutions.holistic
    holistic    = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False)

    cap = cv2.VideoCapture(video_path)
    frames_kpts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        lh = (np.array([[lm.x, lm.y, lm.z]
                         for lm in results.left_hand_landmarks.landmark],
                        dtype=np.float32).flatten()
              if results.left_hand_landmarks else np.zeros(63, dtype=np.float32))

        rh = (np.array([[lm.x, lm.y, lm.z]
                         for lm in results.right_hand_landmarks.landmark],
                        dtype=np.float32).flatten()
              if results.right_hand_landmarks else np.zeros(63, dtype=np.float32))

        frames_kpts.append(np.concatenate([lh, rh]))  # (126,)

    cap.release()
    holistic.close()
    return np.stack(frames_kpts) if frames_kpts else np.zeros((1, KEYPOINT_DIM))


def preprocess_dataset(split: str = "train", vocab_size: int = 100):
    """
    Extract keypoints for all videos in a split and save as .npy files.
    Also writes a manifest JSON mapping each sample to its label.
    """
    raw_dir  = DATA_RAW_DIR
    proc_dir = DATA_PROC_DIR / split
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Load WLASL annotation JSON
    anno_path = raw_dir / "WLASL_v0.3.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"WLASL annotation not found at {anno_path}")

    with open(anno_path) as f:
        data = json.load(f)

    # Build vocabulary — top vocab_size most frequent signs
    sign_counts = {}
    for entry in data:
        for inst in entry["instances"]:
            if inst["split"] == split:
                sign_counts[entry["gloss"]] = sign_counts.get(entry["gloss"], 0) + 1

    top_signs  = sorted(sign_counts, key=sign_counts.get, reverse=True)[:vocab_size]
    vocab      = {sign: idx for idx, sign in enumerate(top_signs)}

    # Save vocabulary
    with open(DATA_PROC_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary: {len(vocab)} signs")

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
            video_path = str(raw_dir / f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue  # unavailable clip — skip silently

            save_path = proc_dir / f"{video_id}.npy"
            if not save_path.exists():
                kpts = extract_keypoints_from_video(video_path)
                np.save(str(save_path), kpts)

            manifest.append({
                "path":      str(save_path),
                "label":     gloss,
                "label_idx": label_idx,
            })

    with open(DATA_PROC_DIR / f"{split}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{split}: {len(manifest)} samples saved")


# ─── Augmentations ────────────────────────────────────────────────────────────

def augment_keypoints(kpts: np.ndarray, training: bool = True) -> np.ndarray:
    """
    Apply training augmentations to a (T, 126) keypoint array.

    Augmentations:
        1. Horizontal flip — swap left/right hands (doubles effective dataset)
        2. Temporal jitter — randomly drop or repeat frames
        3. Gaussian noise — simulate MediaPipe detection noise
        4. Wrist-relative normalization — translation + scale invariance
    """
    if not training:
        return normalize_keypoints(kpts)

    # 1. Horizontal flip (50% chance)
    if np.random.rand() < 0.5:
        kpts = flip_keypoints(kpts)

    # 2. Temporal jitter (20% of frames affected)
    kpts = temporal_jitter(kpts, jitter_prob=0.1)

    # 3. Gaussian noise
    kpts = kpts + np.random.randn(*kpts.shape).astype(np.float32) * 0.01

    # 4. Normalize relative to dominant wrist
    kpts = normalize_keypoints(kpts)

    return kpts


def flip_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Mirror left/right hands: swap first 63 and last 63 features, flip x."""
    flipped = kpts.copy()
    lh = kpts[:, :63].copy()
    rh = kpts[:, 63:].copy()
    # Swap hands and flip x coordinate (index 0, 3, 6, ... of each landmark)
    lh[:, 0::3] = 1.0 - rh[:, 0::3]
    rh[:, 0::3] = 1.0 - kpts[:, 0::3]  # original lh
    flipped[:, :63] = rh
    flipped[:, 63:] = lh
    return flipped


def temporal_jitter(kpts: np.ndarray, jitter_prob: float = 0.1) -> np.ndarray:
    """Randomly drop or repeat individual frames."""
    T = kpts.shape[0]
    result = []
    for t in range(T):
        r = np.random.rand()
        if r < jitter_prob / 2 and len(result) > 0:
            continue  # drop frame
        elif r < jitter_prob and t > 0:
            result.append(kpts[t - 1])  # repeat previous
        result.append(kpts[t])
    return np.stack(result) if result else kpts


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints relative to the dominant wrist position.
    Makes representation invariant to signer position in frame.
    """
    # Right wrist = landmark 0 of right hand (indices 63, 64, 65)
    wrist = kpts[:, 63:66].copy()  # (T, 3)
    normalized = kpts.copy()
    # Subtract wrist position from all x,y coords (not z)
    for i in range(0, KEYPOINT_DIM, 3):
        normalized[:, i]     -= wrist[:, 0]  # x
        normalized[:, i + 1] -= wrist[:, 1]  # y
    return normalized


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WLASLDataset(Dataset):
    """
    PyTorch Dataset for WLASL keypoint sequences.

    Args:
        split:      'train', 'val', or 'test'
        vocab_size: number of sign classes
        max_len:    maximum sequence length in frames (clips are padded/truncated)
        augment:    apply augmentations (training only)
    """

    def __init__(self, split: str = "train", vocab_size: int = 100,
                 max_len: int = MAX_SEQ_LEN, augment: bool = True):
        self.split     = split
        self.max_len   = max_len
        self.augment   = augment and (split == "train")

        manifest_path = DATA_PROC_DIR / f"{split}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                f"Run: python src/dataloader.py --extract --split {split}")

        with open(manifest_path) as f:
            self.samples = json.load(f)

        # Filter to vocab_size if needed
        self.samples = [s for s in self.samples if s["label_idx"] < vocab_size]
        print(f"[{split}] {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        kpts   = np.load(sample["path"]).astype(np.float32)  # (T, 126)

        # Augment
        kpts = augment_keypoints(kpts, training=self.augment)

        # Record actual length before padding
        T = min(kpts.shape[0], self.max_len)

        # Pad or truncate to max_len
        if kpts.shape[0] < self.max_len:
            pad  = np.zeros((self.max_len - kpts.shape[0], KEYPOINT_DIM), dtype=np.float32)
            kpts = np.vstack([kpts, pad])
        else:
            kpts = kpts[:self.max_len]

        return {
            "keypoints":    torch.tensor(kpts, dtype=torch.float32),   # (max_len, 126)
            "label":        torch.tensor([sample["label_idx"]], dtype=torch.long),  # (1,)
            "input_length": torch.tensor(T, dtype=torch.long),
            "label_length": torch.tensor(1, dtype=torch.long),
        }


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloader(split: str = "train", vocab_size: int = 100,
                   batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    """
    Returns a DataLoader for the given split.

    Example:
        train_loader = get_dataloader('train', vocab_size=100, batch_size=32)
        for batch in train_loader:
            kpts   = batch['keypoints']       # (B, T, 126)
            labels = batch['label']           # (B, 1)
            ...
    """
    dataset = WLASLDataset(
        split=split,
        vocab_size=vocab_size,
        augment=(split == "train"))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"))


# ─── CLI for preprocessing ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true",
                        help="Extract keypoints from raw videos")
    parser.add_argument("--split",   type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--vocab",   type=int, default=100)
    args = parser.parse_args()

    if args.extract:
        preprocess_dataset(split=args.split, vocab_size=args.vocab)
    else:
        # Quick sanity check
        loader = get_dataloader(args.split, vocab_size=args.vocab, batch_size=4)
        batch  = next(iter(loader))
        print(f"keypoints: {batch['keypoints'].shape}")
        print(f"labels:    {batch['label'].shape}")
        print(f"lengths:   {batch['input_length']}")
