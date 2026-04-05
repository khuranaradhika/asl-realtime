"""
src/dataloader.py

Dataset and DataLoader for WLASL + MS-ASL keypoint sequences.
Handles MediaPipe keypoint extraction, augmentation, and batching.

Usage:
    # Extract WLASL keypoints (run train first to build vocab):
    python src/dataloader.py --extract --split train --vocab 100
    python src/dataloader.py --extract --split val   --vocab 100
    python src/dataloader.py --extract --split test  --vocab 100

    # Extract MS-ASL keypoints (after downloading videos):
    python src/dataloader.py --extract-msasl --split train --vocab 100
    python src/dataloader.py --extract-msasl --split val   --vocab 100

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

KEYPOINT_DIM      = 126   # 21 left hand pts × 3 + 21 right hand pts × 3
MAX_SEQ_LEN       = 150   # frames — clips longer than this are truncated
DATA_RAW_DIR      = Path("data/raw/wlasl")
DATA_RAW_MSASL    = Path("data/raw/msasl")
DATA_PROC_DIR     = Path("data/processed")


# ─── Keypoint extraction (run once as preprocessing) ─────────────────────────

def _get_hand_detector():
    import urllib.request
    import mediapipe as mp
    model_path = Path("data/hand_landmarker.task")
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print(f"Downloading MediaPipe hand model...")
        urllib.request.urlretrieve(url, str(model_path))
        print("Done.")
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def extract_keypoints_from_video(video_path: str) -> np.ndarray:
    """
    Run MediaPipe HandLandmarker on every frame of a video and return
    a (T, 126) array of hand keypoints. Uses Tasks API (mediapipe 0.10.30+).
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise ImportError("pip3 install mediapipe opencv-python")

    detector = _get_hand_detector()
    cap = cv2.VideoCapture(video_path)
    frames_kpts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        lh = np.zeros(63, dtype=np.float32)
        rh = np.zeros(63, dtype=np.float32)
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name
            coords = np.array([[lm.x, lm.y, lm.z]
                                for lm in hand_landmarks],
                               dtype=np.float32).flatten()
            if handedness == "Left":
                lh = coords
            else:
                rh = coords
        frames_kpts.append(np.concatenate([lh, rh]))

    cap.release()
    detector.close()
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

    # Build vocabulary from train split only — reuse for val/test
    vocab_path = DATA_PROC_DIR / "vocab.json"
    if split == "train":
        sign_counts = {}
        for entry in data:
            for inst in entry["instances"]:
                if inst["split"] == "train":
                    sign_counts[entry["gloss"]] = sign_counts.get(entry["gloss"], 0) + 1
        top_signs = sorted(sign_counts, key=sign_counts.get, reverse=True)[:vocab_size]
        vocab     = {sign: idx for idx, sign in enumerate(top_signs)}
        with open(vocab_path, "w") as f:
            json.dump(vocab, f, indent=2)
        print(f"Vocabulary built from train: {len(vocab)} signs")
    else:
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"vocab.json not found. Run --extract --split train first.")
        with open(vocab_path) as f:
            vocab = json.load(f)
        print(f"Vocabulary loaded from train: {len(vocab)} signs")

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
            video_path = str(raw_dir / "videos" / f"{video_id}.mp4")
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
    lh_orig = kpts[:, :63].copy()
    rh_orig = kpts[:, 63:].copy()
    flipped = kpts.copy()
    # new left hand = mirrored original right hand
    flipped[:, :63] = rh_orig
    flipped[:, :63][:, 0::3] = 1.0 - rh_orig[:, 0::3]
    # new right hand = mirrored original left hand
    flipped[:, 63:] = lh_orig
    flipped[:, 63:][:, 0::3] = 1.0 - lh_orig[:, 0::3]
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
    Uses right wrist if detected, falls back to left wrist, skips if neither.
    Makes representation invariant to signer position in frame.
    """
    normalized = kpts.copy()
    rh_wrist = kpts[:, 63:66]  # (T, 3) right wrist
    lh_wrist = kpts[:, 0:3]    # (T, 3) left wrist

    # Per-frame: pick right wrist if non-zero, else left wrist
    rh_detected = (rh_wrist != 0).any(axis=1)  # (T,)
    lh_detected = (lh_wrist != 0).any(axis=1)  # (T,)

    wrist = np.zeros((kpts.shape[0], 3), dtype=np.float32)
    wrist[rh_detected]                          = rh_wrist[rh_detected]
    wrist[~rh_detected & lh_detected]           = lh_wrist[~rh_detected & lh_detected]

    for i in range(0, KEYPOINT_DIM, 3):
        normalized[:, i]     -= wrist[:, 0]  # x
        normalized[:, i + 1] -= wrist[:, 1]  # y
    return normalized


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WLASLDataset(Dataset):
    """
    PyTorch Dataset for WLASL (and optionally combined WLASL+MS-ASL) keypoints.

    Args:
        split:      'train', 'val', or 'test'
        vocab_size: number of sign classes
        max_len:    maximum sequence length in frames (clips are padded/truncated)
        augment:    apply augmentations (training only)
        combined:   if True, load combined_<split>_manifest.json (WLASL + MS-ASL)
    """

    def __init__(self, split: str = "train", vocab_size: int = 100,
                 max_len: int = MAX_SEQ_LEN, augment: bool = True,
                 combined: bool = False):
        self.split   = split
        self.max_len = max_len
        self.augment = augment and (split == "train")

        if combined:
            manifest_path = DATA_PROC_DIR / f"combined_{split}_manifest.json"
            label = "combined"
        else:
            manifest_path = DATA_PROC_DIR / f"{split}_manifest.json"
            label = "wlasl"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                f"Run: python src/dataloader.py --extract --split {split}"
                + (" then --extract-msasl" if combined else ""))

        with open(manifest_path) as f:
            self.samples = json.load(f)

        self.samples = [s for s in self.samples if s["label_idx"] < vocab_size]
        print(f"[{split}/{label}] {len(self.samples)} samples loaded")

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
                   batch_size: int = 32, num_workers: int = 4,
                   augment: bool = True, combined: bool = False) -> DataLoader:
    """
    Returns a DataLoader for the given split.

    Args:
        augment:  whether to apply training augmentations. Only applies to
                  train split — val/test are never augmented. Set to False
                  to run a no-augmentation baseline (EXP-001).
        combined: if True, load combined WLASL+MS-ASL manifest.

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
        augment=(split == "train") and augment,
        combined=combined)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"))


# ─── MS-ASL extraction ────────────────────────────────────────────────────────

def preprocess_msasl(split: str = "train", vocab_size: int = 100):
    """
    Extract keypoints for all MS-ASL clips in a split and append to the
    combined manifest. Reuses the vocab built from WLASL train.

    MS-ASL video layout: data/raw/msasl/videos/{split}/{idx:05d}.mp4
    MS-ASL annotation:   data/raw/msasl/MSASL_{split}.json
      fields used: clean_text (label word), idx (video file index)
    """
    vocab_path = DATA_PROC_DIR / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            "vocab.json not found. Run --extract --split train (WLASL) first.")
    with open(vocab_path) as f:
        vocab = json.load(f)
    print(f"Vocabulary loaded: {len(vocab)} signs")

    anno_path = DATA_RAW_MSASL / f"MSASL_{split}.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"MS-ASL annotation not found at {anno_path}")
    with open(anno_path) as f:
        entries = json.load(f)

    proc_dir = DATA_PROC_DIR / f"msasl_{split}"
    proc_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped_no_vocab = 0
    skipped_no_video = 0

    for idx, entry in enumerate(tqdm(entries, desc=f"Extracting MS-ASL {split}")):
        word = entry.get("clean_text", "").lower().strip()
        if word not in vocab:
            skipped_no_vocab += 1
            continue

        video_path = DATA_RAW_MSASL / "videos" / split / f"{idx:05d}.mp4"
        if not video_path.exists():
            skipped_no_video += 1
            continue

        save_path = proc_dir / f"{idx:05d}.npy"
        if not save_path.exists():
            kpts = extract_keypoints_from_video(str(video_path))
            np.save(str(save_path), kpts)

        manifest.append({
            "path":      str(save_path),
            "label":     word,
            "label_idx": vocab[word],
            "source":    "msasl",
        })

    msasl_manifest_path = DATA_PROC_DIR / f"msasl_{split}_manifest.json"
    with open(msasl_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"MS-ASL {split}: {len(manifest)} samples saved")
    print(f"  Skipped (not in vocab): {skipped_no_vocab}")
    print(f"  Skipped (no video):     {skipped_no_video}")

    # Merge with WLASL manifest into combined manifest
    wlasl_path = DATA_PROC_DIR / f"{split}_manifest.json"
    combined   = []
    if wlasl_path.exists():
        with open(wlasl_path) as f:
            wlasl_entries = json.load(f)
        for e in wlasl_entries:
            e.setdefault("source", "wlasl")
        combined = wlasl_entries

    combined += manifest

    combined_path = DATA_PROC_DIR / f"combined_{split}_manifest.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined {split} manifest: {len(combined)} samples → {combined_path}")


# ─── CLI for preprocessing ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",       action="store_true",
                        help="Extract WLASL keypoints from raw videos")
    parser.add_argument("--extract-msasl", action="store_true",
                        help="Extract MS-ASL keypoints and merge with WLASL manifest")
    parser.add_argument("--split",   type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--vocab",   type=int, default=100)
    args = parser.parse_args()

    if args.extract:
        preprocess_dataset(split=args.split, vocab_size=args.vocab)
    elif args.extract_msasl:
        preprocess_msasl(split=args.split, vocab_size=args.vocab)
    else:
        # Quick sanity check
        loader = get_dataloader(args.split, vocab_size=args.vocab, batch_size=4)
        batch  = next(iter(loader))
        print(f"keypoints: {batch['keypoints'].shape}")
        print(f"labels:    {batch['label'].shape}")
        print(f"lengths:   {batch['input_length']}")