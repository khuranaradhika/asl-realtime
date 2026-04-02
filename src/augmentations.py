"""
src/augmentations.py

Keypoint augmentation transforms for training.
Called by WLASLDataset — not a runtime concern of the dataloader itself.
"""

import numpy as np
from src.config import KEYPOINT_DIM


def augment_keypoints(kpts: np.ndarray, training: bool = True) -> np.ndarray:
    """
    Apply training augmentations to a (T, 126) keypoint array.

    Augmentations:
        1. Horizontal flip  — swap left/right hands (doubles effective dataset)
        2. Temporal jitter  — randomly drop or repeat frames
        3. Gaussian noise   — simulate MediaPipe detection noise
        4. Wrist-relative normalization — translation + scale invariance
    """
    if not training:
        return normalize_keypoints(kpts)

    if np.random.rand() < 0.5:
        kpts = flip_keypoints(kpts)

    kpts = temporal_jitter(kpts, jitter_prob=0.1)
    kpts = kpts + np.random.randn(*kpts.shape).astype(np.float32) * 0.01
    kpts = normalize_keypoints(kpts)

    return kpts


def flip_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Mirror left/right hands: swap first 63 and last 63 features, flip x."""
    flipped = kpts.copy()
    lh = kpts[:, :63].copy()
    rh = kpts[:, 63:].copy()
    rh[:, 0::3] = 1.0 - rh[:, 0::3]
    lh[:, 0::3] = 1.0 - lh[:, 0::3]
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
            continue
        elif r < jitter_prob and t > 0:
            result.append(kpts[t - 1])
        result.append(kpts[t])
    return np.stack(result) if result else kpts


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints relative to the dominant (right) wrist position.
    Makes representation invariant to signer position in frame.
    """
    wrist      = kpts[:, 63:66].copy()  # right wrist xyz — (T, 3)
    normalized = kpts.copy()
    for i in range(0, KEYPOINT_DIM, 3):
        normalized[:, i]     -= wrist[:, 0]  # x
        normalized[:, i + 1] -= wrist[:, 1]  # y
    return normalized
