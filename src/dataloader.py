"""
src/dataloader.py

WLASLDataset and DataLoader factory.
Keypoint extraction lives in scripts/preprocess.py.
Augmentations live in src/augmentations.py.

Usage:
    from src.dataloader import get_dataloader
    loader = get_dataloader(split='train', vocab_size=2000, batch_size=32)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import KEYPOINT_DIM, MAX_SEQ_LEN, DATA_PROC_DIR
from src.augmentations import augment_keypoints


class WLASLDataset(Dataset):
    """
    PyTorch Dataset for WLASL keypoint sequences.

    Args:
        split:      'train', 'val', or 'test'
        vocab_size: number of sign classes
        max_len:    maximum sequence length in frames (clips are padded/truncated)
        augment:    apply augmentations (training only)
    """

    def __init__(self, split: str = "train", vocab_size: int = 2000,
                 max_len: int = MAX_SEQ_LEN, augment: bool = True):
        self.split   = split
        self.max_len = max_len
        self.augment = augment and (split == "train")

        manifest_path = DATA_PROC_DIR / f"{split}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                f"Run: python scripts/preprocess.py --split {split} --vocab {vocab_size}")

        with open(manifest_path) as f:
            self.samples = json.load(f)

        self.samples = [s for s in self.samples if s["label_idx"] < vocab_size]
        print(f"[{split}] {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        kpts   = np.load(sample["path"]).astype(np.float32)  # (T, 126)

        kpts = augment_keypoints(kpts, training=self.augment)

        T = min(kpts.shape[0], self.max_len)

        if kpts.shape[0] < self.max_len:
            pad  = np.zeros((self.max_len - kpts.shape[0], KEYPOINT_DIM), dtype=np.float32)
            kpts = np.vstack([kpts, pad])
        else:
            kpts = kpts[:self.max_len]

        return {
            "keypoints":    torch.tensor(kpts, dtype=torch.float32),          # (max_len, 126)
            "label":        torch.tensor([sample["label_idx"]], dtype=torch.long),  # (1,)
            "input_length": torch.tensor(T, dtype=torch.long),
            "label_length": torch.tensor(1, dtype=torch.long),
        }


def get_dataloader(split: str = "train", vocab_size: int = 2000,
                   batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    """
    Returns a DataLoader for the given split.

    Example:
        train_loader = get_dataloader('train', vocab_size=2000, batch_size=32)
        for batch in train_loader:
            kpts   = batch['keypoints']   # (B, T, 126)
            labels = batch['label']       # (B, 1)
    """
    dataset = WLASLDataset(split=split, vocab_size=vocab_size,
                           augment=(split == "train"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"))
