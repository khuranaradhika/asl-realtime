"""
src/config.py

Shared constants and paths used across the project.
All other modules import from here — no more scattered magic numbers.
"""

from pathlib import Path

# ── Keypoint dimensions ────────────────────────────────────────────────────────
KEYPOINT_DIM = 126      # 21 left-hand landmarks × 3 + 21 right-hand landmarks × 3
MAX_SEQ_LEN  = 150      # frames — clips longer than this are truncated at training time

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_RAW_DIR   = Path("data/raw/wlasl")
DATA_PROC_DIR  = Path("data/processed")
CHECKPOINT_DIR = Path("models/checkpoints")

# ── Demo ───────────────────────────────────────────────────────────────────────
WINDOW_FRAMES = 60      # frames to buffer before running inference
SMOOTH_WINDOW = 5       # smooth predictions over this many inference calls
