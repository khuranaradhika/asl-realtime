"""
scripts/overnight_pipeline.py

Safe full rebuild pipeline. Preserves ALL existing work:
- Never overwrites existing .npy keypoint files
- Backs up all existing manifests before touching them
- Backs up existing vocab and checkpoints
- Can be safely stopped and restarted at any point

Run from project root:
    python3 scripts/overnight_pipeline.py
"""

import json, shutil, random, numpy as np, cv2, mediapipe as mp, urllib.request
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
import time

# ── Config ────────────────────────────────────────────────────────
REPO_ID    = "akasheroor/American-Sign-Language-Dataset"
BATCH_DIR  = Path("/tmp/aslense_batch")
ASLENSE_NPY_DIR = Path("data/processed/aslense")
PROC_DIR   = Path("data/processed")
BACKUP_DIR = Path("data/processed/backup_v1")
BATCH_SIZE = 300

BATCH_DIR.mkdir(parents=True, exist_ok=True)
ASLENSE_NPY_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 0: Backup everything existing ───────────────────────────
print("=" * 60)
print("STEP 0: Backing up existing work")
print("=" * 60)

files_to_backup = [
    'vocab.json',
    'train_manifest.json',
    'val_manifest.json',
    'test_manifest.json',
    'combined_train_manifest.json',
    'aslcitizen_train_manifest.json',
    'train_manifest_stratified.json',
    'val_manifest_stratified.json',
]

for fname in files_to_backup:
    src = PROC_DIR / fname
    dst = BACKUP_DIR / fname
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"  Backed up: {fname}")
    elif src.exists() and dst.exists():
        print(f"  Already backed up: {fname}")
    else:
        print(f"  Skipped (not found): {fname}")

# Back up checkpoints
ckpt_dir    = Path("models/checkpoints")
ckpt_backup = Path("models/checkpoints_v1_backup")
if ckpt_dir.exists() and not ckpt_backup.exists():
    shutil.copytree(ckpt_dir, ckpt_backup)
    print(f"  Backed up checkpoints → models/checkpoints_v1_backup/")
elif ckpt_backup.exists():
    print(f"  Checkpoints already backed up")

print(f"\nAll existing work preserved in: {BACKUP_DIR}")
print(f"To restore: cp {BACKUP_DIR}/* {PROC_DIR}/")

# ── Step 1: Load new vocab ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Loading new vocab")
print("=" * 60)

vocab_path = PROC_DIR / 'vocab.json'
if not vocab_path.exists():
    raise FileNotFoundError("vocab.json not found — run vocab builder first")

with open(vocab_path) as f:
    vocab = json.load(f)
print(f"Vocab: {len(vocab)} words")

# Verify this is the new expanded vocab (not the old 300-class one)
if len(vocab) <= 300:
    old_vocab = BACKUP_DIR / 'vocab.json'
    print(f"WARNING: vocab.json only has {len(vocab)} words.")
    print(f"This looks like the old vocab. Using backup instead.")
    print(f"Please run the vocab builder script first.")
    raise SystemExit(1)

# ── Step 2: MediaPipe setup ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Setting up MediaPipe")
print("=" * 60)

model_path = Path("data/hand_landmarker.task")
if not model_path.exists():
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(url, str(model_path))
print("MediaPipe model ready")

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode
detector = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

def extract_keypoints(video_path):
    cap    = cv2.VideoCapture(str(video_path))
    frames = []
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
        frames.append(np.concatenate([lh, rh]))
    cap.release()
    return np.stack(frames) if frames else np.zeros((1, 126))

# ── Step 3: Download + extract Aslense ───────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Downloading + extracting Aslense videos")
print("=" * 60)

print("Fetching file list from HuggingFace...")
all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
mp4_files = [f for f in all_files if f.endswith('.mp4')]
print(f"Total Aslense videos: {len(mp4_files)}")

# Build download queue — skip already extracted
to_process = []
already_done = 0
not_in_vocab = 0
for f in mp4_files:
    word = f.split('-', 1)[-1].replace('.mp4', '').lower().strip()
    if word not in vocab:
        not_in_vocab += 1
        continue
    npy_path = ASLENSE_NPY_DIR / (Path(f).stem + '.npy')
    if npy_path.exists():
        already_done += 1
        continue
    to_process.append((f, word, npy_path))

print(f"Already extracted: {already_done} ✓")
print(f"Not in vocab:      {not_in_vocab}")
print(f"To download now:   {len(to_process)}")
print(f"Estimated time:    {len(to_process) * 8 / 3600:.1f} hours")

# Load already-done files into manifest
aslense_manifest = []
for npy in ASLENSE_NPY_DIR.glob('*.npy'):
    stem     = npy.stem
    parts    = stem.split('-', 1)
    word     = parts[-1].lower().strip() if len(parts) > 1 else ''
    if word in vocab:
        aslense_manifest.append({
            "path":      str(npy),
            "label":     word,
            "label_idx": vocab[word],
            "source":    "aslense",
        })
print(f"Pre-loaded {len(aslense_manifest)} existing Aslense samples into manifest")

failed = 0
total_batches = (len(to_process) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_start in range(0, len(to_process), BATCH_SIZE):
    batch     = to_process[batch_start:batch_start + BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    print(f"\nBatch {batch_num}/{total_batches} "
          f"({len(aslense_manifest)} extracted so far)")

    # Download batch
    downloaded = []
    for filepath, word, npy_path in tqdm(batch, desc="Download", leave=False):
        tmp_path = BATCH_DIR / Path(filepath).name
        try:
            local = hf_hub_download(
                repo_id=REPO_ID,
                filename=filepath,
                repo_type="dataset",
                local_dir=str(BATCH_DIR),
                local_dir_use_symlinks=False)
            Path(local).rename(tmp_path)
            downloaded.append((tmp_path, word, npy_path))
        except Exception as e:
            failed += 1

    # Extract keypoints + delete video immediately
    for tmp_path, word, npy_path in tqdm(downloaded, desc="Extract", leave=False):
        try:
            kpts = extract_keypoints(tmp_path)
            np.save(str(npy_path), kpts)
            aslense_manifest.append({
                "path":      str(npy_path),
                "label":     word,
                "label_idx": vocab[word],
                "source":    "aslense",
            })
        except Exception as e:
            failed += 1
        finally:
            tmp_path.unlink(missing_ok=True)  # delete video immediately

    # Save manifest checkpoint every batch so progress isn't lost
    with open(PROC_DIR / 'aslense_manifest.json', 'w') as f:
        json.dump(aslense_manifest, f, indent=2)

detector.close()
print(f"\nAslense extraction complete:")
print(f"  Extracted: {len(aslense_manifest)}")
print(f"  Failed:    {failed}")

# ── Step 4: Rebuild WLASL manifests with new vocab ────────────────
print("\n" + "=" * 60)
print("STEP 4: Rebuilding WLASL manifests with new vocab")
print("=" * 60)

with open('data/raw/wlasl/WLASL_v0.3.json') as f:
    wlasl_data = json.load(f)

for split in ['train', 'val', 'test']:
    split_manifest = []
    missing_npy    = 0
    for entry in wlasl_data:
        gloss = entry['gloss'].lower()
        if gloss not in vocab:
            continue
        for inst in entry['instances']:
            if inst['split'] != split:
                continue
            vid_id   = inst['video_id']
            npy_path = Path(f"data/processed/{split}/{vid_id}.npy")
            if not npy_path.exists():
                missing_npy += 1
                continue
            split_manifest.append({
                "path":      str(npy_path),
                "label":     gloss,
                "label_idx": vocab[gloss],
                "source":    "wlasl",
            })
    out_path = PROC_DIR / f'{split}_manifest.json'
    with open(out_path, 'w') as f:
        json.dump(split_manifest, f, indent=2)
    print(f"  WLASL {split}: {len(split_manifest)} samples "
          f"({missing_npy} missing .npy skipped)")

# ── Step 5: Update ASL Citizen manifest ───────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Updating ASL Citizen manifest")
print("=" * 60)

# Use backup to avoid reading partially-modified file
aslc_src = BACKUP_DIR / 'aslcitizen_train_manifest.json'
if not aslc_src.exists():
    aslc_src = PROC_DIR / 'aslcitizen_train_manifest.json'

with open(aslc_src) as f:
    aslc = json.load(f)

aslc_updated = []
for s in aslc:
    word = s['label'].lower()
    if word in vocab:
        s['label_idx'] = vocab[word]
        aslc_updated.append(s)

with open(PROC_DIR / 'aslcitizen_train_manifest.json', 'w') as f:
    json.dump(aslc_updated, f, indent=2)
print(f"  ASL Citizen: {len(aslc_updated)} samples updated")

# ── Step 6: Build combined + stratified split ─────────────────────
print("\n" + "=" * 60)
print("STEP 6: Building combined + stratified split")
print("=" * 60)

with open(PROC_DIR / 'train_manifest.json') as f:
    wlasl_train = json.load(f)

combined_all = wlasl_train + aslc_updated + aslense_manifest

# Save raw combined
with open(PROC_DIR / 'combined_train_manifest.json', 'w') as f:
    json.dump(combined_all, f, indent=2)
print(f"Combined (raw): {len(combined_all)} samples")

# Stratified split
random.seed(42)
by_class  = defaultdict(list)
for s in combined_all:
    by_class[s['label']].append(s)

new_train, new_val = [], []
for label, samples in by_class.items():
    random.shuffle(samples)
    n_val = max(1, min(3, len(samples) // 5))
    new_val   += samples[:n_val]
    new_train += samples[n_val:]

with open(PROC_DIR / 'train_manifest_stratified.json', 'w') as f:
    json.dump(new_train, f, indent=2)
with open(PROC_DIR / 'val_manifest_stratified.json', 'w') as f:
    json.dump(new_val, f, indent=2)

# Make stratified the active split
shutil.copy(PROC_DIR / 'train_manifest_stratified.json',
            PROC_DIR / 'combined_train_manifest.json')
shutil.copy(PROC_DIR / 'val_manifest_stratified.json',
            PROC_DIR / 'val_manifest.json')

from collections import Counter
val_counts = Counter(s['label'] for s in new_val)
print(f"Stratified train: {len(new_train)} samples")
print(f"Stratified val:   {len(new_val)} samples")
print(f"Val classes with 2+ samples: "
      f"{sum(1 for c in val_counts.values() if c >= 2)}")

# ── Done ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
print(f"\nDataset summary:")
print(f"  Vocab:          {len(vocab)} classes")
print(f"  Train samples:  {len(new_train)}")
print(f"  Val samples:    {len(new_val)}")
print(f"\nBackups preserved at: {BACKUP_DIR}")
print(f"  (original 300-class vocab, manifests, checkpoints)")
print(f"\nNext step — retrain:")
print(f"  python3 -m src.train --vocab {len(vocab)} --epochs 150 --combined --workers 0")