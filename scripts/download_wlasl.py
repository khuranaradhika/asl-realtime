"""
scripts/download_wlasl.py

Robust WLASL video downloader. Run from your project root:
    python3 scripts/download_wlasl.py

Downloads non-YouTube videos from WLASL_v0.3.json directly into
data/raw/wlasl/videos/. YouTube videos require yt-dlp (optional).

Requirements:
    pip3 install requests tqdm yt-dlp
"""

import json
import os
import time
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

try:
    import requests
except ImportError:
    raise ImportError("pip3 install requests")


# ── Config ────────────────────────────────────────────────────────────────────

ANNO_PATH  = Path("data/raw/wlasl/WLASL_v0.3.json")
SAVE_DIR   = Path("data/raw/wlasl/videos")
TIMEOUT    = 15   # seconds per request
MIN_BYTES  = 1000 # skip files smaller than this (probably error pages)
HEADERS    = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/120.0.0.0 Safari/537.36"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def download_direct(url: str, save_path: Path) -> bool:
    """Download a direct MP4/SWF URL. Returns True on success."""
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=HEADERS, stream=True)
        if r.status_code != 200:
            return False
        data = b"".join(r.iter_content(chunk_size=8192))
        if len(data) < MIN_BYTES:
            return False
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def download_youtube(url: str, save_path: Path) -> bool:
    """Download a YouTube video using yt-dlp. Returns True on success."""
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", str(save_path), url],
            capture_output=True, timeout=60)
        return result.returncode == 0 and save_path.exists()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main(vocab_size: int = 100, skip_youtube: bool = False):
    if not ANNO_PATH.exists():
        raise FileNotFoundError(
            f"Annotation file not found at {ANNO_PATH}.\n"
            f"Copy it first:\n"
            f"  cp /tmp/wlasl/start_kit/WLASL_v0.3.json data/raw/wlasl/")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNO_PATH) as f:
        data = json.load(f)

    # Collect all video entries, optionally limited to top vocab_size signs
    # Count sign frequency to get top-N
    sign_counts = {}
    for entry in data:
        for inst in entry["instances"]:
            sign_counts[entry["gloss"]] = sign_counts.get(entry["gloss"], 0) + 1
    top_signs = set(sorted(sign_counts, key=sign_counts.get, reverse=True)[:vocab_size])

    # Build download queue
    queue = []
    for entry in data:
        if entry["gloss"] not in top_signs:
            continue
        for inst in entry["instances"]:
            url      = inst.get("url", "")
            vid_id   = inst["video_id"]
            if not url:
                continue
            ext       = "swf" if ".swf" in url else "mp4"
            save_path = SAVE_DIR / f"{vid_id}.{ext}"
            queue.append((url, vid_id, save_path))

    print(f"Total videos to attempt: {len(queue)}")
    print(f"Already downloaded: {sum(1 for _, _, p in queue if p.exists())}")
    print(f"Save directory: {SAVE_DIR.resolve()}\n")

    success = 0
    skipped = 0
    failed  = 0

    for url, vid_id, save_path in tqdm(queue, desc="Downloading"):
        # Already have it
        if save_path.exists() and save_path.stat().st_size > MIN_BYTES:
            skipped += 1
            continue

        if is_youtube(url):
            if skip_youtube:
                continue
            ok = download_youtube(url, save_path)
        else:
            ok = download_direct(url, save_path)

        if ok:
            success += 1
        else:
            failed += 1
            # Clean up empty/partial file
            if save_path.exists():
                save_path.unlink()

        # Brief pause to be polite to servers
        time.sleep(0.1)

    total = success + skipped
    print(f"\n✓ Done.")
    print(f"  Downloaded:  {success}")
    print(f"  Already had: {skipped}")
    print(f"  Failed:      {failed}")
    print(f"  Total usable: {total}")
    print(f"\nFiles saved to: {SAVE_DIR.resolve()}")

    if total < 500:
        print("\n⚠ Warning: fewer than 500 clips. Training may be unstable.")
        print("  Consider running with --include-youtube and installing yt-dlp:")
        print("  pip3 install yt-dlp && python3 scripts/download_wlasl.py")
    else:
        print(f"\n✓ Ready to extract keypoints:")
        print(f"  python3 src/dataloader.py --extract --split train --vocab 100")
        print(f"  python3 src/dataloader.py --extract --split val   --vocab 100")
        print(f"  python3 src/dataloader.py --extract --split test  --vocab 100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab",           type=int,  default=100,
                        help="Only download videos for top N signs (default: 100)")
    parser.add_argument("--include-youtube", action="store_true",
                        help="Also attempt YouTube downloads via yt-dlp")
    args = parser.parse_args()
    main(vocab_size=args.vocab, skip_youtube=not args.include_youtube)