"""
src/demo.py

Real-time ASL recognition demo using ONNX model + MediaPipe HandLandmarker (Tasks API).
Shows live webcam feed with hand skeleton overlay and predicted word.

Usage:
    python src/demo.py --model models/sign_model.onnx --vocab 100
"""

import argparse
import json
import collections
import time
import numpy as np
from pathlib import Path


WINDOW_FRAMES  = 60   # frames to buffer before running inference
SMOOTH_WINDOW  = 5    # smooth predictions over this many inference calls
KEYPOINT_DIM   = 126

# MediaPipe hand landmark connections (21 landmarks, 0-indexed)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (5,9),(9,10),(10,11),(11,12),   # middle
    (9,13),(13,14),(14,15),(15,16), # ring
    (13,17),(17,18),(18,19),(19,20),(0,17), # pinky + palm
]


def load_vocab(vocab_path: str = "data/processed/vocab.json") -> dict:
    """Returns {label_idx: sign_word} mapping."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab.items()}  # invert: idx → word


def _get_hand_detector():
    """Load MediaPipe HandLandmarker via Tasks API (same as dataloader)."""
    import urllib.request
    import mediapipe as mp
    model_path = Path("data/hand_landmarker.task")
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print("Downloading MediaPipe hand model...")
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


def extract_keypoints(frame, detector):
    """Extract 126-dim keypoint vector from a BGR frame using Tasks API."""
    import cv2
    import mediapipe as mp
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

    return np.concatenate([lh, rh]), result


def draw_skeleton(frame, result):
    """Draw hand landmark skeletons using Tasks API result."""
    import cv2
    h, w = frame.shape[:2]
    colors = {"Left": (121, 44, 250), "Right": (245, 66, 230)}

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name
        color = colors.get(handedness, (255, 255, 255))
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 1)
        for pt in pts:
            cv2.circle(frame, pt, 3, color, -1)

    return frame


def greedy_ctc_decode(log_probs: np.ndarray, blank: int) -> list:
    """Simple greedy CTC decoding: argmax → remove blanks and repeats."""
    preds  = log_probs.argmax(axis=-1)  # (T,)
    result = []
    prev   = None
    for token in preds:
        if token != blank and token != prev:
            result.append(int(token))
        prev = token
    return result


def run_demo(onnx_path: str, vocab_size: int = 100):
    try:
        import cv2
        import mediapipe as mp
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Run: pip install -r requirements.txt")

    idx_to_word = load_vocab()
    sess        = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    detector    = _get_hand_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Make sure a camera is connected.")

    frame_buffer   = collections.deque(maxlen=WINDOW_FRAMES)
    pred_history   = collections.deque(maxlen=SMOOTH_WINDOW)
    current_word   = "Waiting..."
    fps_tracker    = collections.deque(maxlen=30)
    last_inference = time.time()

    print("Demo running — press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()

        # Extract keypoints
        kpts, result = extract_keypoints(frame, detector)
        frame_buffer.append(kpts)

        # Run inference every WINDOW_FRAMES frames
        if len(frame_buffer) == WINDOW_FRAMES and (time.time() - last_inference) > 0.5:
            seq = np.stack(list(frame_buffer), axis=0)[np.newaxis].astype(np.float32)
            msk = np.zeros((1, WINDOW_FRAMES), dtype=bool)
            log_probs = sess.run(["log_probs"],
                                  {"keypoints": seq, "padding_mask": msk})[0]
            # log_probs: (T, 1, C) → squeeze to (T, C)
            log_probs = log_probs[:, 0, :]
            decoded   = greedy_ctc_decode(log_probs, blank=vocab_size)
            if decoded:
                pred_history.append(decoded[0])
                if pred_history:
                    most_common = collections.Counter(pred_history).most_common(1)[0][0]
                    current_word = idx_to_word.get(most_common, "Unknown")
            last_inference = time.time()

        # Draw skeleton
        frame = draw_skeleton(frame, result)

        # FPS
        fps_tracker.append(1.0 / max(time.perf_counter() - t_start, 1e-6))
        fps = np.mean(fps_tracker)

        # Overlay text
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, current_word,
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 1)
        cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{WINDOW_FRAMES}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        cv2.imshow("ASL Real-Time Demo — Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/sign_model.onnx")
    parser.add_argument("--vocab", type=int, default=100)
    args = parser.parse_args()
    run_demo(args.model, args.vocab)
