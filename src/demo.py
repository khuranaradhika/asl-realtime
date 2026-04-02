"""
src/demo.py

Real-time ASL recognition demo using ONNX model + MediaPipe HandLandmarker.
Shows live webcam feed with hand skeleton overlay and predicted word.

Usage:
    python src/demo.py --model models/sign_model.onnx --vocab 2000
"""

import argparse
import json
import collections
import time
import numpy as np
from pathlib import Path

from src.config import KEYPOINT_DIM, WINDOW_FRAMES, SMOOTH_WINDOW
from src.decode import greedy_decode_sequence
from src.keypoints import get_hand_detector, extract_keypoints_from_frame


def load_vocab(vocab_path: str = "data/processed/vocab.json") -> dict:
    """Returns {label_idx: sign_word} mapping."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab.items()}


def draw_skeleton(frame, result):
    """Draw hand skeleton overlays on the frame using HandLandmarker result."""
    import cv2
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2

    mp_drawing = mp.solutions.drawing_utils
    hand_colors = [(121, 22, 76), (245, 117, 66)]  # Left=purple, Right=orange

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name
        color      = hand_colors[0] if handedness == "Left" else hand_colors[1]

        lm_proto = landmark_pb2.NormalizedLandmarkList()
        lm_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])
        mp_drawing.draw_landmarks(
            frame, lm_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=color,           thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250),  thickness=1))
    return frame


def run_demo(onnx_path: str, vocab_size: int = 2000):
    try:
        import cv2
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Run: pip install -r requirements.txt")

    idx_to_word = load_vocab()
    sess        = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    detector    = get_hand_detector()

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

        rgb             = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kpts, mp_result = extract_keypoints_from_frame(rgb, detector)
        frame_buffer.append(kpts)

        if len(frame_buffer) == WINDOW_FRAMES and (time.time() - last_inference) > 0.5:
            seq       = np.stack(list(frame_buffer), axis=0)[np.newaxis].astype(np.float32)
            msk       = np.zeros((1, WINDOW_FRAMES), dtype=bool)
            log_probs = sess.run(["log_probs"],
                                  {"keypoints": seq, "padding_mask": msk})[0]
            log_probs = log_probs[:, 0, :]  # (T, 1, C) → (T, C)
            decoded   = greedy_decode_sequence(log_probs, blank=vocab_size)
            if decoded:
                pred_history.append(decoded[0])
                most_common  = collections.Counter(pred_history).most_common(1)[0][0]
                current_word = idx_to_word.get(most_common, "Unknown")
            last_inference = time.time()

        frame = draw_skeleton(frame, mp_result)

        fps_tracker.append(1.0 / max(time.perf_counter() - t_start, 1e-6))
        fps = np.mean(fps_tracker)

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
    parser.add_argument("--vocab", type=int, default=2000)
    args = parser.parse_args()
    run_demo(args.model, args.vocab)
