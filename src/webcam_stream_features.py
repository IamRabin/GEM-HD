#!/usr/bin/env python3
"""
Stream per-second gaze-like features from the webcam to data/processed/current.parquet.

If MediaPipe is available, uses FaceMesh to estimate eye landmarks and a proxy
for gaze motion and blink rate. Otherwise, falls back to simple motion-based
signals from OpenCV to keep the dashboard updating.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List
import json

import cv2
import numpy as np
import pandas as pd


FEATS = [
    "fixation_duration_ms",
    "saccade_velocity_deg_s",
    "blink_rate_hz",
    "scanpath_entropy",
    "aoi_hit_rate",
]


def save_parquet_append(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prev = pd.read_parquet(path)
        out = pd.concat([prev, df], ignore_index=True)
    else:
        out = df
    out.to_parquet(path, index=False)
    return path


def try_load_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        return mp
    except Exception:
        return None


def compute_entropy(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> float:
    if xs.size == 0:
        return float("nan")
    H, _, _ = np.histogram2d(xs, ys, bins=[32, 18], range=[[0, width], [0, height]])
    p = H.ravel()
    if p.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(p * (np.log2(p, where=p > 0)))
    return float(ent)


def _open_camera(dev: int) -> cv2.VideoCapture:
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(0)
    for be in backends:
        cap = cv2.VideoCapture(dev, be)
        if cap.isOpened():
            return cap
    # Final fallback
    cap = cv2.VideoCapture(dev)
    return cap


def stream(args: argparse.Namespace) -> None:
    cap = _open_camera(int(args.device))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open webcam (device {args.device})")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    mp = try_load_mediapipe()
    use_mp = mp is not None
    if use_mp:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    # Buffers for the current one-second window
    xs: List[float] = []
    ys: List[float] = []
    motion_vals: List[float] = []
    blink_counter = 0
    ear_vals: List[float] = []
    prev_gray: Optional[np.ndarray] = None
    sec_start = time.time()

    # Eye indices for a coarse EAR proxy
    LEFT_EYE = {"upper": 159, "lower": 145, "left": 33, "right": 133}
    RIGHT_EYE = {"upper": 386, "lower": 374, "left": 263, "right": 362}

    def _ear(landmarks: np.ndarray, spec: Dict[str, int]) -> float:
        up = landmarks[spec["upper"], :2]
        lo = landmarks[spec["lower"], :2]
        le = landmarks[spec["left"], :2]
        ri = landmarks[spec["right"], :2]
        vert = np.linalg.norm(up - lo)
        horiz = np.linalg.norm(le - ri) + 1e-6
        return float(vert / horiz)

    # Resolve absolute paths to avoid CWD issues
    base_dir = Path(__file__).resolve().parents[1]
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    live_dir = base_dir / "results/figures"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_frame = live_dir / "live.jpg"
    status_file = base_dir / "results/stream_status.json"

    print(f"[webcam] streaming to {out_path} (CTRL+C to stop)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Attempt lightweight recovery: reopen camera
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.5)
                cap = _open_camera(int(args.device))
                time.sleep(0.5)
                continue

            # Save a downscaled live frame for the dashboard (avoid heavy IO)
            try:
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(live_frame), small)
            except Exception:
                pass

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_vals.append(float(diff.mean()))
            prev_gray = gray

            if use_mp:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0]
                    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                    ear_val = 0.5 * (_ear(pts, LEFT_EYE) + _ear(pts, RIGHT_EYE))
                    ear_vals.append(ear_val)
                    # Use iris average for coarse 2D point
                    iris_idx = [468, 469, 470, 471, 473, 474, 475, 476]
                    iris_center = pts[iris_idx, :3].mean(axis=0)
                    xs.append(float(iris_center[0] * width))
                    ys.append(float(iris_center[1] * height))
            else:
                # Fallback: use center of mass of intensity as a proxy point
                M = cv2.moments(gray)
                cx = (M["m10"] / (M["m00"] + 1e-6)) if M["m00"] > 0 else width / 2
                cy = (M["m01"] / (M["m00"] + 1e-6)) if M["m00"] > 0 else height / 2
                xs.append(float(cx))
                ys.append(float(cy))

            # Every second, aggregate and append a row
            if (time.time() - sec_start) >= 1.0:
                if use_mp and ear_vals:
                    # Simple blink detection: threshold on EAR
                    ear_arr = np.array(ear_vals, dtype=np.float32)
                    blinks = int((ear_arr < 0.21).sum() > 2)
                else:
                    blinks = int(np.mean(motion_vals) > 5.0)  # heuristic

                # Fixation proxy: low motion frames
                if motion_vals:
                    low_motion = np.mean(np.array(motion_vals) < (np.mean(motion_vals) + 0.5 * np.std(motion_vals)))
                else:
                    low_motion = 0.5
                fixation_ms = float(low_motion * 1000.0)

                # Saccade proxy: percentile of motion
                saccade_deg_s = float(np.percentile(np.array(motion_vals or [0.0]), 95))

                ent = compute_entropy(np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), width, height)

                row = pd.DataFrame([{ 
                    "fixation_duration_ms": fixation_ms,
                    "saccade_velocity_deg_s": saccade_deg_s,
                    "blink_rate_hz": float(blinks),
                    "scanpath_entropy": float(ent),
                    "aoi_hit_rate": np.nan,
                }])
                save_parquet_append(row, out_path)
                # Write health status
                try:
                    status = {"last_frame_ts": time.time(), "frame_size": [int(width), int(height)]}
                    status_file.parent.mkdir(parents=True, exist_ok=True)
                    status_file.write_text(json.dumps(status))
                except Exception:
                    pass
                # reset window buffers
                xs.clear(); ys.clear(); motion_vals.clear(); ear_vals.clear()
                sec_start = time.time()
    finally:
        cap.release()
        if use_mp:
            face_mesh.close()


def main():
    ap = argparse.ArgumentParser(description="Stream webcam features to current.parquet")
    ap.add_argument("--output", default="data/processed/current.parquet")
    ap.add_argument("--device", default="0", help="Webcam device index (default 0)")
    args = ap.parse_args()
    stream(args)


if __name__ == "__main__":
    main()


