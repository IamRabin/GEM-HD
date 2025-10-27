from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class MediaPipeConfig:
    static_image_mode: bool = False
    max_num_faces: int = 1
    refine_landmarks: bool = True  # iris/lids
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


# Landmark indices for eyelid aspect ratio (approx, using FaceMesh)
# Using left eye: upper lid (159), lower lid (145); horizontal corners (133, 33)
# Using right eye: upper lid (386), lower lid (374); horizontal corners (362, 263)
LEFT_EYE = {"upper": 159, "lower": 145, "left": 33, "right": 133}
RIGHT_EYE = {"upper": 386, "lower": 374, "left": 263, "right": 362}


def _eye_aspect_ratio(landmarks: np.ndarray, spec: Dict[str, int]) -> float:
    up = landmarks[spec["upper"], :2]
    lo = landmarks[spec["lower"], :2]
    le = landmarks[spec["left"], :2]
    ri = landmarks[spec["right"], :2]
    vert = np.linalg.norm(up - lo)
    horiz = np.linalg.norm(le - ri) + 1e-6
    return float(vert / horiz)


def _angular_speed_deg_per_s(vecs: List[np.ndarray], fps: float) -> np.ndarray:
    if not vecs:
        return np.array([], dtype=np.float32)
    v = np.vstack(vecs)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    dots = np.sum(v[1:] * v[:-1], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    ang = np.concatenate([[0.0], ang])  # same length as frames
    return ang * fps


def process_video_to_per_second(
    video_path: str,
    fps_hint: Optional[float] = None,
    ear_blink_threshold: float = 0.21,
    ear_blink_min_frames: int = 2,
    max_frames: Optional[int] = None,
    save_eye_crops_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Process a raw webcam video to per-second aggregates using MediaPipe Face Mesh.

    Returns per-second DataFrame with fixation_duration_ms, saccade_velocity_deg_s, blink_rate_hz, scanpath_entropy.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = fps_hint or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    try:
        import mediapipe as mp
    except Exception as e:  # pragma: no cover - optional dependency
        cap.release()
        raise RuntimeError("mediapipe is required for video processing") from e

    mp_face_mesh = mp.solutions.face_mesh
    cfg = MediaPipeConfig()
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=cfg.static_image_mode,
        max_num_faces=cfg.max_num_faces,
        refine_landmarks=cfg.refine_landmarks,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )

    # Buffers
    gaze_vecs: List[np.ndarray] = []
    left_ears: List[float] = []
    right_ears: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    ts: List[float] = []

    t = 0.0
    sec = 1.0 / float(fps)

    # Prepare optional eye crop saving
    save_left_dir: Optional[Path] = None
    save_right_dir: Optional[Path] = None
    if save_eye_crops_dir is not None:
        base = Path(save_eye_crops_dir)
        save_left_dir = base / "left"
        save_right_dir = base / "right"
        save_left_dir.mkdir(parents=True, exist_ok=True)
        save_right_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_count >= max_frames:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            # Eye aspect ratios
            left_ears.append(_eye_aspect_ratio(pts, LEFT_EYE))
            right_ears.append(_eye_aspect_ratio(pts, RIGHT_EYE))
            # Use iris center approx (average of four iris landmarks when refine_landmarks=True): 468..471 left, 473..476 right
            iris_idx = [468, 469, 470, 471, 473, 474, 475, 476]
            iris_pts = pts[iris_idx, :3]
            iris_center = iris_pts.mean(axis=0)
            # Eyeball center approx: midpoint between eye corners and a fixed depth offset
            left_corner = pts[LEFT_EYE["left"], :3]
            right_corner = pts[RIGHT_EYE["right"], :3]
            eye_mid = 0.5 * (left_corner + right_corner)
            gaze_vec = iris_center - eye_mid
            gaze_vecs.append(gaze_vec)
            # For entropy, keep 2D normalized coordinates (use iris_center x,y)
            xs.append(float(iris_center[0] * width))
            ys.append(float(iris_center[1] * height))

            # Optional: save simple eye crops around each eye using a bounding box from key landmarks
            if save_left_dir is not None and save_right_dir is not None:
                # Build bounding boxes from the four eyelid/corner landmarks for each eye
                def _crop_from_indices(spec: Dict[str, int]) -> Optional[np.ndarray]:
                    idxs = [spec["upper"], spec["lower"], spec["left"], spec["right"]]
                    pts_px = np.stack([
                        [pts[i, 0] * width, pts[i, 1] * height] for i in idxs
                    ], axis=0)
                    min_xy = np.floor(pts_px.min(axis=0)).astype(int)
                    max_xy = np.ceil(pts_px.max(axis=0)).astype(int)
                    # Expand box by a margin
                    w = max_xy[0] - min_xy[0]
                    h = max_xy[1] - min_xy[1]
                    margin_x = max(4, int(0.4 * w))
                    margin_y = max(4, int(0.4 * h))
                    x0 = max(0, min_xy[0] - margin_x)
                    y0 = max(0, min_xy[1] - margin_y)
                    x1 = min(width, max_xy[0] + margin_x)
                    y1 = min(height, max_xy[1] + margin_y)
                    if x1 <= x0 or y1 <= y0:
                        return None
                    return frame[y0:y1, x0:x1]

                left_crop = _crop_from_indices(LEFT_EYE)
                right_crop = _crop_from_indices(RIGHT_EYE)
                if left_crop is not None:
                    cv2.imwrite(str(save_left_dir / f"frame_{frame_count:06d}.png"), left_crop)
                if right_crop is not None:
                    cv2.imwrite(str(save_right_dir / f"frame_{frame_count:06d}.png"), right_crop)
        else:
            left_ears.append(np.nan)
            right_ears.append(np.nan)
            gaze_vecs.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            xs.append(width / 2)
            ys.append(height / 2)

        ts.append(t)
        t += sec
        frame_count += 1

    cap.release()
    face_mesh.close()

    # Blink detection via EAR threshold with simple debounce
    ear = np.nanmean(np.vstack([left_ears, right_ears]), axis=0)
    blink_frames = ear < ear_blink_threshold
    blink_count = 0
    i = 0
    n = blink_frames.size
    while i < n:
        if blink_frames[i]:
            j = i
            while j < n and blink_frames[j]:
                j += 1
            if (j - i) >= ear_blink_min_frames:
                blink_count += 1
            i = j
        else:
            i += 1

    ang_speed = _angular_speed_deg_per_s(gaze_vecs, fps)

    # Per-second aggregates
    if not ts:
        return pd.DataFrame(columns=["timestamp", "fixation_duration_ms", "saccade_velocity_deg_s", "blink_rate_hz", "scanpath_entropy", "aoi_hit_rate"])  # noqa: E501

    t_arr = np.array(ts, dtype=np.float32)
    num_seconds = int(np.floor(t_arr[-1])) + 1
    sec_idx = np.clip(t_arr.astype(np.int32), 0, num_seconds - 1)

    records: List[dict] = []
    for s in range(num_seconds):
        mask = sec_idx == s
        if not np.any(mask):
            continue
        v_s = ang_speed[mask]
        # Fixation when angular speed < 30 deg/s (typical threshold)
        fixation_ms = 1000.0 * float((v_s < 30.0).sum()) / float(fps)
        saccade_deg_s = float(np.percentile(v_s, 95)) if v_s.size > 0 else float("nan")

        # Blink rate: blinks per second within this bin (uniformly distribute)
        # Approximate by total blinks divided by total duration
        blink_rate_hz = float(blink_count) / max(1.0, num_seconds)

        xs_s = np.array(xs, dtype=np.float32)[mask]
        ys_s = np.array(ys, dtype=np.float32)[mask]
        H, _, _ = np.histogram2d(xs_s, ys_s, bins=[32, 18], range=[[0, width], [0, height]])
        p = H.ravel()
        p = p / p.sum() if p.sum() > 0 else p
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(p * (np.log2(p, where=p > 0))) if p.size > 0 else float("nan")

        records.append(
            {
                "timestamp": int(s),
                "fixation_duration_ms": float(fixation_ms),
                "saccade_velocity_deg_s": float(saccade_deg_s),
                "blink_rate_hz": float(blink_rate_hz),
                "scanpath_entropy": float(ent),
                # AOIs not defined for webcam; leave optional field as NaN
                "aoi_hit_rate": np.nan,
            }
        )

    return pd.DataFrame.from_records(records)



