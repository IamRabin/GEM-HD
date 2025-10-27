from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


S2_PATTERN = re.compile(r"^P(\d+)_S2\.mp4$", re.IGNORECASE)
ZIP_PATTERN = re.compile(r"^P(\d+)\.zip$", re.IGNORECASE)


@dataclass(frozen=True)
class SessionSpec:
    participant: str  # e.g., "P1"
    scene: int = 2


def _list_root(root: str) -> List[str]:
    try:
        return os.listdir(root)
    except FileNotFoundError:
        return []


def discover_s2_participants(root: str) -> List[str]:
    """Discover participants that have Scene 2 data.

    Heuristics:
    - Match top-level scene preview videos like `P1_S2.mp4`.
    - Match `P*.zip` and verify it contains a `P*_S2/target.npy`.
    - Fallback to unzipped directories containing `P*_S2/target.npy` under root or root/P*/
    """
    entries = _list_root(root)

    via_preview = {
        f"P{m.group(1)}" for e in entries if (m := S2_PATTERN.match(e)) is not None
    }

    via_zip: set[str] = set()
    for e in entries:
        m = ZIP_PATTERN.match(e)
        if not m:
            continue
        pid = f"P{m.group(1)}"
        zip_path = os.path.join(root, e)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                has_target = any(
                    name.replace("\\", "/").endswith(f"{pid}_S2/target.npy") for name in zf.namelist()
                )
                if has_target:
                    via_zip.add(pid)
        except zipfile.BadZipFile:
            continue

    # Unzipped directories
    via_dir: set[str] = set()
    # Case A: root contains P*_S2 directories
    for e in entries:
        if re.match(r"^P\d+_S2$", e, re.IGNORECASE):
            pid = e.split("_", 1)[0]
            if os.path.exists(os.path.join(root, e, "target.npy")):
                via_dir.add(pid)
    # Case B: root contains P* directories with nested P*_S2/target.npy
    for e in entries:
        if re.match(r"^P\d+$", e, re.IGNORECASE) and os.path.isdir(os.path.join(root, e)):
            pid = e
            nested = os.path.join(root, pid, f"{pid}_S2", "target.npy")
            if os.path.exists(nested):
                via_dir.add(pid)

    participants = sorted(via_preview.union(via_zip).union(via_dir), key=lambda s: int(s[1:]))
    return participants


def _load_target_from_zip(root: str, participant: str) -> Optional[np.ndarray]:
    zip_path = os.path.join(root, f"{participant}.zip")
    if not os.path.exists(zip_path):
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Normalize names for cross-platform paths
            for name in zf.namelist():
                norm = name.replace("\\", "/")
                if norm.endswith(f"{participant}_S2/target.npy"):
                    with zf.open(name) as f:
                        data = f.read()
                        bio = io.BytesIO(data)
                        # Allow pickle for .npy default; dataset provenance trusted locally
                        arr = np.load(bio, allow_pickle=True)
                        arr = np.asarray(arr)
                        return arr
    except zipfile.BadZipFile:
        return None
    return None


def _load_target_from_dir(root: str, participant: str) -> Optional[np.ndarray]:
    # Accept either root/P*_S2/target.npy or root/P*/P*_S2/target.npy
    direct = os.path.join(root, f"{participant}_S2", "target.npy")
    nested = os.path.join(root, participant, f"{participant}_S2", "target.npy")
    path = direct if os.path.exists(direct) else (nested if os.path.exists(nested) else None)
    if path is None:
        return None
    try:
        arr = np.load(path, allow_pickle=True)
        return np.asarray(arr)
    except Exception:
        return None


def _infer_fps_and_length(target_xy: np.ndarray) -> Tuple[int, int]:
    # README states: preview videos are 60 fps, and each sequence contains 26552 frames
    fps = 60
    n = int(target_xy.shape[0]) if target_xy is not None else 0
    return fps, n


def load_scene2_targets(
    root: str, participants: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Load target coordinates for Scene 2 as a stand-in for screen-coordinate gaze.

    Returns long-form DataFrame with columns:
    - participant, scene, frame, t_sec, target_x, target_y
    - head_yaw_deg, head_pitch_deg, head_roll_deg (NaN for ARGaze, not available)
    """
    if participants is None:
        participants = discover_s2_participants(root)

    records: List[dict] = []
    for pid in participants:
        target_xy = _load_target_from_zip(root, pid)
        if target_xy is None:
            target_xy = _load_target_from_dir(root, pid)
        if target_xy is None:
            # Fallback: no targets found
            continue

        fps, n = _infer_fps_and_length(target_xy)
        t = np.arange(n, dtype=np.float32) / float(fps)
        # target.npy is expected shape (N, 2) with x,y coordinates
        if target_xy.ndim == 1 and target_xy.size == n * 2:
            target_xy = target_xy.reshape(n, 2)
        if target_xy.shape[1] != 2:
            # Skip malformed
            continue

        for i in range(n):
            x, y = float(target_xy[i, 0]), float(target_xy[i, 1])
            records.append(
                {
                    "participant": pid,
                    "scene": 2,
                    "frame": i,
                    "t_sec": float(t[i]),
                    "target_x": x,
                    "target_y": y,
                    # Head pose not available from this dataset content
                    "head_yaw_deg": np.nan,
                    "head_pitch_deg": np.nan,
                    "head_roll_deg": np.nan,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df



