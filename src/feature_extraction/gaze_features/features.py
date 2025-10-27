from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _compute_velocity_px_per_s(x: np.ndarray, y: np.ndarray, fps: float) -> np.ndarray:
    dx = np.diff(x, prepend=x[:1])
    dy = np.diff(y, prepend=y[:1])
    v = np.hypot(dx, dy) * fps
    return v


def _per_second_bins(t_sec: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    # Bin edges from 0 to last whole second
    last_t = float(t_sec[-1] if t_sec.size > 0 else 0.0)
    num_seconds = int(math.floor(last_t)) + 1
    edges = np.arange(0, num_seconds + 1, dtype=np.float32)
    # Map each frame to its integer second bin index
    idx = np.clip(t_sec.astype(np.int32), 0, num_seconds - 1) if num_seconds > 0 else np.array([], dtype=np.int32)
    return idx, edges


def compute_per_second_aggregates(
    df_long: pd.DataFrame,
    screen_size_px: Tuple[int, int] = (1280, 720),
    fps: float = 60.0,
    velocity_threshold_px_s: float = 100.0,
    deg_per_px: Optional[float] = None,
) -> pd.DataFrame:
    """Compute per-second features from long-form target/gaze samples.

    Returns wide-form per-second DataFrame with columns:
    participant, scene, second,
    fixation_duration_ms, saccade_velocity_deg_s (or px_s), blink_rate_hz, scanpath_entropy
    """
    if df_long.empty:
        return pd.DataFrame(
            columns=[
                "participant",
                "scene",
                "second",
                "fixation_duration_ms",
                "saccade_velocity_deg_s",
                "saccade_velocity_px_s",
                "blink_rate_hz",
                "scanpath_entropy",
            ]
        )

    records = []
    for (pid, scene), g in df_long.groupby(["participant", "scene"], sort=True):
        x = g["target_x"].to_numpy(dtype=np.float32)
        y = g["target_y"].to_numpy(dtype=np.float32)
        t = g["t_sec"].to_numpy(dtype=np.float32)

        v = _compute_velocity_px_per_s(x, y, fps)
        bin_idx, edges = _per_second_bins(t, fps)
        num_seconds = edges.size - 1

        width, height = screen_size_px
        # 2D histogram bins for entropy; coarse grid to be robust
        grid_x, grid_y = 32, 18

        for s in range(num_seconds):
            mask = bin_idx == s
            if not np.any(mask):
                continue
            v_s = v[mask]
            # Fixation duration: sum of frames where velocity below threshold
            num_fix_frames = float((v_s < velocity_threshold_px_s).sum())
            fixation_ms = 1000.0 * (num_fix_frames / fps)

            # Saccade velocity: 95th percentile per second
            saccade_px_s = float(np.percentile(v_s, 95)) if v_s.size > 0 else float("nan")
            saccade_deg_s = float(saccade_px_s * deg_per_px) if deg_per_px is not None else float("nan")

            # Blink rate: not available from ARGaze dataset → NaN
            blink_rate_hz = float("nan")

            # Scanpath entropy over this second
            xs = x[mask]
            ys = y[mask]
            # Clamp to screen bounds if they are pixel coords; otherwise this is a reasonable normalization
            xs = np.clip(xs, 0, width - 1)
            ys = np.clip(ys, 0, height - 1)
            H, _, _ = np.histogram2d(xs, ys, bins=[grid_x, grid_y], range=[[0, width], [0, height]])
            p = H.ravel()
            p = p / p.sum() if p.sum() > 0 else p
            # Shannon entropy in bits
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.nansum(p * (np.log2(p, where=p > 0))) if p.size > 0 else float("nan")

            records.append(
                {
                    "participant": pid,
                    "scene": int(scene),
                    "second": int(s),
                    "fixation_duration_ms": float(fixation_ms),
                    "saccade_velocity_deg_s": saccade_deg_s,
                    "saccade_velocity_px_s": saccade_px_s,
                    "blink_rate_hz": blink_rate_hz,
                    "scanpath_entropy": float(ent),
                }
            )

    out = pd.DataFrame.from_records(records)
    return out



