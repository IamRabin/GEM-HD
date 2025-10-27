from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd

from .argaze import discover_s2_participants, load_scene2_targets
from .features import compute_per_second_aggregates
from .mediapipe_eye import process_video_to_per_second


def _cmd_argaze(args: argparse.Namespace) -> None:
    root = args.root
    out = args.output
    deg_per_px = args.deg_per_px

    participants = discover_s2_participants(root)
    if not participants:
        raise SystemExit("No Scene 2 participants found.")

    df_long = load_scene2_targets(root, participants)
    per_sec = compute_per_second_aggregates(
        df_long,
        screen_size_px=(args.screen_width, args.screen_height),
        fps=60.0,
        velocity_threshold_px_s=args.fix_vt_px_s,
        deg_per_px=deg_per_px,
    )
    # If deg_per_px is None, saccade_velocity_deg_s column will be NaN; px/s column provided as fallback
    per_sec.to_parquet(out, index=False)
    print(f"Wrote {len(per_sec)} rows to {out}")


def _cmd_video(args: argparse.Namespace) -> None:
    out = args.output
    df = process_video_to_per_second(
        args.input,
        fps_hint=args.fps_hint,
        max_frames=args.max_frames,
    )
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gaze-features", description="ARGaze and webcam feature extraction")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_argaze = sub.add_parser("argaze", help="Process ARGaze Scene 2 to parquet")
    p_argaze.add_argument("--root", required=True, help="Path to ARGaze root directory")
    p_argaze.add_argument("--output", required=True, help="Output parquet path (e.g., ref.parquet)")
    p_argaze.add_argument("--screen-width", type=int, default=1280)
    p_argaze.add_argument("--screen-height", type=int, default=720)
    p_argaze.add_argument("--fix-vt-px-s", type=float, default=100.0, help="Velocity threshold for fixation (px/s)")
    p_argaze.add_argument("--deg-per-px", type=float, default=None, help="Conversion factor if known")
    p_argaze.set_defaults(func=_cmd_argaze)

    p_video = sub.add_parser("video", help="Process raw webcam video to parquet")
    p_video.add_argument("--input", required=True, help="Path to webcam video file")
    p_video.add_argument("--output", required=True, help="Output parquet path")
    p_video.add_argument("--fps-hint", type=float, default=None)
    p_video.add_argument("--max-frames", type=int, default=None, help="Process at most N frames")
    p_video.set_defaults(func=_cmd_video)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()



