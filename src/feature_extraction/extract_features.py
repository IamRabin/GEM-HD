from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .gaze_features.cli import build_parser as _build_core_parser
from .gaze_features.argaze import discover_s2_participants, load_scene2_targets
from .gaze_features.features import compute_per_second_aggregates
from .gaze_features.mediapipe_eye import process_video_to_per_second


DEFAULT_ARGAZE_ROOT = (Path(__file__).resolve().parents[2] / "data" / "raw" / "ARGaze")

def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(prog="extract-features", description="Feature extraction into data/processed")
	p.add_argument("--argaze-root", default=str(DEFAULT_ARGAZE_ROOT), help="ARGaze root under data/raw")
	p.add_argument("--write-ref", action="store_true", help="Write ref.parquet from ARGaze Scene 2")
	p.add_argument("--video", default=None, help="Path to video to process (auto-discovers under data/demo_videos/ or data/webcam.mp4)")
	p.add_argument("--video-out", default=str(Path("data/processed/current.parquet")))
	p.add_argument("--max-frames", type=int, default=None)
	p.add_argument("--save-eye-crops-dir", default=None, help="Optional directory to save left/right eye crops during webcam processing")
	return p


def main(argv: Optional[list[str]] = None) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)

	processed_dir = Path("data/processed")
	processed_dir.mkdir(parents=True, exist_ok=True)

	if args.write_ref:
		participants = discover_s2_participants(args.argaze_root)
		if not participants:
			raise SystemExit("No Scene 2 participants detected in ARGaze root")
		df_long = load_scene2_targets(args.argaze_root, participants)
		per_sec = compute_per_second_aggregates(df_long)
		# Normalize schema to required columns
		if "timestamp" not in per_sec.columns and "second" in per_sec.columns:
			per_sec = per_sec.rename(columns={"second": "timestamp"})
		# Ensure required columns exist
		required_cols = [
			"timestamp",
			"fixation_duration_ms",
			"saccade_velocity_deg_s",
			"blink_rate_hz",
			"scanpath_entropy",
			"aoi_hit_rate",
		]
		for c in required_cols:
			if c not in per_sec.columns:
				per_sec[c] = pd.NA
		per_sec = per_sec[required_cols]
		ref_path = processed_dir / "ref.parquet"
		per_sec.to_parquet(ref_path, index=False)
		print(f"Wrote {len(per_sec)} rows to {ref_path}")

	# Auto-discover a video if not provided
	video_path = args.video
	if video_path is None:
		candidates = []
		candidates += sorted(glob.glob(str(Path("data/demo_videos") / "*.mp4")))
		candidates += [str(Path("data/webcam.mp4"))]
		video_path = next((c for c in candidates if Path(c).exists()), None)
		if video_path is None:
			print("No video provided and none found under data/demo_videos/ or data/webcam.mp4")
		else:
			print(f"Auto-discovered video: {video_path}")

	if video_path is not None:
		df = process_video_to_per_second(
			video_path,
			max_frames=args.max_frames,
			save_eye_crops_dir=args.save_eye_crops_dir,
		)
		# Normalize schema to required columns
		required_cols = [
			"timestamp",
			"fixation_duration_ms",
			"saccade_velocity_deg_s",
			"blink_rate_hz",
			"scanpath_entropy",
			"aoi_hit_rate",
		]
		for c in required_cols:
			if c not in df.columns:
				df[c] = pd.NA
		df = df[required_cols]
		out_path = Path(args.video_out)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		df.to_parquet(out_path, index=False)
		print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
	main()
