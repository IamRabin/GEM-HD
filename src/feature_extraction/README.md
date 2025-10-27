Feature Extraction Scripts
==========================

Layout
------
- data/raw/ARGaze: source dataset root
- data/processed: outputs
  - ref.parquet: baseline features from ARGaze Scene 2
  - current.parquet: webcam/demo features
- data/demo_videos: optional short videos

Usage
-----
- Build ref from ARGaze:
```
python -m feature_extraction.extract_features --write-ref
```
- Process a demo video:
```
python -m feature_extraction.extract_features --video data/demo_videos/webcam.mp4 --video-out data/processed/current.parquet --max-frames 300
```

If you omit --video, the script auto-discovers a video under `data/demo_videos/*.mp4` or `data/webcam.mp4`.

Optional: save eye crops during webcam processing for debugging/alignment with ARGaze eye views:
```
python -m feature_extraction.extract_features --video data/demo_videos/webcam.mp4 --save-eye-crops-dir data/processed/eye_crops
```

Alternative CLI
---------------
If you prefer subcommands, you can also use the internal CLI:
```
python -m feature_extraction.gaze_features.cli argaze --root data/raw/ARGaze --output data/processed/ref.parquet
python -m feature_extraction.gaze_features.cli video --input data/webcam.mp4 --output data/processed/current.parquet
```

Schema
------
Both outputs conform to the same per-second schema:

- `timestamp`: integer second (0-based)
- `fixation_duration_ms`
- `saccade_velocity_deg_s`
- `blink_rate_hz`
- `scanpath_entropy`
- `aoi_hit_rate` (optional; NaN for webcam by default)