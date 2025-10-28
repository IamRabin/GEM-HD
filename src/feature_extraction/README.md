_Feature Extraction Scripts_
==========================

**What this does** (simplified)
------------------------------
We watch a video of someone’s face and, for every second, write a tiny “eye report card” with simple numbers: how long the eyes stayed still (fixations), how fast they jumped (saccades), how often they blinked, and how messy the looking path was (entropy). For the ARGaze dataset, we treat provided screen targets as a stand‑in for gaze and compute the same per‑second stats.

Layout
------
- data/raw/ARGaze: source dataset root
- data/processed: outputs
  - ref.parquet: baseline features from ARGaze Scene 2
  - current.parquet: webcam/demo features
- data/demo_videos: test short videos 

**Under the hood** (models and features)
------------------------------------
- Webcam/demo videos
  - Face/eye landmarks via MediaPipe Face Mesh (with `refine_landmarks=True` for iris/lids).
  - Blink detection from Eye Aspect Ratio (EAR) between eyelid and corner landmarks; a blink is EAR < 0.21 for ≥2 frames.
  - Gaze direction approximated from iris center relative to the midpoint of eye corners; per‑frame angular speed (deg/s) is computed from changes in that vector.
  - Per‑second features:
    - fixation_duration_ms: number of frames with angular speed < 30 deg/s, converted to ms
    - saccade_velocity_deg_s: 95th percentile of angular speed within the second
    - blink_rate_hz: estimated blinks per second over the clip
    - scanpath_entropy: Shannon entropy over a coarse 2D histogram of iris positions (32×18 bins)
    - aoi_hit_rate: not defined for webcam, saved as NaN
  - Optional: save raw eye crops for debugging with `--save-eye-crops-dir`.

- ARGaze (reference) path
  - Loads Scene 2 `target.npy` arrays (screen target coordinates) for discovered participants.
  - Treats these targets as proxy gaze in screen pixels at 60 fps and computes the same per‑second stats.
  - Saccade velocity is computed in px/s and, if a degree‑per‑pixel factor is provided, also in deg/s.
  - Blink rate is not available from ARGaze targets (saved as NaN). AOI hit rate can be added later if AOIs are defined.


**Usage**
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

**Schema**
------
Both outputs conform to the same per-second schema:

- `timestamp`: integer second (0-based)
- `fixation_duration_ms`
- `saccade_velocity_deg_s`
- `blink_rate_hz`
- `scanpath_entropy`
- `aoi_hit_rate` (optional; NaN for webcam by default)

What’s a Parquet file?
----------------------
Parquet is a compact, columnar table format. Think “CSV, but smaller and faster,” with real data types and compression. Load it with pandas:
```
import pandas as pd
df = pd.read_parquet("data/processed/current.parquet")
print(df.head())
```

Notes and limitations
---------------------
- MediaPipe gaze is an approximation from 2D face/iris landmarks; it’s robust and fast, not a calibrated eye tracker.
- The 30 deg/s fixation threshold and EAR blink threshold are conventional defaults; adjust as needed for your data.
- For ARGaze, blink rate and AOI hits are not derived from the target arrays; they’re left as NaN unless you add AOI definitions and blink sources.