# GEM-HD
Gaze-based Early Mental Health Detection

![GEM-HD](results/figures/GEM.png)

## What this project does (code-accurate)
GEM-HD turns webcam eye behavior into numbers each second and predicts a continuous engagement score in [0, 1]. It also quantifies how uncertain the prediction is and produces simple risk labels and reports.

Concretely, the code:
- Extracts per-second eye-gaze features from video/webcam: `fixation_duration_ms`, `saccade_velocity_deg_s`, `blink_rate_hz`, `scanpath_entropy`, `aoi_hit_rate` (see `src/feature_extraction`).
- Trains a small neural net (`TinyMLP`) or an optional `MultimodalModel` to predict engagement (see `src/model.py`, `src/train.py`).
- Estimates uncertainty with Monte Carlo Dropout (epistemic) and Conformal Prediction intervals (see `src/train.py`, `src/predict.py`, `src/conformal.py`).
- Generates a clinical-style JSON summary with risk labels and recommendations (see `src/predict.py`).
- Serves a live Streamlit dashboard that shows webcam frames, a rolling prediction with uncertainty, drift/feedback summaries, and the HTML model card (see `streamlit_app.py`).

## Key components
- `src/feature_extraction/`: build `ref.parquet` (reference) and `current.parquet` (your data) with per-second features.
- `src/model.py`: `TinyMLP` (tabular features) and `MultimodalModel` (optionally fuses simple eye-image features).
- `src/train.py`: trains model, calibrates conformal intervals, saves `artifacts/mental_health_model.pt` and reports.
- `src/predict.py`: loads the trained model, runs MC Dropout + Conformal, writes predictions and a clinical report.
- `src/infer_sklearn.py`: quick baseline using scikit-learn MLP; writes `current_with_pred.parquet` with `y_pred`, `y_std`, `ema_engagement`.
- `src/evaluation_agent/model_card_generator.py`: builds `results/model_card_V.1.0.html` from metrics/drift/feedback.
- `streamlit_app.py`: live dashboard reading files under `results/` and camera.

## Data flow
1. Feature extraction → `data/processed/ref.parquet` and `data/processed/current.parquet`
2. Train (on ref) → `artifacts/mental_health_model.pt` + training reports
3. Predict (on current) → predictions Parquet + clinical report JSON
4. Optional quick path → `src/infer_sklearn.py` to generate `current_with_pred.parquet`
5. Dashboard → reads `results/*.json` and `results/model_card_*.html`

## How to run

### 1) Extract features (video/webcam)
```bash
# Build reference features from ARGaze (writes data/processed/ref.parquet)
python -m feature_extraction.extract_features --write-ref

# Process a demo/webcam video (writes data/processed/current.parquet)
python -m feature_extraction.extract_features --video data/demo_videos/webcam.mp4 --video-out data/processed/current.parquet --max-frames 300
```

### 2) Train the model (TinyMLP by default)
```bash
python src/train.py \
  --ref_path data/processed/ref.parquet \
  --output_dir artifacts \
  --model_type tiny \
  --epochs 30 --batch_size 128 --alpha 0.10 --mc_passes 30
```

### 3) Predict and create a clinical report
```bash
python src/predict.py \
  --current_path data/processed/current.parquet \
  --model_path artifacts/mental_health_model.pt \
  --output_path results/mental_health_assessment.parquet \
  --patient_id PATIENT_001 --mc_passes 30
```

### 4) Quick baseline (no PyTorch training)
```bash
python src/infer_sklearn.py \
  --ref data/processed/ref.parquet \
  --current data/processed/current.parquet \
  --output data/processed/current_with_pred.parquet \
  --alpha 0.10
```

### 5) Generate model card HTML
```bash
python src/evaluation_agent/model_card_generator.py --run configs/run.yaml --version V.1.0
```

### 6) Run the live dashboard
```bash
streamlit run streamlit_app.py
```

## Outputs
- Predictions Parquet (e.g., `results/mental_health_assessment.parquet`) with:
  - `y_pred`, `epistemic_std`, `conformal_lower`, `conformal_upper`, `risk_level`, `ema_engagement`
- Clinical report JSON (same stem as predictions) with trend/volatility and recommendations
- Model artifact: `artifacts/mental_health_model.pt`
- Model card HTML: `results/model_card_V.1.0.html`
- Optional: `data/processed/current_with_pred.parquet` (from `infer_sklearn.py`)

## Uncertainty (implemented in code)
- **Epistemic**: MC Dropout via repeated forward passes in train/predict.
- **Conformal Prediction**: split-conformal half-width `q` for coverage-controlled intervals.

## Minimal dependencies
```bash
pip install torch pandas numpy scikit-learn streamlit matplotlib opencv-python pillow pyyaml
# optional (feature extraction, extended docs): mediapipe
```

## Notes & limitations
- Webcam gaze from face/iris landmarks is approximate (fast and robust, not a calibrated eye tracker).
- Risk labels are heuristic and for decision support only; not a medical diagnosis.