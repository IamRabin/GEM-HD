# GEM-HD
Gaze-based Early Mental Health Detection

![GEM-HD](results/figures/GEM.png)

## Overview

We use webcam-based eye-gaze trajectories to assess cognitive engagement and fatigue.
Our system estimates uncertainty, detects drift, and self-evaluates — producing a transparent model card.

## Features
- **Gaze Engagement Prediction**: Predicts cognitive engagement from gaze features
- **Mental Health Risk Assessment**: Categorizes risk levels (HIGH_RISK/MODERATE_RISK/LOW_RISK/UNCERTAIN)
- **Comprehensive Uncertainty Estimation**: Epistemic, aleatoric, and conformal prediction
- **Clinical Decision Support**: Automated recommendations and temporal analysis
- **Production Ready**: Minimal, clean codebase for deployment

## Folder Structure

```
src/
├── model.py                    # Model definitions (TinyMLP, MultimodalModel)
├── train.py                    # Training script with uncertainty estimation
├── predict.py                  # Prediction script with clinical reports
├── conformal.py                # Uncertainty analysis (epistemic, aleatoric, conformal)
├── utils_io.py                 # Data I/O utilities
├── feature_extraction/         # Feature extraction from webcam data
└── evaluation_agent/          # Model evaluation and drift detection
```

## Team Roles
- Member 1 – Data & Feature Engineer
- Member 2 – ML & Uncertainty Lead
- Member 3 – Evaluation & Ops Engineer

## How to Run

### 1. Feature Extraction
```bash
python src/feature_extraction/extract_features.py
```

### 2. Train Model
```bash
python src/train.py --ref_path ref.parquet --epochs 30
```

### 3. Mental Health Assessment
```bash
python src/predict.py --current_path current.parquet --model_path artifacts/mental_health_model.pt --patient_id "PATIENT_001"
```

### 4. Evaluation & Drift Detection
```bash
python src/evaluation_agent/main.py --run configs/run.yaml --judge configs/judge.yaml
```

## Example Output

### Predictions (`mental_health_assessment.parquet`):
- `y_pred` - Engagement score (0-1)
- `risk_level` - Risk categorization
- `epistemic_std` - Model confidence
- `conformal_lower/upper` - 90% confidence intervals

### Clinical Report (`mental_health_assessment.json`):
- Risk assessment and distribution
- Temporal pattern analysis
- Clinical recommendations
- Data quality metrics

### Model Card (`results/model_card.md`):
- Model performance metrics
- Uncertainty analysis
- Clinical validation results

### Drift Report (`results/drift_report.json`):
- Model drift detection
- Performance monitoring
- Data quality assessment

## Mental Health Indicators
- **Low engagement** → Depression, attention disorders
- **High engagement** → Good mental health
- **Variable patterns** → Anxiety, mood disorders

## Uncertainty Types
1. **Epistemic Uncertainty**: Model parameter confidence (MC Dropout)
2. **Aleatoric Uncertainty**: Data noise assessment (MSE-based)
3. **Conformal Prediction**: Statistical coverage guarantees (90% intervals)

## Purpose
- Cognitive Health Assessment
- Multimodal AI for Mental Health
- Tools for Early Detection
- Clinical Decision Support

## Requirements
```bash
pip install torch pandas numpy scipy scikit-learn mediapipe opencv-python
```

## About
Gaze-based Early Mental Health Detection using comprehensive uncertainty estimation and clinical decision support.