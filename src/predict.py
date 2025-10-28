#!/usr/bin/env python3
"""
Mental Health Detection Prediction Script
Predicts gaze engagement for early mental health detection
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import json

from utils_io import load_parquet_or_csv, save_parquet_or_csv
from model import create_model
from conformal import SplitConformalRegressor

FEATS = ["fixation_duration_ms", "saccade_velocity_deg_s", "blink_rate_hz", "scanpath_entropy", "aoi_hit_rate"]

@torch.no_grad()
def mc_predict(model, X_np, device, T=30, batch=256):
    """Monte Carlo prediction with dropout for uncertainty estimation"""
    model.train()  # enable dropout
    n = X_np.shape[0]
    acc = []
    for _ in range(T):
        preds=[]
        for i in range(0, n, batch):
            xb = torch.from_numpy(X_np[i:i+batch]).to(device)
            # Handle different model interfaces
            if hasattr(model, 'forward') and model.__class__.__name__ == 'TinyMLP':
                preds.append(model(xb).cpu().numpy())
            else:
                preds.append(model(xb, None).cpu().numpy())
        acc.append(np.concatenate(preds, axis=0))
    acc = np.stack(acc, axis=0)
    return acc.mean(axis=0), acc.std(axis=0)

def categorize_mental_health_risk(engagement_score, uncertainty_std, conformal_width):
    """Categorize mental health risk based on engagement and uncertainty"""
    if engagement_score < 0.3 and uncertainty_std < 0.05:
        return "HIGH_RISK"  # Low engagement, high confidence
    elif engagement_score < 0.5 and conformal_width > 0.3:
        return "MODERATE_RISK"  # Moderate engagement, high uncertainty
    elif engagement_score > 0.7 and uncertainty_std < 0.05:
        return "LOW_RISK"  # High engagement, high confidence
    else:
        return "UNCERTAIN"  # Need more data

def analyze_temporal_patterns(engagement_history, window_size=10):
    """Analyze temporal patterns for mental health indicators"""
    if len(engagement_history) < window_size:
        return {"trend": 0, "volatility": 0, "stability": 1, "pattern_type": "INSUFFICIENT_DATA"}
    
    # Trend analysis
    trend = np.polyfit(range(len(engagement_history)), engagement_history, 1)[0]
    
    # Volatility analysis
    volatility = np.std(engagement_history)
    
    # Stability analysis
    stability = 1.0 / (volatility + 1e-6)
    
    # Pattern classification
    if trend > 0.01 and volatility < 0.1:
        pattern_type = "IMPROVING_STABLE"
    elif trend < -0.01 and volatility < 0.1:
        pattern_type = "DECLINING_STABLE"
    elif volatility > 0.2:
        pattern_type = "HIGHLY_VARIABLE"
    else:
        pattern_type = "STABLE"
    
    return {
        'trend': float(trend),
        'volatility': float(volatility),
        'stability': float(stability),
        'pattern_type': pattern_type
    }

def generate_clinical_report(predictions_df, patient_id="unknown"):
    """Generate clinical decision support report"""
    
    # Overall statistics
    mean_engagement = predictions_df['y_pred'].mean()
    mean_uncertainty = predictions_df['epistemic_std'].mean()
    mean_interval_width = (predictions_df['conformal_upper'] - predictions_df['conformal_lower']).mean()
    
    # Risk categorization
    risk_levels = []
    for _, row in predictions_df.iterrows():
        risk = categorize_mental_health_risk(
            row['y_pred'], 
            row['epistemic_std'], 
            row['conformal_upper'] - row['conformal_lower']
        )
        risk_levels.append(risk)
    
    risk_counts = pd.Series(risk_levels).value_counts()
    dominant_risk = risk_counts.index[0] if len(risk_counts) > 0 else "UNCERTAIN"
    
    # Temporal analysis
    temporal_analysis = analyze_temporal_patterns(predictions_df['y_pred'].values)
    
    # Clinical recommendations
    recommendations = []
    if dominant_risk == "HIGH_RISK":
        recommendations.extend([
            "Schedule immediate clinical assessment",
            "Monitor for signs of depression or attention disorders",
            "Consider additional cognitive evaluations"
        ])
    elif dominant_risk == "MODERATE_RISK":
        recommendations.extend([
            "Continue monitoring engagement patterns",
            "Schedule follow-up assessment within 2 weeks",
            "Consider lifestyle interventions"
        ])
    elif dominant_risk == "LOW_RISK":
        recommendations.extend([
            "Continue current care plan",
            "Routine follow-up as scheduled",
            "Maintain healthy engagement activities"
        ])
    else:
        recommendations.extend([
            "Collect additional data for better assessment",
            "Monitor patterns over extended period",
            "Consider multiple assessment methods"
        ])
    
    # Generate report
    report = {
        'patient_id': patient_id,
        'assessment_date': datetime.now().isoformat(),
        'overall_assessment': {
            'mean_engagement': float(mean_engagement),
            'mean_uncertainty': float(mean_uncertainty),
            'mean_interval_width': float(mean_interval_width),
            'dominant_risk_level': dominant_risk,
            'risk_distribution': risk_counts.to_dict()
        },
        'temporal_analysis': temporal_analysis,
        'clinical_recommendations': recommendations,
        'data_quality': {
            'n_samples': len(predictions_df),
            'confidence_level': 'HIGH' if mean_uncertainty < 0.05 else 'MODERATE',
            'assessment_reliability': 'HIGH' if mean_interval_width < 0.3 else 'MODERATE'
        }
    }
    
    return report

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    cur = load_parquet_or_csv(Path(args.current_path))
    
    # Load model checkpoint
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    feat_cols = ckpt["feat_cols"]
    assert feat_cols == FEATS, f"Feature mismatch: expected {FEATS}, got {feat_cols}"
    
    x_mean = ckpt["x_mean"]
    x_std = ckpt["x_std"]
    q = float(ckpt["q_conformal"])
    alpha = float(ckpt["alpha"])
    
    # Get model type and kwargs
    model_type = ckpt.get("model_type", "tiny")
    model_kwargs = ckpt.get("model_kwargs", {})
    
    print(f"Loading {model_type} model for mental health assessment...")
    
    # Create model
    model = create_model(model_type, **model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    
    # Load uncertainty parameters
    uncertainty_params = ckpt.get("uncertainty_params", {})
    conformal_q = uncertainty_params.get("conformal_q", q)
    
    # Process current data
    Xc = cur[FEATS].values.astype(np.float32)
    Xc_n = (Xc - x_mean) / x_std
    
    # Comprehensive uncertainty estimation
    print(f"Making mental health assessment predictions...")
    
    # Epistemic uncertainty (MC Dropout)
    print("  - Computing epistemic uncertainty (MC Dropout)...")
    yhat, epistemic_std = mc_predict(model, Xc_n, device, T=args.mc_passes)
    
    # Get conformal intervals
    print("  - Computing conformal prediction intervals...")
    cp = SplitConformalRegressor(alpha=alpha)
    cp.q_ = conformal_q  # Set from saved model
    conformal_lower, conformal_upper = cp.interval(yhat)
    
    # Create output dataframe
    out = cur.copy()
    out["y_pred"] = yhat.astype(float)
    out["epistemic_std"] = epistemic_std.astype(float)
    out["conformal_lower"] = conformal_lower.astype(float)
    out["conformal_upper"] = conformal_upper.astype(float)
    
    # Add mental health specific columns
    out["risk_level"] = [categorize_mental_health_risk(
        yhat[i], epistemic_std[i], conformal_upper[i] - conformal_lower[i]
    ) for i in range(len(yhat))]
    
    # Simple EMA for UI smoothing
    beta = 0.85
    m = 0.0
    ema = []
    for v in yhat:
        m = beta * m + (1 - beta) * float(v)
        ema.append(m)
    out["ema_engagement"] = np.array(ema, dtype=float)
    
    # Generate clinical report
    clinical_report = generate_clinical_report(out, args.patient_id)
    
    # Save results
    path = save_parquet_or_csv(out, Path(args.output_path))
    report_path = Path(args.output_path).with_suffix('.json')
    
    with open(report_path, 'w') as f:
        json.dump(clinical_report, f, indent=2)
    
    print(f"Wrote predictions: {path}")
    print(f"Wrote clinical report: {report_path}")
    print(f"Model: {model_type}")
    print(f"Patient ID: {args.patient_id}")
    print(f"Assessment: {clinical_report['overall_assessment']['dominant_risk_level']}")
    print(f"Mean engagement: {yhat.mean():.3f} ± {yhat.std():.3f}")
    print(f"Epistemic uncertainty: {epistemic_std.mean():.3f} ± {epistemic_std.std():.3f}")
    print(f"Conformal intervals: [{conformal_lower.mean():.3f}, {conformal_upper.mean():.3f}]")
    
    # Print clinical recommendations
    print("\n=== CLINICAL RECOMMENDATIONS ===")
    for i, rec in enumerate(clinical_report['clinical_recommendations'], 1):
        print(f"{i}. {rec}")
    
    print(f"\nTemporal pattern: {clinical_report['temporal_analysis']['pattern_type']}")
    print(f"Data quality: {clinical_report['data_quality']['confidence_level']} confidence")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mental Health Assessment using Gaze Engagement")
    p.add_argument("--current_path", type=str, default="current.parquet")
    p.add_argument("--model_path", type=str, default="artifacts/mental_health_model.pt")
    p.add_argument("--output_path", type=str, default="artifacts/mental_health_assessment.parquet")
    p.add_argument("--mc_passes", type=int, default=30)
    p.add_argument("--patient_id", type=str, default="unknown", help="Patient identifier for clinical report")
    args = p.parse_args()
    main(args)