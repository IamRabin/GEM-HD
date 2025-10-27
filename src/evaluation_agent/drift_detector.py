"""
Phase 3 — Drift & Outlier Detection
-----------------------------------
Compares reference vs. current data distributions
to detect drift and anomalies in gaze-based features.
Outputs: results/drift_report.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
import yaml

from data_loader import load_data
from utils.logger import setup_logger


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ks_drift(ref_df, cur_df, features, p_threshold=0.05):
    """Run univariate Kolmogorov–Smirnov drift test for each feature."""
    drift = {}
    for f in features:
        stat, p = ks_2samp(ref_df[f], cur_df[f])
        drift[f] = {
            "KS_stat": float(stat),
            "p_value": float(p),
            "drift_detected": bool(p < p_threshold),
        }
    return drift


def mahalanobis_drift(ref_df, cur_df):
    """Compute Mahalanobis distance of current samples relative to reference distribution."""
    emp_cov = EmpiricalCovariance().fit(ref_df)
    md = emp_cov.mahalanobis(cur_df)
    return float(np.mean(md)), float(np.std(md))


def outlier_detection(cur_df):
    """Detect proportion of outliers using IsolationForest."""
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(cur_df)
    outlier_ratio = (preds == -1).mean()
    return float(outlier_ratio)

def plot_drift_summary(ks_report, out_dir):
    """Generate a simple drift summary bar chart of p-values per feature."""
    features = list(ks_report.keys())
    p_values = [v["p_value"] for v in ks_report.values()]
    drift_flags = [v["drift_detected"] for v in ks_report.values()]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(features, p_values, color=["red" if d else "green" for d in drift_flags])
    plt.axhline(0.05, color="black", linestyle="--", label="p=0.05 threshold")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("p-value (KS-test)")
    plt.title("Feature Drift Summary")
    plt.legend()
    plt.tight_layout()

    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "drift_summary.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def compute_drift(run_yaml: str):
    cfg = load_yaml(run_yaml)

    # --- setup logger ---
    log_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    logger = setup_logger(log_dir=log_dir)
    logger.info("[Phase 3: Drift & Outlier Detection]")

    # --- load cleaned, normalized datasets from Phase 1 ---
    ref_df, cur_df = load_data(run_yaml)
    features = [c for c in ref_df.columns if c not in ["y_pred", "y_std", "ema_engagement"]]

    logger.info(f"Loaded reference: {len(ref_df)} rows, current: {len(cur_df)} rows")
    logger.info(f"Using {len(features)} features for drift analysis: {features}")

    # --- 1️⃣ Univariate KS tests ---
    ks_report = ks_drift(ref_df, cur_df, features, cfg["drift"]["p_value_threshold"])

    # --- 2️⃣ Multivariate Mahalanobis distance ---
    md_mean, md_std = mahalanobis_drift(ref_df[features], cur_df[features])

    # --- 3️⃣ Outlier ratio ---
    outlier_ratio = outlier_detection(cur_df[features])

    drift_summary = {
        "ks_report": ks_report,
        "mahalanobis_mean": md_mean,
        "mahalanobis_std": md_std,
        "outlier_ratio": outlier_ratio,
    }

    out_path = log_dir / "drift_report.json"
    with open(out_path, "w") as f:
        json.dump(drift_summary, f, indent=2)

    logger.info(f"Drift report saved → {out_path}")

    plot_path = plot_drift_summary(ks_report, log_dir)
    logger.info(f" Drift summary plot saved → {plot_path}")

    n_drifted = sum(1 for f, d in ks_report.items() if d["drift_detected"])
    logger.info(f"Detected drift in {n_drifted}/{len(features)} features.")
    logger.info(f"Mahalanobis mean: {md_mean:.3f} | Outlier ratio: {outlier_ratio:.3f}")

    return drift_summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Phase 3 — Drift Detection")
    ap.add_argument("--run", required=True, help="Path to run.yaml config")
    args = ap.parse_args()

    compute_drift(args.run)
