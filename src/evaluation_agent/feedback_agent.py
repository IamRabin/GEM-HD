"""
Reads metrics & drift outputs and produces recommended actions.
Outputs: results/feedback_report.json
"""

import json
from pathlib import Path
import yaml

from data_loader import load_yaml, load_json
from utils.logger import setup_logger

import matplotlib.pyplot as plt
from matplotlib.table import Table



def save_feedback_table_as_png(actions, out_dir):
    """Render the feedback summary table as a PNG for model card visualization."""
    if not actions:
        return None

    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.35 * len(actions)))
    ax.axis("off")
    table = Table(ax, bbox=[0, 0, 1, 1])

    n_cols = 2
    width = 1.0 / n_cols
    height = 1.0 / (len(actions) + 1)

    # Header row (no textprops; use normal facecolor)
    headers = ["Action", "Reason"]
    for j, h in enumerate(headers):
        cell = table.add_cell(0, j, width, height, text=h, loc="center", facecolor="#40466e")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    # Data rows
    for i, a in enumerate(actions, start=1):
        color = "#e6f2ff"
        if a["type"].lower() == "retrain":
            color = "#ffd6d6"  # light red
        elif a["type"].lower() == "recalibrate":
            color = "#fff4cc"  # light yellow
        elif a["type"].lower() == "investigate":
            color = "#e6ffe6"  # light green

        table.add_cell(i, 0, width, height, text=a["type"].upper(), loc="center", facecolor=color)
        table.add_cell(i, 1, width, height, text=a["reason"], loc="left", facecolor="white")

    ax.add_table(table)

    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    img_path = plots_dir / "feedback_table.png"
    plt.savefig(img_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return img_path



def generate_feedback(run_yaml: str):
    cfg = load_yaml(run_yaml)
    out_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    logger = setup_logger(log_dir=out_dir)
    logger.info("[Phase 4: Feedback & Action Suggestion]")

    metrics_path = out_dir / "metrics.json"
    drift_path = out_dir / "drift_report.json"

    if not metrics_path.exists() or not drift_path.exists():
        logger.error("Missing metrics.json or drift_report.json — run Phases 2 and 3 first.")
        return None

    metrics = load_json(metrics_path)
    drift = load_json(drift_path)

    actions = []
    summary = {"retrain_needed": False, "recalibration_needed": False, "pipeline_check": False}

    # ----- Evaluate metrics thresholds -----
    mae_thr = cfg["feedback"]["mae_threshold"]
    ece_thr = cfg["feedback"]["ece_threshold"]

    if metrics.get("MAE") and metrics["MAE"] > mae_thr:
        actions.append({
            "type": "retrain",
            "reason": f"MAE = {metrics['MAE']:.3f} > threshold {mae_thr}"
        })
        summary["retrain_needed"] = True
        logger.warning(f"MAE {metrics['MAE']:.3f} > {mae_thr} → Retraining suggested")

    if metrics.get("ECE") and metrics["ECE"] > ece_thr:
        actions.append({
            "type": "recalibrate",
            "reason": f"ECE = {metrics['ECE']:.3f} > threshold {ece_thr}"
        })
        summary["recalibration_needed"] = True
        logger.warning(f"ECE {metrics['ECE']:.3f} > {ece_thr} → Recalibration advised")

    # ----- Check drift report -----
    ks_report = drift.get("ks_report", {})
    drifted_features = [f for f, d in ks_report.items() if d.get("drift_detected")]
    if drifted_features:
        actions.append({
            "type": "retrain",
            "reason": f"Drift detected in {', '.join(drifted_features)}"
        })
        summary["retrain_needed"] = True
        logger.warning(f"Drift detected in {len(drifted_features)} features → Retraining suggested")

    outlier_ratio = drift.get("outlier_ratio", 0)
    if outlier_ratio > 0.05:
        actions.append({
            "type": "investigate",
            "reason": f"Outlier ratio = {outlier_ratio:.3f} > 0.05"
        })
        summary["pipeline_check"] = True
        logger.warning(f"Outlier ratio {outlier_ratio:.3f} > 0.05 → Investigate data pipeline")

    # ----- Compose feedback report -----
    feedback = {"actions": actions, "summary": summary}
    out_path = out_dir / "feedback_report.json"
    with open(out_path, "w") as f:
        json.dump(feedback, f, indent=2)

    logger.info(f"Feedback report saved → {out_path}")
    if not actions:
        logger.info("Model and data within healthy thresholds — no action needed.")
    
    img_path = save_feedback_table_as_png(actions, out_dir)
    if img_path:
        logger.info(f"📊 Feedback summary table saved as PNG → {img_path}")

    return feedback


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 4 — Feedback Agent")
    ap.add_argument("--run", required=True, help="Path to run.yaml config")
    args = ap.parse_args()
    generate_feedback(args.run)
