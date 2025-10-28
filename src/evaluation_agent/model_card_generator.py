from pathlib import Path
import yaml, json
from datetime import datetime
from dreams_mc.make_model_card import generate_modelcard
from utils.logger import setup_logger


def generate_model_card(run_yaml: str, version_num: str = "V.1.0"):
    """
    ------------------------------------------
    Uses an existing model_card.yaml file as the base template,
    appends dynamic metrics, drift, and feedback results to the
    performance_comments section, and generates an HTML model card.
    """

    cfg = yaml.safe_load(open(run_yaml))
    out_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    logger = setup_logger(log_dir=out_dir)
    logger.info("[ Model Card Generation Started]")

    # --- Define key paths ---
    metrics_path = out_dir / "metrics.json"
    drift_path = out_dir / "drift_report.json"
    feedback_path = out_dir / "feedback_report.json"
    # Use project-relative template to be cross-platform
    base_card_path = (Path(__file__).resolve().parent / "config" / "model_card.yaml").resolve()

    missing = [p for p in [metrics_path, drift_path, feedback_path, base_card_path] if not p.exists()]
    if missing:
        logger.error(f" Missing required files: {missing}")
        return None

    metrics = json.load(open(metrics_path))
    drift = json.load(open(drift_path))
    feedback = json.load(open(feedback_path))
    template = yaml.safe_load(open(base_card_path))

    dynamic_data = {
        "MAE": metrics.get("MAE"),
        "RMSE": metrics.get("RMSE"),
        "ECE": metrics.get("ECE"),
        "Uncertainty_Error_SpearmanR": metrics.get("Uncertainty_Error_SpearmanR"),
        "Detected_drift_features": [
            f for f, d in drift.get("ks_report", {}).items() if d.get("drift_detected")
        ],
        "Outlier_ratio": drift.get("outlier_ratio"),
        "Recommended_actions": [a["type"] for a in feedback.get("actions", [])],
    }

    template["performance_comments"] = (
        template.get("performance_comments", "")
        + "\n\n<h3>Performance Metrics Summary</h3>\n"
        + yaml.dump(dynamic_data, sort_keys=False)
    )

    # --- Update version dynamically ---
    template["model_version"] = version_num

    # --- Write updated YAML to results directory ---
    model_card_yaml_path = out_dir / "model_card_filled.yaml"
    with open(model_card_yaml_path, "w") as f:
        yaml.dump(template, f, sort_keys=False)

    logger.info(f"🧾 Model card YAML generated → {model_card_yaml_path}")

    output_html = out_dir / f"model_card_{version_num}.html"
    generate_modelcard(str(model_card_yaml_path), str(output_html), version_num)
    logger.info(f"Model card HTML generated → {output_html}")

    return output_html


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate GEM-HD model card from pre-made YAML template")
    ap.add_argument("--run", required=True, help="Path to run.yaml config")
    ap.add_argument("--version", default="V.1.0", help="Model version string")
    args = ap.parse_args()

    generate_model_card(args.run, args.version)
