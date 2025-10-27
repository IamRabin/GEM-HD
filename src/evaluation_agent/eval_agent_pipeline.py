"""
Main GEM-HD Evaluation Pipeline
-------------------------------
Sequentially runs Phases 1–5:
1. Data loading & cleaning
2. Metrics computation
3. Drift detection
4. Feedback agent
5. Model card generation
"""

from data_loader import clean_and_prepare_data ,load_yaml
from metrics import compute_and_save_metrics
from drift_detector import compute_drift
from feedback_agent import generate_feedback
from model_card_generator import generate_model_card
from utils.logger import setup_logger


from pathlib import Path
import yaml


def run_pipeline(run_yaml: str):
    cfg = load_yaml(run_yaml)
    out_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    logger = setup_logger(log_dir=out_dir)

    logger.info("🚀 Starting GEM-HD Evaluation Pipeline")

    # 1️⃣ Data Loader
    logger.info("🔹 Phase 1: Loading and cleaning data...")
    clean_and_prepare_data(run_yaml)

    # 2️⃣ Metrics
    logger.info("🔹 Phase 2: Running metrics computation...")
    compute_and_save_metrics(run_yaml)

    # 3️⃣ Drift
    logger.info("🔹 Phase 3: Running drift detection...")
    compute_drift(run_yaml)

    # 4️⃣ Feedback
    logger.info("🔹 Phase 4: Running feedback agent...")
    generate_feedback(run_yaml)

    # 5️⃣ Model Card
    logger.info("🔹 Phase 5: Generating model card...")
    generate_model_card(run_yaml, version_num="V.1.0")

    logger.info("✅ GEM-HD pipeline complete. Model card ready in results/")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run full GEM-HD evaluation pipeline")
    ap.add_argument("--run", required=True, help="Path to run.yaml")
    args = ap.parse_args()
    run_pipeline(args.run)
