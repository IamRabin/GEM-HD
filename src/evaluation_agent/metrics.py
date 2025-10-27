import json, math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from data_loader import load_yaml
from utils.logger import setup_logger

# ---------- config ----------
DEFAULT_RESULTS_DIR = Path("results")
CURVE_MAX_Z = 3.0  # cover most half-normal mass 3std


# ---------- core metrics ----------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": mae, "RMSE": rmse}

def ece_regression(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, bins: int = 21) -> dict:
    """
    Regression calibration via standardized absolute error:
      z = |y - mu| / sigma
    Compare empirical CDF of z to Half-Normal CDF F(t)=erf(t/sqrt(2)).
    Returns mean absolute deviation (ECE) and the curve points for plotting.
    """
    eps = 1e-8
    sigma = np.clip(y_std, eps, None)
    z = np.abs(y_true - y_pred) / sigma

    ts = np.linspace(0.0, CURVE_MAX_Z, bins)
    empirical = np.array([(z <= t).mean() for t in ts], dtype=float)
    theoretical = np.array([math.erf(t / math.sqrt(2.0)) for t in ts], dtype=float)
    ece = float(np.mean(np.abs(empirical - theoretical)))

    curve = [{"t": float(t), "emp": float(e), "theo": float(th)} for t, e, th in zip(ts, empirical, theoretical)]
    return {"ECE": ece, "curve": curve}

def uncertainty_error_correlation(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict:
    abs_err = np.abs(y_true - y_pred)
    rho, p = spearmanr(abs_err, y_std)
    return {"Uncertainty_Error_SpearmanR": float(rho), "p_value": float(p)}

# ---------- plotting (optional) ----------
def plot_calibration_curve(curve: list, out_path: Path):
    ts = [c["t"] for c in curve]
    emp = [c["emp"] for c in curve]
    theo = [c["theo"] for c in curve]
    plt.figure(figsize=(5,4))
    plt.plot(ts, emp, label="Empirical")
    plt.plot(ts, theo, label="Half-Normal (ideal)", linestyle="--")
    plt.xlabel("t (standardized error threshold)")
    plt.ylabel("CDF")
    plt.title("Regression Calibration Curve")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_err_vs_uncert(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, out_path: Path):
    abs_err = np.abs(y_true - y_pred)
    plt.figure(figsize=(5,4))
    plt.scatter(y_std, abs_err, s=8, alpha=0.6)
    plt.xlabel("Uncertainty (σ)")
    plt.ylabel("|Error|")
    plt.title("|Error| vs Uncertainty")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- main entry ----------
def compute_and_save_metrics(run_yaml: str, results_dir: str | Path | None = None, bins: int = 21, make_plots: bool = True):
    
    cfg = load_yaml(run_yaml)
    logger = setup_logger(log_dir=cfg["paths"]["output_dir"])
    logger.info("[Phase 2: Metrics Computation]")
    logger.info(f"Loaded configuration: {run_yaml}")

    cur_path = Path(cfg["paths"]["current_data"]).expanduser().resolve()
    if results_dir:
       out_dir = Path(results_dir).expanduser().resolve()
    else:
       out_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

      # --- log paths clearly ---
    logger.info(f"Current dataset: {cur_path}")
    logger.info(f"Results will be saved to: {out_dir}")

    df = pd.read_parquet(cur_path)
    
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # required columns
    needed = ["y_pred", "y_std"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {cur_path}: {missing}")

    y_pred = df["y_pred"].to_numpy(dtype=float)
    y_std  = df["y_std"].to_numpy(dtype=float)

    metrics = {}
    # If ground truth exists, compute full set; otherwise, compute what we can.
    if "ema_engagement" in df.columns:
        y_true = df["ema_engagement"].to_numpy(dtype=float)
        metrics.update(regression_metrics(y_true, y_pred))
        ece_obj = ece_regression(y_true, y_pred, y_std, bins=bins)
        metrics["ECE"] = ece_obj["ECE"]
        ue = uncertainty_error_correlation(y_true, y_pred, y_std)
        metrics.update(ue)

        # plots
        if make_plots:
            plot_calibration_curve(ece_obj["curve"], out_dir / "plots" / "calibration_curve.png")
            plot_err_vs_uncert(y_true, y_pred, y_std, out_dir / "plots" / "err_vs_uncertainty.png")
    else:
        # no ground truth available → skip accuracy/calibration; still summarize uncertainty
        metrics["MAE"] = None
        metrics["RMSE"] = None
        metrics["ECE"] = None
        metrics["Uncertainty_Error_SpearmanR"] = None
        metrics["uncertainty_mean"] = float(np.mean(y_std))
        metrics["uncertainty_std"]  = float(np.std(y_std))
        metrics["note"] = "No ground truth provided (ema_engagement missing). Computed uncertainty summary only."

    # save
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Phase 2] Wrote metrics → {out_dir / 'metrics.json'}")


if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to run.yaml (contains paths.current_data)")
    ap.add_argument("--results_dir", default=None, help="Override results output dir")
    ap.add_argument("--bins", type=int, default=21)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    compute_and_save_metrics(args.run, args.results_dir, bins=args.bins, make_plots=not args.no_plots)


    # import yaml
    # import os
    # with open("/home/rabink1/D1/gemhd/GEM-HD/src/evaluation_agent/config/run.yaml") as f:
    #   cfg = yaml.safe_load(f)

    # ref_data = os.path.expanduser(cfg["paths"]["ref_data"])
    # current_data = os.path.expanduser(cfg["paths"]["current_data"])
    # output_dir = os.path.expanduser(cfg["paths"]["output_dir"])
    # print(output_dir)

        