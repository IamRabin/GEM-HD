
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path

EXPECTED_COLS = [
    "fixation_duration_ms",
    "saccade_velocity_deg_s",
    "blink_rate_hz",
    "scanpath_entropy",
    "y_pred",
    "y_std",
    "ema_engagement"  # optional
]

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def validate_schema(df, name):
    # Allow alias: map epistemic_std → y_std if needed
    if "y_std" not in df.columns and "epistemic_std" in df.columns:
        df = df.rename(columns={"epistemic_std": "y_std"})

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"⚠️  {name}: Missing columns {missing}")
    # Keep only expected columns
    df = df[[c for c in EXPECTED_COLS if c in df.columns]]
    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def clean_data(df):
    # Fill NaN with column means
    df = df.fillna(df.mean(numeric_only=True))
    # Remove extreme outliers (3σ rule)
    for col in df.select_dtypes(include=[np.number]).columns:
        mean, std = df[col].mean(), df[col].std()
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    return df.reset_index(drop=True)


def normalize_features(df, exclude=None):
    if exclude is None:
        exclude = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in exclude:
            min_, max_ = df[col].min(), df[col].max()
            if max_ > min_:
                df[col] = (df[col] - min_) / (max_ - min_)
    return df

def load_data(run_config_path):
    cfg = load_yaml(run_config_path)
    ref_path = Path(cfg["paths"]["ref_data"]).expanduser()
    cur_path = Path(cfg["paths"]["current_data"]).expanduser()

    print(f"Loading reference data: {ref_path}")
    ref_df = pd.read_parquet(ref_path)
    ref_df = validate_schema(ref_df, "ref")
    ref_df = clean_data(ref_df)
    ref_df = normalize_features(ref_df, exclude=["y_pred", "y_std", "ema_engagement"])

    print(f" Loading current data: {cur_path}")
    cur_df = pd.read_parquet(cur_path)
    cur_df = validate_schema(cur_df, "current")
    cur_df = clean_data(cur_df)
    cur_df = normalize_features(cur_df, exclude=["y_pred", "y_std", "ema_engagement"])

    print(f" Loaded {len(ref_df)} ref rows, {len(cur_df)} current rows.")
    return ref_df, cur_df

# Create output folder if not exists
Path("data/processed").mkdir(parents=True, exist_ok=True)

def make_dummy_data(n=500, drift=False, seed=42):
    np.random.seed(seed)
    
    # baseline means
    mean_fix = 350 if not drift else 400
    mean_sac = 150 if not drift else 130
    mean_blink = 0.25 if not drift else 0.32
    mean_entropy = 0.55 if not drift else 0.68
    
    # generate random data
    df = pd.DataFrame({
        "fixation_duration_ms": np.random.normal(mean_fix, 60, n).clip(150, 700),
        "saccade_velocity_deg_s": np.random.normal(mean_sac, 30, n).clip(40, 300),
        "blink_rate_hz": np.random.normal(mean_blink, 0.05, n).clip(0.1, 0.6),
        "scanpath_entropy": np.random.normal(mean_entropy, 0.12, n).clip(0.2, 1.0),
    })
    
    # create engagement (higher fixation + lower entropy → higher engagement)
    df["ema_engagement"] = (
        1 - (df["scanpath_entropy"] - 0.2) / 0.8
    ) * 0.6 + (df["saccade_velocity_deg_s"] / 250) * 0.4
    df["ema_engagement"] = df["ema_engagement"].clip(0, 1)
    
    # simulate model predictions (Member 2)
    noise = np.random.normal(0, 0.05, n)
    df["y_pred"] = df["ema_engagement"] + noise
    df["y_pred"] = df["y_pred"].clip(0, 1)
    
    # simulate uncertainty (higher when drift=True)
    base_unc = 0.08 if not drift else 0.15
    df["y_std"] = np.random.normal(base_unc, 0.03, n).clip(0.03, 0.25)
    
    return df


def clean_and_prepare_data(run_yaml):
    """
    loads, validates, cleans, and normalizes
    reference and current datasets for downstream phases.
    Returns cleaned DataFrames.
    """
    ref_df, cur_df = load_data(run_yaml)

    # Save processed outputs to ensure reproducibility
    cfg = load_yaml(run_yaml)
    ref_path = Path(cfg["paths"]["ref_data"]).expanduser()
    cur_path = Path(cfg["paths"]["current_data"]).expanduser()

    ref_df.to_parquet(ref_path)
    cur_df.to_parquet(cur_path)

    print(f"✅ Cleaned data saved to {ref_path} and {cur_path}")
    return ref_df, cur_df


if __name__ == "__main__":
    ref_df = make_dummy_data(n=500, drift=False)
    cur_df = make_dummy_data(n=400, drift=True)

    ref_df.to_parquet("data/processed/ref.parquet")
    cur_df.to_parquet("data/processed/current_with_pred.parquet")

    print("✅ Dummy data generated successfully!")

    df = pd.read_parquet("data/processed/ref.parquet")
    print(df.head())

