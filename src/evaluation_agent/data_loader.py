
import pandas as pd
import numpy as np
import yaml
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

def validate_schema(df, name):
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
    ref_path = Path(cfg["paths"]["ref_data"])
    cur_path = Path(cfg["paths"]["current_data"])

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
