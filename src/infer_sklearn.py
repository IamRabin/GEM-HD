#!/usr/bin/env python3
"""
Lightweight inference bridge using scikit-learn to generate
current_with_pred.parquet with required columns (y_pred, y_std, ema_engagement).

It trains a small MLPRegressor on ref.parquet using ema_engagement as the
target when available; otherwise, it derives a stable proxy from the features.
Uncertainty is estimated with a split-conformal style residual quantile on the
reference set.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


FEATS = [
    "fixation_duration_ms",
    "saccade_velocity_deg_s",
    "blink_rate_hz",
    "scanpath_entropy",
    "aoi_hit_rate",
]


def load_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def build_target(df: pd.DataFrame) -> np.ndarray:
    if "ema_engagement" in df.columns:
        tgt = pd.to_numeric(df["ema_engagement"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return tgt
    # Deterministic proxy if ema_engagement missing
    # Higher fixation and lower entropy ⇒ higher engagement; add small blink/saccade terms
    fix = pd.to_numeric(df.get("fixation_duration_ms", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sac = pd.to_numeric(df.get("saccade_velocity_deg_s", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    blink = pd.to_numeric(df.get("blink_rate_hz", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ent = pd.to_numeric(df.get("scanpath_entropy", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # Normalize robustly to [0,1] when possible
    def norm(x: np.ndarray) -> np.ndarray:
        minv, maxv = float(np.nanmin(x)), float(np.nanmax(x))
        if maxv > minv:
            return (x - minv) / (maxv - minv)
        return np.zeros_like(x)
    tgt = 0.6 * (1.0 - norm(ent)) + 0.3 * norm(sac) + 0.1 * norm(fix)
    return np.clip(tgt, 0.0, 1.0)


def extract_X(df: pd.DataFrame) -> np.ndarray:
    # keep numeric features only; fill NA
    fcols = [c for c in FEATS if c in df.columns]
    X = df[fcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # Replace NaNs from aoi_hit_rate if entirely missing
    X[np.isnan(X)] = 0.0
    return X


def conformal_half_width(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> float:
    res = np.abs(y_true - y_pred)
    if res.size == 0:
        return 0.1
    q = float(np.quantile(res, 1.0 - alpha, method="higher")) if hasattr(np, "quantile") else float(np.percentile(res, 100 * (1 - alpha)))
    return q


def train_and_predict(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    alpha: float = 0.1,
) -> Tuple[pd.DataFrame, float]:
    # Targets
    y_ref = build_target(ref_df)
    X_ref = extract_X(ref_df)
    X_cur = extract_X(cur_df)

    # Basic scaler: z-score using ref statistics
    mean_ref = X_ref.mean(axis=0)
    std_ref = X_ref.std(axis=0)
    std_ref[std_ref == 0] = 1.0
    X_ref_n = (X_ref - mean_ref) / std_ref
    X_cur_n = (X_cur - mean_ref) / std_ref

    # Small MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
        n_iter_no_change=20,
        verbose=False,
    )
    mlp.fit(X_ref_n, y_ref)

    # In-sample preds for residual quantile
    y_ref_hat = mlp.predict(X_ref_n)
    q = conformal_half_width(y_ref, y_ref_hat, alpha=alpha)

    # Predict current
    y_cur_hat = mlp.predict(X_cur_n).astype(float)
    y_std = np.full_like(y_cur_hat, fill_value=q, dtype=float)

    # Assemble output
    out = cur_df.copy()
    out["y_pred"] = y_cur_hat
    out["y_std"] = y_std
    # Pass through ema_engagement if present; otherwise synthesize a smooth EMA of predictions
    if "ema_engagement" not in out.columns:
        beta = 0.85
        m = 0.0
        ema = []
        for v in y_cur_hat:
            m = beta * m + (1.0 - beta) * float(v)
            ema.append(m)
        out["ema_engagement"] = np.array(ema, dtype=float)
    return out, q


def main():
    ap = argparse.ArgumentParser(description="Train small MLP on ref and predict current; write current_with_pred.parquet")
    ap.add_argument("--ref", default="data/processed/ref.parquet")
    ap.add_argument("--current", default="data/processed/current.parquet")
    ap.add_argument("--output", default="data/processed/current_with_pred.parquet")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    ref_df = load_df(args.ref)
    cur_df = load_df(args.current)

    out_df, q = train_and_predict(ref_df, cur_df, alpha=args.alpha)
    path = save_parquet(out_df, args.output)
    print(f"Wrote predictions to {path}")
    print(f"Conformal half-width (q_{{1-alpha}}) ≈ {q:.4f} at alpha={args.alpha}")


if __name__ == "__main__":
    main()


