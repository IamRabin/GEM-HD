#!/usr/bin/env python3
"""
Stream predictions into data/processed/current_with_pred.parquet so the
Streamlit dashboard can update continuously.

Workflow:
- Train a small scikit-learn MLP on ref.parquet (ema_engagement target or proxy)
- Every --interval seconds, check current.parquet for new rows and append
  predictions with y_std (fixed q from ref residuals) and ema_engagement (EMA)
"""

import argparse
import time
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
    fix = pd.to_numeric(df.get("fixation_duration_ms", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sac = pd.to_numeric(df.get("saccade_velocity_deg_s", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ent = pd.to_numeric(df.get("scanpath_entropy", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    def norm(x: np.ndarray) -> np.ndarray:
        minv, maxv = float(np.nanmin(x)), float(np.nanmax(x))
        if maxv > minv:
            return (x - minv) / (maxv - minv)
        return np.zeros_like(x)
    tgt = 0.6 * (1.0 - norm(ent)) + 0.4 * norm(sac)
    return np.clip(tgt, 0.0, 1.0)

def extract_X_with_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = []
    for c in cols:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            col = np.zeros(len(df), dtype=float)
        parts.append(col)
    X = np.stack(parts, axis=1) if parts else np.zeros((len(df), 0), dtype=float)
    return X

def conformal_half_width(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> float:
    res = np.abs(y_true - y_pred)
    if res.size == 0:
        return 0.1
    q = float(np.quantile(res, 1.0 - alpha, method="higher")) if hasattr(np, "quantile") else float(np.percentile(res, 100 * (1 - alpha)))
    return q

def main():
    ap = argparse.ArgumentParser(description="Stream predictions into current_with_pred.parquet")
    ap.add_argument("--ref", default="data/processed/ref.parquet")
    ap.add_argument("--current", default="data/processed/current.parquet")
    ap.add_argument("--output", default="data/processed/current_with_pred.parquet")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    ref_path = Path(args.ref)
    cur_path = Path(args.current)
    out_path = Path(args.output)

    # Train once
    ref_df = load_df(ref_path)
    # Ensure current exists; if not, derive from output (drop prediction cols)
    if not cur_path.exists() and out_path.exists():
        df = load_df(out_path)
        for c in ["y_pred", "y_std"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        save_parquet(df, cur_path)

    cur_df = load_df(cur_path) if cur_path.exists() else pd.DataFrame(columns=FEATS)
    y_ref = build_target(ref_df)
    feature_cols = [c for c in FEATS if (c in ref_df.columns) or (c in cur_df.columns)]
    X_ref = extract_X_with_cols(ref_df, feature_cols)
    mean_ref = X_ref.mean(axis=0)
    std_ref = X_ref.std(axis=0)
    std_ref[std_ref == 0] = 1.0
    X_ref_n = (X_ref - mean_ref) / std_ref

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=500,
        random_state=42, n_iter_no_change=20, verbose=False,
    )
    mlp.fit(X_ref_n, y_ref)
    y_ref_hat = mlp.predict(X_ref_n)
    q = conformal_half_width(y_ref, y_ref_hat, alpha=args.alpha)

    prev_len = 0
    last_ema = 0.0
    if out_path.exists():
        prev_df = load_df(out_path)
        prev_len = len(prev_df)
        if "ema_engagement" in prev_df.columns and len(prev_df) > 0:
            last_ema = float(pd.to_numeric(prev_df["ema_engagement"], errors="coerce").dropna().iloc[-1])

    print(f"[stream] q={q:.4f} interval={args.interval}s start_len={prev_len}")
    while True:
        try:
            cur_df = load_df(cur_path)
        except Exception:
            time.sleep(args.interval); continue

        n_cur = len(cur_df)
        if n_cur <= prev_len:
            time.sleep(args.interval); continue

        batch = cur_df.iloc[prev_len:n_cur].copy()
        X_cur = extract_X_with_cols(batch, feature_cols)
        X_cur_n = (X_cur - mean_ref) / std_ref
        y_hat = mlp.predict(X_cur_n).astype(float)
        y_std = np.full_like(y_hat, fill_value=q, dtype=float)

        beta = 0.85
        ema_vals = []
        m = last_ema
        for v in y_hat:
            m = beta * m + (1.0 - beta) * float(v)
            ema_vals.append(m)
        last_ema = m

        batch["y_pred"] = y_hat
        batch["y_std"] = y_std
        if "ema_engagement" not in batch.columns:
            batch["ema_engagement"] = np.array(ema_vals, dtype=float)

        if out_path.exists():
            prev_df = load_df(out_path)
            out_df = pd.concat([prev_df, batch], ignore_index=True)
        else:
            out_df = batch
        save_parquet(out_df, out_path)
        prev_len = n_cur
        print(f"[stream] appended {len(batch)} rows; total={len(out_df)}")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()


