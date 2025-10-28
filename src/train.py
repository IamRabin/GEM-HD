#!/usr/bin/env python3
"""
Mental Health Detection Training Script
Trains gaze engagement models for early mental health detection
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from utils_io import load_parquet_or_csv, save_parquet_or_csv
from model import create_model, gpu_status_str
from split_conformal import SplitConformalRegressor

FEATS = ["fixation_duration_ms", "saccade_velocity_deg_s", "blink_rate_hz", "scanpath_entropy", "aoi_hit_rate"]

def mc_predict(model, X_np, device, T=30, batch=256):
    """Monte Carlo prediction with dropout for uncertainty estimation"""
    model.train()  # enable dropout
    with torch.no_grad():
        n = X_np.shape[0]
        acc = []
        for _ in range(T):
            preds = []
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

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ref = load_parquet_or_csv(Path(args.ref_path))
    
    # Extract features
    X = ref[FEATS].values.astype(np.float32)
    y = ref["engagement"].values.astype(np.float32)
    
    # Train/validation/test split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X)); rng.shuffle(idx)
    n_train = int(0.7 * len(idx))
    n_calib = int(0.1 * len(idx))
    tr = idx[:n_train]
    ca = idx[n_train:n_train+n_calib]
    te = idx[n_train+n_calib:]
    
    # Normalize features
    X_mean = X[tr].mean(axis=0, keepdims=True)
    X_std  = X[tr].std(axis=0, keepdims=True) + 1e-6
    Xn = (X - X_mean) / X_std
    Xtr, ytr = Xn[tr], y[tr]
    Xca, yca = Xn[ca], y[ca]
    Xte, yte = Xn[te], y[te]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(gpu_status_str())
    
    # Create model
    model_kwargs = {
        'tabular_feature_dim': Xtr.shape[1],
        'hidden_dim': args.hidden,
        'dropout': args.dropout,
        'freeze_gaze': args.freeze_gaze
    }
    
    model = create_model(args.model_type, **model_kwargs).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    
    def batches(X, y, bs=128):
        n = X.shape[0]
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            j = perm[i:i+bs]
            yield X[j], y[j]
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for ep in range(1, args.epochs+1):
        model.train()
        ep_loss = 0.0
        for xb_np, yb_np in batches(Xtr, ytr, bs=args.batch_size):
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(yb_np).to(device)
            
            opt.zero_grad()
            # Handle different model interfaces
            if hasattr(model, 'forward') and model.__class__.__name__ == 'TinyMLP':
                pred = model(xb)
            else:
                pred = model(xb, None)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb_np)
        
        # Validation
        if ep % max(1, args.epochs//3) == 0 or ep == 1:
            with torch.no_grad():
                model.eval()
                xt = torch.from_numpy(Xte).to(device)
                yt = torch.from_numpy(yte).to(device)
                # Handle different model interfaces
                if hasattr(model, 'forward') and model.__class__.__name__ == 'TinyMLP':
                    test_mae = F.l1_loss(model(xt), yt).item()
                else:
                    test_mae = F.l1_loss(model(xt, None), yt).item()
            print(f"Epoch {ep}/{args.epochs} | Train MAE {ep_loss/len(Xtr):.4f} | Test MAE {test_mae:.4f}")
    
    # Comprehensive uncertainty estimation
    print("\nComputing comprehensive uncertainty estimates...")
    
    # Epistemic uncertainty (MC Dropout)
    print("  - Estimating epistemic uncertainty (MC Dropout)...")
    yhat_te, epistemic_std = mc_predict(model, Xte, device, T=args.mc_passes)
    
    # Conformal prediction calibration with uncertainty
    print("  - Calibrating conformal prediction with uncertainty analysis...")
    yhat_ca, epistemic_std_ca = mc_predict(model, Xca, device, T=args.mc_passes)
    
    cp = SplitConformalRegressor(alpha=args.alpha)
    cp.calibrate_with_uncertainty(yhat_ca, yca, epistemic_std_ca)
    
    # Set epistemic uncertainty for test set correlation analysis
    cp.epistemic_std = epistemic_std
    
    # Coverage analysis
    coverage_stats = cp.compute_coverage(yte, yhat_te)
    coverage = coverage_stats['overall_coverage']
    width = coverage_stats['interval_width']
    
    # Correlation analysis
    correlation_stats = cp.analyze_uncertainty_correlation(yte, yhat_te)
    
    # Save model
    model_save_dict = {
        "state_dict": model.state_dict(),
        "x_mean": X_mean, "x_std": X_std,
        "feat_cols": FEATS,
        "alpha": args.alpha,
        "q_conformal": cp.q_,
        "model_type": args.model_type,
        "model_kwargs": model_kwargs,
        "uncertainty_params": {
            "conformal_q": cp.q_,
            "aleatoric_stats": cp.aleatoric_stats,
            "epistemic_available": True
        }
    }
    
    torch.save(model_save_dict, out_dir / "mental_health_model.pt")
    
    # Write training report
    report = {
        "gpu": gpu_status_str(),
        "model_type": args.model_type,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "n_train": int(len(tr)), "n_calib": int(len(ca)), "n_test": int(len(te)),
        "test_mae": cp.aleatoric_stats['mae'],
        "test_mse": cp.aleatoric_stats['mse'],
        "test_rmse": cp.aleatoric_stats['rmse'],
        "uncertainty_analysis": {
            "epistemic": {
                "mean_std": float(np.mean(epistemic_std)),
                "std_std": float(np.std(epistemic_std))
            },
            "aleatoric": cp.aleatoric_stats,
            "conformal": {
                "alpha": float(args.alpha),
                "quantile": float(cp.q_),
                "coverage": coverage,
                "interval_width": width
            },
            "correlations": correlation_stats
        },
        "features": FEATS,
        "freeze_gaze": args.freeze_gaze,
    }
    
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate comprehensive uncertainty report
    comprehensive_report = cp.get_comprehensive_report(yte, yhat_te)
    with open(out_dir / "uncertainty_report.json", "w") as f:
        json.dump(comprehensive_report, f, indent=2)
    
    print("\n== Training done ==")
    print(f"Model: {args.model_type}")
    print(f"MAE={cp.aleatoric_stats['mae']:.4f} | MSE={cp.aleatoric_stats['mse']:.4f} | RMSE={cp.aleatoric_stats['rmse']:.4f}")
    print(f"Coverage@{int((1-args.alpha)*100)}%={coverage:.3f} | Width={width:.4f}")
    print(f"Epistemic uncertainty: {np.mean(epistemic_std):.4f} ± {np.std(epistemic_std):.4f}")
    print(f"Conformal quantile: {cp.q_:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Mental Health Detection Model")
    p.add_argument("--ref_path", type=str, default="ref.parquet")
    p.add_argument("--output_dir", type=str, default="artifacts")
    p.add_argument("--model_type", type=str, default="tiny", choices=["tiny", "multimodal"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--mc_passes", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_gaze", action="store_true", default=True, help="Freeze gaze extractor initially")
    args = p.parse_args()
    main(args)