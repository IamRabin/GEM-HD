import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent

def save_parquet_or_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return alt

def synth_ref(n=1200, seed=11):
    rng = np.random.default_rng(seed)
    fixation_ms = rng.normal(180, 60, n).clip(40, 600)
    saccade_deg_s = rng.normal(100, 40, n).clip(5, 400)
    blink_hz = rng.normal(0.25, 0.15, n).clip(0.01, 1.2)
    entropy = rng.normal(1.2, 0.4, n).clip(0.0, 3.0)
    aoi = rng.beta(2, 3, n)

    def z(x): x=np.asarray(x); return (x-x.mean())/(x.std()+1e-6)
    score = ( 0.55*z(fixation_ms)
            - 0.35*z(saccade_deg_s)
            - 0.45*z(blink_hz)
            + 0.15*(-np.abs(z(entropy)))
            + 0.20*z(aoi) )
    y = 1/(1+np.exp(-score))
    y = np.clip(y + rng.normal(0, 0.08, n), 0, 1)

    ts = pd.date_range("2025-09-01", periods=n, freq="S")
    return pd.DataFrame({
        "timestamp": ts,
        "fixation_duration_ms": fixation_ms.astype(float),
        "saccade_velocity_deg_s": saccade_deg_s.astype(float),
        "blink_rate_hz": blink_hz.astype(float),
        "scanpath_entropy": entropy.astype(float),
        "aoi_hit_rate": aoi.astype(float),
        "engagement": y.astype(float)      # labels for training only
    })

def synth_current(m=120, seed=13):
    rng = np.random.default_rng(seed)
    fixation_ms = rng.normal(170, 70, m).clip(40, 600)
    saccade_deg_s = rng.normal(115, 45, m).clip(5, 400)
    blink_hz = rng.normal(0.30, 0.20, m).clip(0.01, 1.5)
    entropy = rng.normal(1.25, 0.45, m).clip(0.0, 3.0)
    aoi = rng.beta(2, 3, m)
    ts = pd.date_range("2025-10-26 14:00:00", periods=m, freq="S")
    return pd.DataFrame({
        "timestamp": ts,
        "fixation_duration_ms": fixation_ms.astype(float),
        "saccade_velocity_deg_s": saccade_deg_s.astype(float),
        "blink_rate_hz": blink_hz.astype(float),
        "scanpath_entropy": entropy.astype(float),
        "aoi_hit_rate": aoi.astype(float)
    })

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True, parents=True)
    ref = synth_ref()
    cur = synth_current()
    rpath = save_parquet_or_csv(ref, OUT / "ref.parquet")
    cpath = save_parquet_or_csv(cur, OUT / "current.parquet")
    print(f"Wrote: {rpath}")
    print(f"Wrote: {cpath}")
