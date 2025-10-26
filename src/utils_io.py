from pathlib import Path
import pandas as pd

def save_parquet_or_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return alt
    
def load_parquet_or_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)
    
def load_data(path: Path) -> pd.DataFrame:
    return load_parquet_or_csv(path)
    
def save_data(df: pd.DataFrame, path: Path) -> Path:
    return save_parquet_or_csv(df, path)