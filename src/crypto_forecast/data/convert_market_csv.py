from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crypto_forecast.utils.io import ensure_dir


NUMERIC_FALLBACK_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quotevolume",
    "activebuyvolume",
    "activebuyquotevolume",
    "tradenum",
]


def _load_one_csv(path: Path, ts_col: str, symbol_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"{path} missing timestamp column: {ts_col}")

    # Binance starttime is in milliseconds.
    df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)

    if symbol_col not in df.columns:
        # fallback: infer from filename prefix
        df[symbol_col] = path.name.split("_")[0]

    for c in df.columns:
        if c in NUMERIC_FALLBACK_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([symbol_col, "timestamp"]).reset_index(drop=True)
    return df


def _compute_logreturn(df: pd.DataFrame, symbol_col: str, price_col: str) -> pd.DataFrame:
    if price_col not in df.columns:
        raise ValueError(f"missing target_price_col: {price_col}")

    # logreturn_t = ln(close_t / close_{t-1})
    df["log_close"] = np.log(df[price_col].astype(float))
    df["target_logreturn"] = df.groupby(symbol_col)["log_close"].diff()
    return df


def convert_raw_to_processed(
    raw_dir: str,
    processed_dir: str,
    file_pattern: str,
    symbols: list[str],
    timestamp_col: str,
    symbol_col: str,
    target_price_col: str,
    keep_feature_cols: list[str],
) -> tuple[Path, list[Path]]:
    raw_root = Path(raw_dir)
    out_root = ensure_dir(processed_dir)

    all_files = sorted(raw_root.glob(file_pattern))
    if symbols:
        wanted = set(symbols)
        all_files = [p for p in all_files if p.name.split("_")[0] in wanted]

    if not all_files:
        raise FileNotFoundError(f"No files matched: dir={raw_dir}, pattern={file_pattern}, symbols={symbols}")

    per_symbol_paths: list[Path] = []
    frames: list[pd.DataFrame] = []

    for p in all_files:
        df = _load_one_csv(p, ts_col=timestamp_col, symbol_col=symbol_col)
        df = _compute_logreturn(df, symbol_col=symbol_col, price_col=target_price_col)

        wanted_cols = [symbol_col, "timestamp", "target_logreturn"]
        for col in keep_feature_cols:
            if col in df.columns and col not in wanted_cols:
                wanted_cols.append(col)

        out = df[wanted_cols].copy()
        out = out.dropna(subset=["target_logreturn"]).reset_index(drop=True)

        symbol = str(out[symbol_col].iloc[0])
        symbol_path = out_root / f"{symbol}_1d_logreturn.parquet"
        out.to_parquet(symbol_path, index=False)
        per_symbol_paths.append(symbol_path)
        frames.append(out)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values([symbol_col, "timestamp"]).reset_index(drop=True)
    combined_path = out_root / "combined_1d_logreturn.parquet"
    combined.to_parquet(combined_path, index=False)

    return combined_path, per_symbol_paths
