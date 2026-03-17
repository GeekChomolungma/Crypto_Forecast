from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def split_by_time(df: pd.DataFrame, train_end: str, val_end: str) -> SplitData:
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts = pd.Timestamp(val_end, tz="UTC")

    train_df = df[df["timestamp"] <= train_end_ts].copy()
    val_df = df[(df["timestamp"] > train_end_ts) & (df["timestamp"] <= val_end_ts)].copy()
    test_df = df[df["timestamp"] > val_end_ts].copy()
    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df)


def _to_task_dict(g: pd.DataFrame, target_col: str, cov_cols: list[str]) -> dict[str, Any]:
    target = g[target_col].to_numpy(dtype=np.float32)
    out: dict[str, Any] = {"target": target}

    cov_cols = [c for c in cov_cols if c in g.columns and c != target_col]
    if cov_cols:
        out["past_covariates"] = {c: g[c].to_numpy(dtype=np.float32) for c in cov_cols}

    return out


def build_tasks_for_fit(df: pd.DataFrame, symbol_col: str, target_col: str, cov_cols: list[str], min_history: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for _, g in df.groupby(symbol_col):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < min_history:
            continue
        tasks.append(_to_task_dict(g=g, target_col=target_col, cov_cols=cov_cols))

    if not tasks:
        raise ValueError("No tasks built for fit. Check split dates/min_history.")
    return tasks
