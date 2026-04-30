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
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def split_by_time(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> SplitData:
    train_start_ts = pd.Timestamp(train_start, tz="UTC")
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_start_ts = pd.Timestamp(val_start, tz="UTC")
    val_end_ts = pd.Timestamp(val_end, tz="UTC")
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    test_end_ts = pd.Timestamp(test_end, tz="UTC")

    if train_start_ts > train_end_ts:
        raise ValueError(
            f"Expected train_start <= train_end, but got train_start={train_start_ts}, train_end={train_end_ts}."
        )
    if val_start_ts > val_end_ts:
        raise ValueError(
            f"Expected val_start <= val_end, but got val_start={val_start_ts}, val_end={val_end_ts}."
        )
    if test_start_ts > test_end_ts:
        raise ValueError(
            f"Expected test_start <= test_end, but got test_start={test_start_ts}, test_end={test_end_ts}."
        )
    if not (train_end_ts < val_start_ts and val_end_ts < test_start_ts):
        raise ValueError(
            "Expected non-overlapping ordered splits: train_end < val_start and val_end < test_start, "
            f"but got train=[{train_start_ts}, {train_end_ts}], "
            f"val=[{val_start_ts}, {val_end_ts}], test=[{test_start_ts}, {test_end_ts}]."
        )

    train_mask = (df["timestamp"] >= train_start_ts) & (df["timestamp"] <= train_end_ts)
    val_mask = (df["timestamp"] >= val_start_ts) & (df["timestamp"] <= val_end_ts)
    test_mask = (df["timestamp"] >= test_start_ts) & (df["timestamp"] <= test_end_ts)

    return SplitData(
        train_df=df.loc[train_mask].copy(),
        val_df=df.loc[val_mask].copy(),
        test_df=df.loc[test_mask].copy(),
        train_start=train_start_ts,
        train_end=train_end_ts,
        val_start=val_start_ts,
        val_end=val_end_ts,
        test_start=test_start_ts,
        test_end=test_end_ts,
    )


def _to_task_dict(g: pd.DataFrame, target_col: str, cov_cols: list[str]) -> dict[str, Any]:
    target = g[target_col].to_numpy(dtype=np.float32)
    out: dict[str, Any] = {"target": target} # could be like (n_variates, history_length) in chronos predict() inputs dict format, but here we just use 1D as the target col is only logreturn, so (history_length,)

    cov_cols = [c for c in cov_cols if c in g.columns and c != target_col]
    if cov_cols:
        out["past_covariates"] = {c: g[c].to_numpy(dtype=np.float32) for c in cov_cols} # a dict, each k-v like: {covariate name : `torch.Tensor` or `np.ndarray` of shape (history_length,)}

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
