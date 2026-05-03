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


def get_target_col(data_cfg: dict[str, Any]) -> str:
    target_col = data_cfg.get("target_col")
    if not isinstance(target_col, str) or not target_col:
        raise ValueError("Missing required config key: data.target_col")
    return target_col


def get_time_col(data_cfg: dict[str, Any]) -> str:
    time_col = data_cfg.get("time_col")
    if not isinstance(time_col, str) or not time_col:
        raise ValueError("Missing required config key: data.time_col")
    return time_col


def get_past_covariate_cols(data_cfg: dict[str, Any], target_col: str) -> list[str]:
    if "past_covariates" not in data_cfg:
        if "keep_feature_cols" in data_cfg:
            raise ValueError(
                "Config key data.keep_feature_cols has been renamed to data.past_covariates. "
                "Please update configs/experiment.yaml."
            )
        raise ValueError("Missing required config key: data.past_covariates")

    raw_cols = data_cfg["past_covariates"]
    if raw_cols is None:
        return []
    if not isinstance(raw_cols, list) or not all(isinstance(c, str) for c in raw_cols):
        raise ValueError("Expected data.past_covariates to be a list of column names.")

    duplicate_cols = sorted({c for c in raw_cols if raw_cols.count(c) > 1})
    if duplicate_cols:
        raise ValueError(f"Duplicate columns in data.past_covariates: {duplicate_cols}")
    if target_col in raw_cols:
        raise ValueError(
            f"data.past_covariates must not include target column {target_col!r}; "
            "it is passed to Chronos-2 as the target."
        )
    return list(raw_cols)


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    *,
    processed_path: str | None = None,
    context: str,
) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if not missing:
        return

    location = f" in {processed_path}" if processed_path else ""
    available_preview = ", ".join(map(str, df.columns[:30]))
    if len(df.columns) > 30:
        available_preview += ", ..."
    raise ValueError(
        f"Missing required column(s) for {context}{location}: {missing}. "
        f"Available columns: [{available_preview}]"
    )


def ensure_timestamp_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if "timestamp" in df.columns:
        return df
    df["timestamp"] = pd.to_datetime(df[time_col], utc=True)
    return df


def _to_task_dict(g: pd.DataFrame, target_col: str, cov_cols: list[str]) -> dict[str, Any]:
    target = g[target_col].to_numpy(dtype=np.float32)
    out: dict[str, Any] = {"target": target} # could be like (n_variates, history_length) in chronos predict() inputs dict format, but here we just use 1D as the target col is only logreturn, so (history_length,)

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
