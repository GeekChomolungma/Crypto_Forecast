from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from chronos import Chronos2Pipeline

from crypto_forecast.data.build_chronos_inputs import split_by_time
from crypto_forecast.utils.io import ensure_dir


@dataclass
class DecisionAlignedRow:
    symbol: str
    ts_decision: pd.Timestamp


def _default_ckpt_path(cfg: dict[str, Any]) -> Path:
    run_name = cfg["project"]["run_name"]
    ckpt_subdir = cfg["infer"].get("ckpt_subdir")
    if ckpt_subdir:
        return Path(cfg["paths"]["checkpoints_dir"]) / ckpt_subdir
    return Path(cfg["paths"]["checkpoints_dir"]) / run_name / cfg["train"]["finetuned_ckpt_name"]


def _load_pipeline(cfg: dict[str, Any], ckpt_path: Path) -> Chronos2Pipeline:
    model_cfg = cfg["model"]
    return Chronos2Pipeline.from_pretrained(
        str(ckpt_path),
        device_map=model_cfg["device_map"],
        torch_dtype=model_cfg["torch_dtype"],
    )


def generate_decision_aligned_predictions(cfg: dict[str, Any], processed_path: Path) -> Path:
    infer_cfg = cfg["infer"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    df = pd.read_parquet(processed_path).sort_values([data_cfg["symbol_col"], "timestamp"]).reset_index(drop=True)
    split = split_by_time(df, train_end=cfg["split"]["train_end"], val_end=cfg["split"]["val_end"])

    # Use train+val as available history; evaluate on test decision points.
    history_df = pd.concat([split.train_df, split.val_df], axis=0, ignore_index=True)
    test_df = split.test_df

    ckpt_path = _default_ckpt_path(cfg)
    pipeline = _load_pipeline(cfg, ckpt_path)

    horizon = int(infer_cfg["horizon"])
    q_levels = list(model_cfg["quantile_levels"])
    min_history = int(data_cfg["min_history"])

    rows: list[dict[str, Any]] = []

    for symbol, g_all in df.groupby(data_cfg["symbol_col"]):
        g_all = g_all.sort_values("timestamp").reset_index(drop=True)
        decision_mask = g_all["timestamp"].isin(test_df[test_df[data_cfg["symbol_col"]] == symbol]["timestamp"])
        decision_idx = g_all.index[decision_mask].tolist()

        for i in decision_idx:
            # Need next-step ground truth availability.
            if i + horizon >= len(g_all):
                continue
            if i < min_history:
                continue

            hist = g_all.iloc[: i + 1].copy()

            target_hist = torch.tensor(hist["target_logreturn"].to_numpy(dtype="float32"))
            past_covs = {
                c: torch.tensor(hist[c].to_numpy(dtype="float32"))
                for c in data_cfg["keep_feature_cols"]
                if c in hist.columns and c != "target_logreturn"
            }

            inputs = [{"target": target_hist, "past_covariates": past_covs}]
            quantiles_list, mean_list = pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=horizon,
                quantile_levels=q_levels,
                batch_size=int(infer_cfg["batch_size"]),
                context_length=int(model_cfg["context_length"]),
                cross_learning=bool(model_cfg["cross_learning"]),
                limit_prediction_length=False,
            )

            q_tensor = quantiles_list[0][0]  # (horizon, num_quantiles)
            m_tensor = mean_list[0][0]       # (horizon,)

            row: dict[str, Any] = {
                "symbol": symbol,
                "ts_decision": g_all.loc[i, "timestamp"],
            }

            # Current-time (t) observable features.
            for c in data_cfg["keep_feature_cols"]:
                if c in g_all.columns:
                    row[f"{c}_t"] = g_all.loc[i, c]

            for step in range(1, horizon + 1):
                row[f"pred_ret_t+{step}_mean"] = float(m_tensor[step - 1].item())
                row[f"y_true_ret_t+{step}"] = float(g_all.loc[i + step, "target_logreturn"])

                for q_idx, q in enumerate(q_levels):
                    row[f"pred_ret_t+{step}_q{q}"] = float(q_tensor[step - 1, q_idx].item())

            rows.append(row)

    out_dir = ensure_dir(Path(cfg["paths"]["predictions_dir"]) / cfg["project"]["run_name"])
    out_path = out_dir / "predictions_decision_aligned.csv"
    out_df = pd.DataFrame(rows).sort_values(["symbol", "ts_decision"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    return out_path
