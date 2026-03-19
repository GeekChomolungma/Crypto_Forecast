from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from chronos import Chronos2Pipeline

from crypto_forecast.data.build_chronos_inputs import split_by_time
from crypto_forecast.utils.io import ensure_dir


@dataclass
class DecisionAlignedRow:
    symbol: str
    ts_decision: pd.Timestamp


ModelSource = Literal["local", "hf"]


def _default_ckpt_path(cfg: dict[str, Any]) -> Path:
    run_name = cfg["project"]["run_name"]
    ckpt_subdir = cfg["infer"].get("ckpt_subdir")
    if ckpt_subdir:
        return Path(cfg["paths"]["checkpoints_dir"]) / ckpt_subdir
    return Path(cfg["paths"]["checkpoints_dir"]) / run_name / cfg["train"]["finetuned_ckpt_name"]


def _resolve_model_ref(
    cfg: dict[str, Any],
    model_source: ModelSource,
    model_ref: str | None = None,
) -> str:
    """
    Resolve the actual model reference for Chronos2Pipeline.from_pretrained().

    Priority:
    1) explicit `model_ref` argument (local path or HF model id)
    2) `model_source == "local"`: use default fine-tuned checkpoint path
    3) `model_source == "hf"`: use infer.hf_model_id, fallback to model.model_id
    """
    if model_ref:
        return model_ref

    if model_source == "local":
        return str(_default_ckpt_path(cfg))
    else:  # model_source == "hf"
        # HuggingFace mode defaults to infer.hf_model_id then model.model_id
        return str(cfg["infer"].get("hf_model_id") or cfg["model"]["model_id"])


def _load_pipeline(cfg: dict[str, Any], model_ref: str) -> Chronos2Pipeline:
    model_cfg = cfg["model"]
    return Chronos2Pipeline.from_pretrained(
        model_ref,
        device_map=model_cfg["device_map"],
        torch_dtype=model_cfg["torch_dtype"],
    )


def generate_decision_aligned_predictions(
    cfg: dict[str, Any],
    processed_path: Path,
    model_source: ModelSource = "local",
    model_ref: str | None = None,
    output_tag: str | None = None,
) -> Path:
    infer_cfg = cfg["infer"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    split_cfg = cfg["split"]

    df = pd.read_parquet(processed_path).sort_values([data_cfg["symbol_col"], "timestamp"]).reset_index(drop=True)
    split = split_by_time(
        df,
        train_end=split_cfg["train_end"],
        val_end=split_cfg["val_end"],
        train_start=split_cfg.get("train_start"),
    )

    # Use train+val as available history; evaluate on test decision points.
    history_df = pd.concat([split.train_df, split.val_df], axis=0, ignore_index=True)
    test_df = split.test_df

    _ = history_df  # reserved for future explicit history/test slicing controls

    resolved_model_ref = _resolve_model_ref(cfg=cfg, model_source=model_source, model_ref=model_ref)
    pipeline = _load_pipeline(cfg, resolved_model_ref)

    horizon = int(infer_cfg["horizon"])
    q_levels = list(model_cfg["quantile_levels"])
    min_history = int(data_cfg["min_history"])

    rows: list[dict[str, Any]] = []

    train_start = split_cfg.get("train_start")
    train_start_ts = pd.Timestamp(train_start, tz="UTC") if train_start is not None else None

    for symbol, g_all in df.groupby(data_cfg["symbol_col"]):
        g_all = g_all.sort_values("timestamp").reset_index(drop=True)
        # Align inference history window with split.train_start when provided.
        if train_start_ts is not None:
            g_all = g_all[g_all["timestamp"] >= train_start_ts].reset_index(drop=True)

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

            q_tensor = quantiles_list[0][0]  # (horizon, num_quantiles) or (prediction_length, len(quantile_levels))
            m_tensor = mean_list[0][0]       # (horizon,) or (prediction_length,)

            row: dict[str, Any] = {
                "symbol": symbol,
                "ts_decision": g_all.loc[i, "timestamp"],
            }

            # Current-time (t) observable features.
            for c in data_cfg["keep_feature_cols"]:
                if c in g_all.columns:
                    row[f"{c}_t"] = g_all.loc[i, c]
            
            # Future-time (t+1, t+2, ...) predictions and ground truths.
            for step in range(1, horizon + 1):
                row[f"pred_ret_t+{step}_mean"] = float(m_tensor[step - 1].item()) # predicted date 2025.01.02
                row[f"y_true_ret_t+{step}"] = float(g_all.loc[i + step, "target_logreturn"]) # ground truth date 2025.01.02

                for q_idx, q in enumerate(q_levels):
                    row[f"pred_ret_t+{step}_q{q}"] = float(q_tensor[step - 1, q_idx].item())

            rows.append(row)

    base_name = cfg["project"]["run_name"]
    source_suffix = "zeroshot_hf" if model_source == "hf" else "finetuned_local"
    out_name = f"{base_name}__{source_suffix}"
    if output_tag:
        out_name = f"{out_name}__{output_tag}"

    out_dir = ensure_dir(Path(cfg["paths"]["predictions_dir"]) / out_name)
    out_path = out_dir / f"{output_tag}_predictions_decision_aligned.csv"
    out_df = pd.DataFrame(rows).sort_values(["symbol", "ts_decision"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    return out_path
