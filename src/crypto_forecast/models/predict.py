from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import re

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


@dataclass
class ParsedRunName:
    base_run_name: str
    weight_symbol: str
    init_mode: str
    loss_mode: str
    run_tag: str
    timestamp: str
    run_dir: Path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower() or "unknown"


def _parse_run_dir_name(run_dir: Path) -> ParsedRunName | None:
    parts = run_dir.name.split("__")
    # <base_run_name>__<SYMBOL>__<INIT_MODE>__<LOSS_MODE>__<RUN_TAG>__<TIMESTAMP>
    if len(parts) < 6:
        return None
    base = "__".join(parts[:-5])
    if not base:
        return None
    return ParsedRunName(
        base_run_name=base,
        weight_symbol=parts[-5],
        init_mode=parts[-4],
        loss_mode=parts[-3],
        run_tag=parts[-2],
        timestamp=parts[-1],
        run_dir=run_dir,
    )


def _latest_or_specific_local_ckpt_path(
    cfg: dict[str, Any],
    weight_symbol: str | None,
    init_mode: str | None,
    loss_mode: str | None,
    ckpt_timestamp: str | None,
    run_tag: str | None,
) -> tuple[Path, str]:
    ckpt_root = Path(cfg["paths"]["checkpoints_dir"])
    finetuned_ckpt_name = cfg["train"]["finetuned_ckpt_name"]
    base_run_name = cfg["project"]["run_name"]

    candidates: list[ParsedRunName] = []
    if ckpt_root.exists():
        for run_dir in ckpt_root.iterdir():
            if not run_dir.is_dir():
                continue
            parsed = _parse_run_dir_name(run_dir)
            if parsed is None:
                continue
            if not (run_dir / finetuned_ckpt_name).exists():
                continue
            if parsed.base_run_name != base_run_name:
                continue
            if weight_symbol and parsed.weight_symbol != weight_symbol:
                continue
            if init_mode and parsed.init_mode != init_mode:
                continue
            if loss_mode and parsed.loss_mode != loss_mode:
                continue
            if run_tag and parsed.run_tag != run_tag:
                continue
            if ckpt_timestamp and parsed.timestamp != ckpt_timestamp:
                continue
            candidates.append(parsed)

    if candidates:
        chosen = sorted(candidates, key=lambda x: x.timestamp)[-1]
        return chosen.run_dir / finetuned_ckpt_name, chosen.timestamp

    requested_filtered_search = any([weight_symbol, init_mode, loss_mode, ckpt_timestamp, run_tag])
    if requested_filtered_search:
        raise FileNotFoundError(
            "No local finetuned checkpoint matched filters under "
            f"{ckpt_root}. filters="
            f"(base_run_name={base_run_name}, weight_symbol={weight_symbol}, "
            f"init_mode={init_mode}, loss_mode={loss_mode}, run_tag={run_tag}, "
            f"ckpt_timestamp={ckpt_timestamp})"
        )

    # Backward-compatible fallback to explicit ckpt_subdir/default legacy path.
    ckpt_subdir = cfg["infer"].get("ckpt_subdir")
    if ckpt_subdir:
        fallback = ckpt_root / ckpt_subdir
    else:
        fallback = ckpt_root / base_run_name / finetuned_ckpt_name
    if not fallback.exists():
        raise FileNotFoundError(
            "Default local checkpoint path does not exist. "
            f"checked={fallback}. "
            "Set infer.ckpt_subdir, or pass explicit --model-ref, or provide local checkpoint filters."
        )
    return fallback, "unknown"


def _resolve_model_ref(
    cfg: dict[str, Any],
    model_source: ModelSource,
    model_ref: str | None = None,
    weight_symbol: str | None = None,
    init_mode: str | None = None,
    loss_mode: str | None = None,
    ckpt_timestamp: str | None = None,
    run_tag: str | None = None,
) -> str:
    """
    Resolve model reference for Chronos2Pipeline.from_pretrained().

    Note:
    - run_infer.py handles loading-mode priority and may pass `model_ref=None`
      for local/hf modes by design.
    - `model_ref` is used here only when manual mode passes it through.
    """
    if model_ref:
        return model_ref

    # Local finetuned checkpoint resolution (latest or specific timestamp).
    if model_source == "local":
        local_ref, _ = _latest_or_specific_local_ckpt_path(
            cfg=cfg,
            weight_symbol=weight_symbol,
            init_mode=init_mode,
            loss_mode=loss_mode,
            ckpt_timestamp=ckpt_timestamp,
            run_tag=run_tag,
        )
        return str(local_ref)

    # HF zero-shot resolution.
    else:  # model_source == "hf"
        # Defaults to infer.hf_model_id then model.model_id
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
    weight_symbol: str | None = None,
    infer_symbol: str | None = None,
    init_mode: str | None = None,
    loss_mode: str | None = None,
    ckpt_timestamp: str | None = None,
    run_tag: str | None = None,
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

    resolved_ckpt_timestamp = "na"
    if model_source == "local" and model_ref is None:
        _, resolved_ckpt_timestamp = _latest_or_specific_local_ckpt_path(
            cfg=cfg,
            weight_symbol=weight_symbol,
            init_mode=init_mode,
            loss_mode=loss_mode,
            ckpt_timestamp=ckpt_timestamp,
            run_tag=run_tag,
        )

    resolved_model_ref = _resolve_model_ref(
        cfg=cfg,
        model_source=model_source,
        model_ref=model_ref,
        weight_symbol=weight_symbol,
        init_mode=init_mode,
        loss_mode=loss_mode,
        ckpt_timestamp=ckpt_timestamp,
        run_tag=run_tag,
    )
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

    target_symbol = infer_symbol or weight_symbol or "unknown"
    run_ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    # Output naming is intentionally explicit for downstream filtering.
    if model_source == "hf":
        hf_ref = model_ref or cfg["infer"].get("hf_model_id") or cfg["model"]["model_id"]
        out_name = f"infer__hf__{_slug(str(hf_ref))}__target_{_slug(target_symbol)}__{run_ts}"
    else:
        w_sym = _slug(weight_symbol or "unknown")
        i_mode = _slug(init_mode or cfg["model"].get("init_mode", "unknown"))
        l_mode = _slug(loss_mode or cfg["model"].get("loss_mode", "unknown"))
        ckpt_ts = _slug(resolved_ckpt_timestamp)
        out_name = (
            f"infer__ft__w_{w_sym}__i_{i_mode}__l_{l_mode}"
            f"__ckpt_{ckpt_ts}__target_{_slug(target_symbol)}__{run_ts}"
        )
    if output_tag:
        out_name = f"{out_name}__tag_{_slug(output_tag)}"

    file_target = _slug(target_symbol)
    file_init_mode = _slug(init_mode or cfg["model"].get("init_mode", "na"))
    file_loss_mode = _slug(loss_mode or cfg["model"].get("loss_mode", "na"))
    file_run_tag = _slug(run_tag or output_tag or "na")

    filename = (
        f"predictions_decision_aligned"
        f"__target_{file_target}"
        f"__init_{file_init_mode}"
        f"__loss_{file_loss_mode}"
        f"__tag_{file_run_tag}.csv"
    )

    out_dir = ensure_dir(Path(cfg["paths"]["predictions_dir"]) / out_name)
    out_path = out_dir / filename
    out_df = pd.DataFrame(rows).sort_values(["symbol", "ts_decision"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    return out_path
