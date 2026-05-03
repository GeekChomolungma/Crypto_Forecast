from __future__ import annotations

import argparse
from pathlib import Path

from crypto_forecast.config import load_config
from crypto_forecast.data.intervals import resolve_interval
from crypto_forecast.models.predict import generate_decision_aligned_predictions


def _resolve_model_loading(
    args: argparse.Namespace,
    cfg: dict,
) -> tuple[str, str | None]:
    """
    Resolve model loading mode with explicit priority:

    1) local finetuned checkpoint mode
    2) hf Chronos-2 zero-shot mode
    3) manual --model-ref mode
    """
    model_source = args.model_source or cfg["infer"].get("model_source", "local")
    if model_source in {"local", "hf"}:
        # In local/hf modes, we intentionally ignore --model-ref to keep behavior explicit.
        return model_source, None

    # model_source == "manual"
    if not args.model_ref:
        raise ValueError("model_source=manual requires --model-ref.")

    # For output naming: if manual ref is a local path => treat as local, else treat as hf-like id.
    manual_path = Path(args.model_ref)
    manual_source = "local" if manual_path.exists() else "hf"
    return manual_source, args.model_ref


def _resolve_processed_path(args: argparse.Namespace, cfg: dict, infer_symbol: str | None, interval: str) -> Path:
    """
    Resolve inference parquet with explicit rule:

    - If --processed is provided, use it directly.
    - Otherwise auto-generate from infer_symbol:
      <processed_dir>/<infer_symbol>_<interval>_Binance_with_indicators.parquet
    """
    if args.processed:
        return Path(args.processed)
    if not infer_symbol:
        raise ValueError(
            "When --processed is not provided, please set --infer-symbol "
            "(or --weight-symbol to reuse as target symbol)."
        )
    return Path(cfg["paths"]["processed_dir"]) / f"{infer_symbol}_{interval}_Binance_with_indicators.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run decision-aligned inference for crypto log-return forecasting.\n\n"
            "Model loading priority:\n"
            "1) local finetuned checkpoint mode\n"
            "2) hf Chronos-2 zero-shot mode\n"
            "3) manual --model-ref mode"
        )
    )

    # Global / common args
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment yaml config.",
    )
    parser.add_argument(
        "--processed",
        type=str,
        default=None,
        help=(
            "Path to processed parquet for inference. "
            "If omitted, defaults to <processed_dir>/<infer_symbol>_<interval>_Binance_with_indicators.parquet "
            "(or weight_symbol when infer_symbol is omitted)."
        ),
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help=(
            "Optional suffix appended to prediction output directory name. "
            "Useful for batch experiments to avoid output overwrite."
        ),
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="Optional interval override. Defaults to data.interval in config.",
    )

    # --------------------------
    # Mode 1: local (highest priority)
    # --------------------------
    # Load local finetuned checkpoint by filters:
    # weight_symbol + init_mode + loss_mode (+ run_tag, + ckpt_timestamp)
    local_group = parser.add_argument_group("Local Mode Args")
    local_group.add_argument(
        "--weight-symbol",
        type=str,
        default=None,
        help="Symbol used to select finetuned checkpoint (e.g. BTCUSDT).",
    )
    local_group.add_argument(
        "--infer-symbol",
        type=str,
        default=None,
        help="Symbol used for processed parquet selection (target prediction symbol).",
    )
    local_group.add_argument(
        "--init-mode",
        type=str,
        choices=["pretrained", "random"],
        default=None,
        help="Init mode used to filter local finetuned checkpoints.",
    )
    local_group.add_argument(
        "--loss-mode",
        type=str,
        choices=["native", "weighted_extreme_time_decay", "magnitude_weighted", "directional_hybrid"],
        default=None,
        help="Loss mode used to filter local finetuned checkpoints.",
    )
    local_group.add_argument(
        "--loss-signature",
        type=str,
        default=None,
        help="Optional exact loss signature filter for local finetuned checkpoint selection.",
    )
    local_group.add_argument(
        "--ckpt-timestamp",
        type=str,
        default=None,
        help=(
            "Checkpoint timestamp suffix (YYYYmmdd_HHMMSS) for local finetuned checkpoint selection. "
            "If omitted, latest matching checkpoint is used."
        ),
    )
    local_group.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional run_tag filter for local finetuned checkpoint selection.",
    )

    # --------------------------
    # Mode 2: hf (second priority)
    # --------------------------
    hf_group = parser.add_argument_group("HF Mode Args")
    hf_group.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help="Optional override for HuggingFace model id when model-source=hf.",
    )

    # --------------------------
    # Mode 3: manual --model-ref (third priority)
    # --------------------------
    manual_group = parser.add_argument_group("Manual Mode Args")
    manual_group.add_argument(
        "--model-ref",
        type=str,
        default=None,
        help=(
            "Manual model reference path/id (used only when --model-source=manual). "
            "Can be a local checkpoint path or a HuggingFace model id."
        ),
    )

    # Mode selector
    parser.add_argument(
        "--model-source",
        type=str,
        choices=["local", "hf", "manual"],
        default=None,
        help=(
            "Model loading mode. "
            "'local' selects local finetuned checkpoint by filters; "
            "'hf' loads from HuggingFace model id; "
            "'manual' loads explicit --model-ref. "
            "If omitted, falls back to infer.model_source in config."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    interval = resolve_interval(cfg, args.interval)
    cfg.setdefault("data", {})["interval"] = interval

    # Resolve model loading mode by configured priority.
    model_source, model_ref = _resolve_model_loading(args=args, cfg=cfg)

    if args.hf_model_id:
        cfg["infer"]["hf_model_id"] = args.hf_model_id

    # Resolve inference target parquet.
    infer_symbol = args.infer_symbol or args.weight_symbol
    processed_path = _resolve_processed_path(args=args, cfg=cfg, infer_symbol=infer_symbol, interval=interval)

    out_path = generate_decision_aligned_predictions(
        cfg=cfg,
        processed_path=processed_path,
        model_source=model_source,
        model_ref=model_ref,
        weight_symbol=args.weight_symbol,
        infer_symbol=infer_symbol,
        init_mode=args.init_mode,
        loss_mode=args.loss_mode,
        loss_signature=args.loss_signature,
        ckpt_timestamp=args.ckpt_timestamp,
        run_tag=args.run_tag,
        output_tag=args.output_tag,
    )
    print(f"[infer] predictions: {out_path}")


if __name__ == "__main__":
    main()
