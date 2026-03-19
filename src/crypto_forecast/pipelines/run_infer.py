from __future__ import annotations

import argparse
from pathlib import Path

from crypto_forecast.config import load_config
from crypto_forecast.models.predict import generate_decision_aligned_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run decision-aligned inference for crypto log-return forecasting.\n\n"
            "Model selection priority:\n"
            "1) --model-ref (highest priority)\n"
            "2) --model-source\n"
            "3) infer.model_source in config (default fallback)"
        )
    )
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
            "If omitted, defaults to <processed_dir>/combined_1d_logreturn.parquet."
        ),
    )

    parser.add_argument(
        "--model-ref",
        type=str,
        default=None,
        help=(
            "Explicit model reference (highest priority). "
            "Can be an absolute/relative local checkpoint path OR a HuggingFace model id. "
            "When provided, model-source is only used for output naming, not model resolution."
        ),
    )

    parser.add_argument(
        "--model-source",
        type=str,
        choices=["local", "hf"],
        default=None,
        help=(
            "Model source type when --model-ref is not provided. "
            "'local' loads fine-tuned checkpoint under outputs/checkpoints; "
            "'hf' loads from HuggingFace model id. "
            "If omitted, falls back to infer.model_source in config."
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_path = (
        Path(args.processed)
        if args.processed
        else Path(cfg["paths"]["processed_dir"]) / "combined_1d_logreturn.parquet"
    )

    # Resolution order:
    # - --model-ref: explicit local path / HF model id (highest priority)
    # - --model-source: local/hf from CLI
    # - infer.model_source from config (fallback)
    model_source = args.model_source or cfg["infer"].get("model_source", "local")

    out_path = generate_decision_aligned_predictions(
        cfg=cfg,
        processed_path=processed_path,
        model_source=model_source,
        model_ref=args.model_ref,
        output_tag=args.output_tag,
    )
    print(f"[infer] predictions: {out_path}")


if __name__ == "__main__":
    main()
