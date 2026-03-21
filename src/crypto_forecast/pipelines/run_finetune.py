from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from crypto_forecast.config import load_config
from crypto_forecast.models.train import finetune_from_processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--processed", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--init-mode", type=str, choices=["pretrained", "random"], default=None)
    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=["native", "weighted_extreme_time_decay", "magnitude_weighted", "directional_hybrid"],
        default=None,
    )
    parser.add_argument("--loss-quantile-extreme-gamma", type=float, default=None)
    parser.add_argument("--loss-quantile-extreme-power", type=float, default=None)
    parser.add_argument("--loss-time-decay", type=float, default=None)
    parser.add_argument("--loss-magnitude-alpha", type=float, default=None)
    parser.add_argument("--loss-directional-lambda", type=float, default=None)
    parser.add_argument("--loss-directional-temperature", type=float, default=None)
    args = parser.parse_args()

    cfg = deepcopy(load_config(args.config))
    if args.run_name:
        cfg["project"]["run_name"] = args.run_name
    if args.init_mode:
        cfg["model"]["init_mode"] = args.init_mode
    if args.loss_mode:
        cfg["model"]["loss_mode"] = args.loss_mode
    cfg["model"].setdefault("loss_params", {})
    if args.loss_quantile_extreme_gamma is not None:
        cfg["model"]["loss_params"]["loss_quantile_extreme_gamma"] = args.loss_quantile_extreme_gamma
    if args.loss_quantile_extreme_power is not None:
        cfg["model"]["loss_params"]["loss_quantile_extreme_power"] = args.loss_quantile_extreme_power
    if args.loss_time_decay is not None:
        cfg["model"]["loss_params"]["loss_time_decay"] = args.loss_time_decay
    if args.loss_magnitude_alpha is not None:
        cfg["model"]["loss_params"]["loss_magnitude_alpha"] = args.loss_magnitude_alpha
    if args.loss_directional_lambda is not None:
        cfg["model"]["loss_params"]["loss_directional_lambda"] = args.loss_directional_lambda
    if args.loss_directional_temperature is not None:
        cfg["model"]["loss_params"]["loss_directional_temperature"] = args.loss_directional_temperature

    processed_path = (
        Path(args.processed)
        if args.processed
        else Path(cfg["paths"]["processed_dir"]) / "BTCUSDT_1d_logreturn.parquet"
    )

    finetuned_path = finetune_from_processed(cfg=cfg, processed_path=processed_path)
    print(f"[finetune] saved: {finetuned_path}")


if __name__ == "__main__":
    main()
