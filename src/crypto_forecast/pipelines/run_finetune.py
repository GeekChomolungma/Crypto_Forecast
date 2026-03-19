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
    parser.add_argument("--loss-mode", type=str, choices=["native", "custom"], default=None)
    args = parser.parse_args()

    cfg = deepcopy(load_config(args.config))
    if args.run_name:
        cfg["project"]["run_name"] = args.run_name
    if args.init_mode:
        cfg["model"]["init_mode"] = args.init_mode
    if args.loss_mode:
        cfg["model"]["loss_mode"] = args.loss_mode

    processed_path = (
        Path(args.processed)
        if args.processed
        else Path(cfg["paths"]["processed_dir"]) / "BTCUSDT_1d_logreturn.parquet"
    )

    finetuned_path = finetune_from_processed(cfg=cfg, processed_path=processed_path)
    print(f"[finetune] saved: {finetuned_path}")


if __name__ == "__main__":
    main()
