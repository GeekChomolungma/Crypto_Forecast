from __future__ import annotations

import argparse
from pathlib import Path

from crypto_forecast.config import load_config
from crypto_forecast.models.train import finetune_from_processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--processed", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_path = Path(args.processed) if args.processed else Path(cfg["paths"]["processed_dir"]) / "BTCUSDT_1d_logreturn.parquet"

    finetuned_path = finetune_from_processed(cfg=cfg, processed_path=processed_path)
    print(f"[finetune] saved: {finetuned_path}")


if __name__ == "__main__":
    main()
