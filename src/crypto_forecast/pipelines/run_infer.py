from __future__ import annotations

import argparse
from pathlib import Path

from crypto_forecast.config import load_config
from crypto_forecast.models.predict import generate_decision_aligned_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--processed", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_path = Path(args.processed) if args.processed else Path(cfg["paths"]["processed_dir"]) / "combined_1d_logreturn.parquet"

    out_path = generate_decision_aligned_predictions(cfg=cfg, processed_path=processed_path)
    print(f"[infer] predictions: {out_path}")


if __name__ == "__main__":
    main()
