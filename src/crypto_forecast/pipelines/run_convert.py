from __future__ import annotations

import argparse

from crypto_forecast.config import load_config
from crypto_forecast.data.convert_market_csv import convert_raw_to_processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    combined_path, per_symbol = convert_raw_to_processed(
        raw_dir=cfg["paths"]["raw_dir"],
        processed_dir=cfg["paths"]["processed_dir"],
        file_pattern=cfg["universe"]["file_pattern"],
        symbols=cfg["universe"].get("symbols", []),
        timestamp_col=cfg["data"]["timestamp_col"],
        symbol_col=cfg["data"]["symbol_col"],
        target_price_col=cfg["data"]["target_price_col"],
        keep_feature_cols=cfg["data"]["keep_feature_cols"],
    )

    print(f"[convert] combined: {combined_path}")
    print(f"[convert] per_symbol: {len(per_symbol)} files")


if __name__ == "__main__":
    main()
