from __future__ import annotations

import argparse

from crypto_forecast.config import load_config
from crypto_forecast.data.convert_market_csv import convert_raw_to_processed
from crypto_forecast.data.intervals import resolve_interval


def _resolve_interval(cfg: dict) -> str:
    return resolve_interval(cfg)


def _resolve_file_pattern(cfg: dict, interval: str) -> str:
    universe_cfg = cfg.get("universe", {})

    template = universe_cfg.get("file_pattern_template")
    if template:
        return str(template).format(interval=interval)

    file_pattern = str(universe_cfg.get("file_pattern") or "*_{interval}_Binance_cleaned.csv")
    if "{interval}" in file_pattern:
        return file_pattern.format(interval=interval)

    return file_pattern


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    interval = _resolve_interval(cfg=cfg)
    file_pattern = _resolve_file_pattern(cfg=cfg, interval=interval)

    per_symbol = convert_raw_to_processed(
        raw_dir=cfg["paths"]["raw_dir"],
        processed_dir=cfg["paths"]["processed_dir"],
        file_pattern=file_pattern,
        interval=interval,
        symbols=cfg["universe"].get("symbols", []),
        timestamp_col=cfg["data"]["timestamp_col"],
        symbol_col=cfg["data"]["symbol_col"],
        target_price_col=cfg["data"]["target_price_col"],
        keep_feature_cols=cfg["data"]["keep_feature_cols"],
    )

    print(f"[convert] interval: {interval}")
    print(f"[convert] file_pattern: {file_pattern}")
    print(f"[convert] per_symbol: {len(per_symbol)} files")
    for path in per_symbol:
        print(f"[convert] wrote: {path}")


if __name__ == "__main__":
    main()
