from __future__ import annotations


SUPPORTED_INTERVALS = {"1d", "4h", "1h", "15m"}

INTERVAL_TO_FREQ = {
    "1d": "1D",
    "4h": "4h",
    "1h": "1h",
    "15m": "15min",
}


def normalize_interval(interval: str) -> str:
    value = str(interval).lower()
    if value not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval={value!r}. Expected one of: {sorted(SUPPORTED_INTERVALS)}")
    return value


def resolve_interval(cfg: dict, override: str | None = None) -> str:
    raw = override if override is not None else cfg.get("data", {}).get("interval")
    if raw is None:
        raise ValueError("Missing required interval. Set data.interval or pass --interval.")
    return normalize_interval(str(raw))


def interval_to_freq(interval: str) -> str:
    return INTERVAL_TO_FREQ[normalize_interval(interval)]
