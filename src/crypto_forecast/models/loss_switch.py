from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


LossMode = Literal[
    "native",
    "weighted_extreme_time_decay",
    "magnitude_weighted",
    "directional_hybrid",
]

LOSS_PARAM_KEYS = (
    "loss_quantile_extreme_gamma",
    "loss_quantile_extreme_power",
    "loss_time_decay",
    "loss_magnitude_alpha",
    "loss_directional_lambda",
    "loss_directional_temperature",
)

DEFAULT_LOSS_PARAMS: dict[str, float] = {
    "loss_quantile_extreme_gamma": 2.0,
    "loss_quantile_extreme_power": 2.0,
    "loss_time_decay": 0.8,
    "loss_magnitude_alpha": 1.0,
    "loss_directional_lambda": 0.2,
    "loss_directional_temperature": 0.1,
}


@dataclass
class LossSwitchResult:
    requested_mode: str
    loss_type: str
    loss_params: dict[str, float]
    signature: str
    note: str


def _canonical_loss_type(loss_mode: str) -> str:
    supported = {
        "native",
        "weighted_extreme_time_decay",
        "magnitude_weighted",
        "directional_hybrid",
    }
    if loss_mode not in supported:
        raise ValueError(f"Unsupported loss_mode/loss_type={loss_mode}")
    return loss_mode


def _fmt_float(value: float) -> str:
    # Stable, path-friendly float formatting for checkpoint signatures.
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace(".", "p")


def build_loss_signature(loss_type: str, loss_params: dict[str, float]) -> str:
    if loss_type == "native":  # Keep explicit all-zero signature for native mode to simplify routing logic.
        # Keep explicit all-zero signature for native mode to simplify routing logic.
        return "qg0_qp0_td0_ma0_dl0_dt0"
    return "_".join(
        [
            f"qg{_fmt_float(loss_params['loss_quantile_extreme_gamma'])}",
            f"qp{_fmt_float(loss_params['loss_quantile_extreme_power'])}",
            f"td{_fmt_float(loss_params['loss_time_decay'])}",
            f"ma{_fmt_float(loss_params['loss_magnitude_alpha'])}",
            f"dl{_fmt_float(loss_params['loss_directional_lambda'])}",
            f"dt{_fmt_float(loss_params['loss_directional_temperature'])}",
        ]
    )


def _extract_loss_params_from_cfg(model_cfg: dict[str, Any]) -> dict[str, float]:
    nested = model_cfg.get("loss_params", {})
    params: dict[str, float] = {}
    for key in LOSS_PARAM_KEYS:
        raw = nested.get(key, model_cfg.get(key, DEFAULT_LOSS_PARAMS[key]))
        params[key] = float(raw)
    return params


def apply_loss_mode(loss_mode: str, model_cfg: dict[str, Any] | None = None) -> LossSwitchResult:
    canonical_loss_type = _canonical_loss_type(loss_mode)
    cfg = model_cfg or {}
    loss_params = _extract_loss_params_from_cfg(cfg)
    signature = build_loss_signature(canonical_loss_type, loss_params)
    return LossSwitchResult(
        requested_mode=loss_mode,
        loss_type=canonical_loss_type,
        loss_params=loss_params,
        signature=signature,
        note=f"Using Chronos-2 loss_type={canonical_loss_type}",
    )
