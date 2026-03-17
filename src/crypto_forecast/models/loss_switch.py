from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LossMode = Literal["native", "custom"]


@dataclass
class LossSwitchResult:
    mode: LossMode
    active: bool
    note: str


def apply_loss_mode(loss_mode: LossMode) -> LossSwitchResult:
    """
    Wiring layer for loss mode selection.

    - native: use Chronos-2 built-in quantile loss
    - custom: placeholder hook; user will later modify chronos2 source loss

    This keeps the experiment config/manifest stable now, while the custom loss
    implementation can be dropped in later.
    """
    if loss_mode == "native":
        return LossSwitchResult(mode="native", active=True, note="Using built-in Chronos-2 loss.")

    if loss_mode == "custom":
        return LossSwitchResult(
            mode="custom",
            active=False,
            note=(
                "Custom loss mode requested but not implemented in this scaffold. "
                "Please patch chronos-forecasting/src/chronos/chronos2/model.py::_compute_loss and rerun."
            ),
        )

    raise ValueError(f"Unsupported loss_mode={loss_mode}")
