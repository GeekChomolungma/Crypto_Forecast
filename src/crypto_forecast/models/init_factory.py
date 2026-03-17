from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoConfig

from chronos import Chronos2Model, Chronos2Pipeline


InitMode = Literal["pretrained", "random"]


def build_pipeline(
    init_mode: InitMode,
    model_id: str,
    device_map: str = "cuda",
    torch_dtype: str = "bfloat16",
) -> Chronos2Pipeline:
    if init_mode == "pretrained":
        return Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    if init_mode == "random":
        config = AutoConfig.from_pretrained(model_id)
        model = Chronos2Model(config)

        if device_map == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        if torch_dtype in ["bfloat16", "float16", "float32"]:
            model = model.to(getattr(torch, torch_dtype))

        return Chronos2Pipeline(model=model)

    raise ValueError(f"Unsupported init_mode={init_mode}")
