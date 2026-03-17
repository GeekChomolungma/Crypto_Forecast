from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_forecast.data.build_chronos_inputs import build_tasks_for_fit, split_by_time
from crypto_forecast.models.init_factory import build_pipeline
from crypto_forecast.models.loss_switch import apply_loss_mode
from crypto_forecast.utils.io import dump_json, ensure_dir


def finetune_from_processed(cfg: dict[str, Any], processed_path: Path) -> Path:
    run_name = cfg["project"]["run_name"]
    ckpt_root = ensure_dir(Path(cfg["paths"]["checkpoints_dir"]) / run_name)

    df = pd.read_parquet(processed_path)
    split_cfg = cfg["split"]
    split = split_by_time(df, train_end=split_cfg["train_end"], val_end=split_cfg["val_end"])

    data_cfg = cfg["data"]
    cov_cols = list(data_cfg["keep_feature_cols"])
    min_history = int(data_cfg["min_history"])

    train_inputs = build_tasks_for_fit(
        df=split.train_df,
        symbol_col=data_cfg["symbol_col"],
        target_col="target_logreturn",
        cov_cols=cov_cols,
        min_history=min_history,
    )

    val_inputs = None
    if len(split.val_df) > 0:
        try:
            val_inputs = build_tasks_for_fit(
                df=split.val_df,
                symbol_col=data_cfg["symbol_col"],
                target_col="target_logreturn",
                cov_cols=cov_cols,
                min_history=min_history,
            )
        except ValueError:
            val_inputs = None

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    wandb_cfg = cfg["wandb"]

    loss_state = apply_loss_mode(model_cfg["loss_mode"])

    pipeline = build_pipeline(
        init_mode=model_cfg["init_mode"],
        model_id=model_cfg["model_id"],
        device_map=model_cfg["device_map"],
        torch_dtype=model_cfg["torch_dtype"],
    )

    extra_trainer_kwargs: dict[str, Any] = {}
    if wandb_cfg.get("enabled", False):
        # HF Trainer auto-enables W&B when report_to includes "wandb".
        # Project/entity/tags are picked up from standard WANDB_* env vars.
        if wandb_cfg.get("project"):
            os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
        tags = wandb_cfg.get("tags", [])
        if tags:
            os.environ["WANDB_TAGS"] = ",".join(str(t) for t in tags)

        report_to = ["wandb"]
        wandb_run_name = f"{run_name}-{int(time.time() * 1000)}"
        extra_trainer_kwargs.update(
            {
                "report_to": report_to,
                "run_name": wandb_run_name,
                "logging_steps": 20,
            }
        )

    finetuned = pipeline.fit(
        inputs=train_inputs,
        validation_inputs=val_inputs,
        prediction_length=int(model_cfg["prediction_length"]),
        finetune_mode=train_cfg["finetune_mode"],
        learning_rate=float(train_cfg["learning_rate"]),
        num_steps=int(train_cfg["num_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        output_dir=ckpt_root,
        finetuned_ckpt_name=train_cfg["finetuned_ckpt_name"],
        remove_printer_callback=bool(train_cfg["remove_printer_callback"]),
        **extra_trainer_kwargs,
    )

    finetuned_path = ckpt_root / train_cfg["finetuned_ckpt_name"]
    manifest = {
        "run_name": run_name,
        "processed_path": str(processed_path),
        "finetuned_path": str(finetuned_path),
        "loss_mode": model_cfg["loss_mode"],
        "loss_mode_state": loss_state.__dict__,
        "init_mode": model_cfg["init_mode"],
        "model_id": model_cfg["model_id"],
        "prediction_length": model_cfg["prediction_length"],
        "wandb_enabled": bool(wandb_cfg.get("enabled", False)),
        "wandb_project": os.environ.get("WANDB_PROJECT"),
        "wandb_entity": os.environ.get("WANDB_ENTITY"),
        "wandb_tags": os.environ.get("WANDB_TAGS"),
    }
    dump_json(ckpt_root / "run_manifest.json", manifest)

    # Ensure saved model exists (fit already saves, this is a safety save).
    finetuned.save_pretrained(finetuned_path)
    return finetuned_path
