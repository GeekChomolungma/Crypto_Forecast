# Crypto_Forecast

Chronos-2 workflow for crypto daily log-return forecasting.

## What this scaffold provides

- CSV -> logreturn conversion for Binance market data
- Full-shot Chronos-2 finetuning entrypoint
- `init_mode` switch (`pretrained` / `random`)
- `loss_mode` switch plumbing (`native` / `custom` placeholder)
- W&B tracking support in `Chronos2Pipeline.fit()`
- Decision-time aligned prediction export schema for downstream testbed

## Quick start

1. Prepare environment

```bash
cd /gpfs/work5/0/prjs1859/Crypto_Forecast
pip install -e .
```

If you are modifying Chronos-2 internals (for example, custom loss), install your local Chronos repository in editable mode first:

```bash
pip install -e /gpfs/work5/0/prjs1859/chronos-forecasting
```

2. Convert raw CSV to processed log-return dataset

```bash
python -m crypto_forecast.pipelines.run_convert --config configs/experiment.yaml
```

3. Finetune (full-shot)

```bash
python -m crypto_forecast.pipelines.run_finetune --config configs/experiment.yaml
```

4. Inference and export decision-aligned CSV

```bash
python -m crypto_forecast.pipelines.run_infer --config configs/experiment.yaml
```

## Output contracts

- Checkpoints: `outputs/checkpoints/<run_name>/finetuned-ckpt`
- Predictions: `outputs/predictions/<run_name>/predictions_decision_aligned.csv`
- Run metadata: `outputs/checkpoints/<run_name>/run_manifest.json`

## Project layout

```text
Crypto_Forecast/
├─ configs/                 # Experiment configuration (data/model/train/infer)
├─ src/crypto_forecast/     # Core pipeline code
├─ data/
│  └─ processed/            # Generated parquet files (ignored by git)
├─ outputs/
│  ├─ checkpoints/          # Fine-tuned model checkpoints
│  └─ predictions/          # Decision-aligned forecast CSVs
└─ wandb/                   # Local W&B runtime files (ignored by git)
```

## Data notes

- Raw data source is configured by `paths.raw_dir` in `configs/experiment.yaml`.
- Current default expects Binance daily CSV files matching `*_1d_Binance.csv`.
- Conversion creates:
  - Per-symbol parquet: `data/processed/<SYMBOL>_1d_logreturn.parquet`
  - Combined parquet: `data/processed/combined_1d_logreturn.parquet`
- `target_logreturn` is defined as `log(close_t / close_{t-1})`.

## Prediction file schema

`predictions_decision_aligned.csv` is decision-time aligned:

- `ts_decision`: decision timestamp `t`
- `*_t` columns: observable features at `t` (e.g., `close_t`, `volume_t`)
- `pred_ret_t+1_mean`: point forecast for next step
- `pred_ret_t+1_q<q>`: quantile forecast for next step
- `y_true_ret_t+1`: realized log-return for evaluation

When `horizon > 1`, columns for `t+2`, `t+3`, ... are also included.

## Experiment tracking and records

- W&B behavior is controlled by:
  - `wandb.enabled`
  - `wandb.project`
  - `wandb.entity`
  - `wandb.tags`
- Each run uses a unique W&B run name with millisecond timestamp suffix.
- Local reproducibility metadata is saved to:
  - `outputs/checkpoints/<run_name>/run_manifest.json`
  - This file records model init mode, loss mode flag, checkpoint path, and W&B fields.

## Suggested run bookkeeping

For each experiment, keep a short note with:

- config snapshot (`configs/experiment.yaml`)
- commit hash of both `Crypto_Forecast` and `chronos-forecasting`
- W&B run URL
- output prediction file path
- key metrics produced by your external testbed
