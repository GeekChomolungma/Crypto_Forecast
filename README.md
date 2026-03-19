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
cd /projects/prjs1859/Crypto_Forecast
pip install -e .
```

If you are modifying Chronos-2 internals (for example, custom loss), install your local Chronos repository in editable mode first:

```bash
pip install -e /projects/prjs1859/chronos-forecasting
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
python -m crypto_forecast.pipelines.run_infer \
  --config configs/experiment.yaml \
  --model-source local \
  --weight-symbol BTCUSDT \
  --infer-symbol BTCUSDT \
  --init-mode pretrained \
  --loss-mode native
```

Common inference variants:

```bash
# Local fine-tuned checkpoint (default mode)
python -m crypto_forecast.pipelines.run_infer \
  --config configs/experiment.yaml \
  --model-source local \
  --weight-symbol BTCUSDT \
  --infer-symbol BTCUSDT \
  --init-mode pretrained \
  --loss-mode native

# Zero-shot from HuggingFace Chronos-2
python -m crypto_forecast.pipelines.run_infer \
  --config configs/experiment.yaml \
  --model-source hf \
  --infer-symbol BTCUSDT \
  --hf-model-id amazon/chronos-2 \
  --output-tag zeroshot
```

## Output contracts

### Finetune output (checkpoints)

Finetune scripts generate run names in this format:

`<base_run_name>__<weight_symbol>__<init_mode>__<loss_mode>__<run_tag>__<timestamp>`

Example:

`chronos2_logret__BTCUSDT__pretrained__native__expA__20260319_113000`

Saved artifacts:

- Checkpoint folder: `outputs/checkpoints/<run_name>/finetuned-ckpt`
- Run manifest: `outputs/checkpoints/<run_name>/run_manifest.json`

Field meaning:

- `base_run_name`: logical experiment family name from config (`project.run_name`)
- `weight_symbol`: which symbol data was used for finetuning weights
- `init_mode`: `pretrained` (Chronos-2 init) or `random`
- `loss_mode`: `native` or `custom`
- `run_tag`: manual experiment label (free text, sanitized in scripts)
- `timestamp`: training launch timestamp (`YYYYmmdd_HHMMSS`)

### Inference output (predictions)

Inference creates a directory under `outputs/predictions/` and one CSV file inside.

For local finetuned inference:

`outputs/predictions/infer__ft__w_<weight_symbol>__i_<init_mode>__l_<loss_mode>__ckpt_<resolved_ckpt_timestamp>__target_<infer_symbol>__<infer_runtime_timestamp>[__tag_<output_tag>]`

For HF zero-shot inference:

`outputs/predictions/infer__hf__<hf_model_id_slug>__target_<infer_symbol>__<infer_runtime_timestamp>[__tag_<output_tag>]`

CSV file name:

`predictions_decision_aligned__target_<infer_symbol>__init_<init_mode>__loss_<loss_mode>__tag_<run_tag_or_output_tag>.csv`

Field meaning:

- `resolved_ckpt_timestamp`: timestamp suffix parsed from selected local checkpoint run directory
- `infer_runtime_timestamp`: timestamp when the inference job runs
- `target_<infer_symbol>`: symbol used to choose inference parquet (`data/processed/<infer_symbol>_1d_logreturn.parquet`)
- `run_tag_or_output_tag`: experiment label used for easier downstream filtering

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
├─ scripts/
│  ├─ zeroshot/             # Zero-shot infer shell + sbatch
│  ├─ finetuning/           # Finetune shell + sbatch
│  └─ infer/                # Unified infer workflows (local/hf, single/batch)
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

`predictions_decision_aligned__*.csv` is decision-time aligned:

- `ts_decision`: decision timestamp `t`
- `*_t` columns: observable features at `t` (e.g., `close_t`, `volume_t`)
- `pred_ret_t+1_mean`: point forecast for next step
- `pred_ret_t+1_q<q>`: quantile forecast for next step
- `y_true_ret_t+1`: realized log-return for evaluation

When `horizon > 1`, columns for `t+2`, `t+3`, ... are also included.

## End-to-end workflow

### 1) Training workflow (finetune)

Training is organized around one weight symbol per run:

1. Prepare processed data (`run_convert`)
2. Run finetune single or batch script
3. Each run writes:
- checkpoint: `outputs/checkpoints/<run_name>/finetuned-ckpt`
- metadata: `outputs/checkpoints/<run_name>/run_manifest.json`

Recommended training entrypoints:

- Single symbol:
`scripts/finetuning/finetune_single.sh`
- Multi symbol loop:
`scripts/finetuning/finetune_batch.sh`

### 2) Inference workflow (single)

Entry: `scripts/infer/infer_single.sh`

Model loading priority:

1. `local` mode (highest):
- select finetuned checkpoint by `weight_symbol + init_mode + loss_mode (+ run_tag)`
- if `ckpt_timestamp=latest`: auto-pick latest matching checkpoint
- if `ckpt_timestamp=<YYYYmmdd_HHMMSS>`: use exact checkpoint
2. `hf` mode:
- load Chronos-2 directly from HuggingFace via `hf_model_id`
3. `manual` mode (only via `run_infer.py`):
- explicit `--model-ref`

Prediction parquet selection:

1. if `--processed` is provided: use it directly
2. otherwise auto-build from symbol:
`data/processed/<infer_symbol>_1d_logreturn.parquet`
3. in scripts, `infer_symbol` defaults to `weight_symbol`

Typical single-infer parameter intent:

- `weight_symbol`: which finetuned weights to load (local mode)
- `infer_symbol`: which symbol parquet to predict on
- `init_mode/loss_mode`: local checkpoint filters
- `ckpt_timestamp`: latest or fixed historical checkpoint
- `run_tag`: optional extra local checkpoint filter
- `output_tag`: label for this inference output set

### 3) Inference workflow (batch)

Entry: `scripts/infer/infer_batch.sh`

Batch behavior:

1. Parse `weight_symbols_csv` and `infer_symbols_csv` (both comma-separated)
2. Enforce one-to-one mapping (same list length):
- `weight_symbols_csv[i] -> infer_symbols_csv[i]`
3. Loop each pair and call `infer_single.sh`
4. Keep global model settings fixed across the batch:
- `model_source`, `init_mode`, `loss_mode`, `ckpt_timestamp`, `run_tag`, `hf_model_id`
5. Auto-append pair into output tag:
`<output_tag_base>_<weight_symbol>_to_<infer_symbol>`

This makes cross-symbol transfer experiments explicit while still separating outputs per pair.

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

## Scripts

Scripts are organized by workflow:

- `scripts/zeroshot/`
  - `zero_shot_single.sh`: run one zero-shot inference job
  - `zero_shot_batch.sh`: batch over all per-symbol parquet files in `data/processed`
  - `sbatch_zero_shot_single.sbatch`: submit single zero-shot job via Slurm
  - `sbatch_zero_shot_batch.sbatch`: submit batch zero-shot job via Slurm
- `scripts/finetuning/`
  - `finetune_single.sh`: single-symbol finetune with checkpoint isolation by `symbol/init_mode/loss_mode/run_tag/timestamp`
  - `sbatch_finetune_single.sbatch`: submit the finetune job via Slurm
  - `finetune_batch.sh`: multi-symbol loop (comma-separated symbols) with global `init_mode/loss_mode`
  - `sbatch_finetune_batch.sbatch`: submit batch finetune loop via Slurm
- `scripts/infer/`
  - `infer_single.sh`: single infer script for both local finetuned and HF zero-shot models
  - `sbatch_infer_single.sbatch`: submit single infer job via Slurm
  - `infer_batch.sh`: batch infer loop over comma-separated target symbols
  - `sbatch_infer_batch.sbatch`: submit batch infer loop via Slurm

Examples:

```bash
bash scripts/zeroshot/zero_shot_single.sh configs/experiment.yaml /projects/prjs1859/Crypto_Forecast/data/processed/BTCUSDT_1d_logreturn.parquet
bash scripts/zeroshot/zero_shot_batch.sh configs/experiment.yaml /projects/prjs1859/Crypto_Forecast/data/processed

# Slurm submission
sbatch scripts/zeroshot/sbatch_zero_shot_single.sbatch configs/experiment.yaml /projects/prjs1859/Crypto_Forecast/data/processed/BTCUSDT_1d_logreturn.parquet
sbatch scripts/zeroshot/sbatch_zero_shot_batch.sbatch configs/experiment.yaml /projects/prjs1859/Crypto_Forecast/data/processed

# Finetuning (symbol/init/loss split, isolated checkpoint folder by run_name)
bash scripts/finetuning/finetune_single.sh configs/experiment.yaml BTCUSDT pretrained native expA
sbatch scripts/finetuning/sbatch_finetune_single.sbatch configs/experiment.yaml ETHUSDT random custom expB

# Finetuning batch (same init/loss across multiple symbols)
bash scripts/finetuning/finetune_batch.sh configs/experiment.yaml BTCUSDT,ETHUSDT pretrained native expBatch1
sbatch scripts/finetuning/sbatch_finetune_batch.sbatch configs/experiment.yaml BTCUSDT,ETHUSDT random custom expBatch2

# Inference (local finetuned: latest matching checkpoint by default)
bash scripts/infer/infer_single.sh configs/experiment.yaml local BTCUSDT LTCUSDT pretrained native latest expA amazon/chronos-2 btc2ltc

# Inference (HF zero-shot)
sbatch scripts/infer/sbatch_infer_single.sbatch configs/experiment.yaml hf BTCUSDT BTCUSDT pretrained native latest "" amazon/chronos-2 zshot_btc

# Inference batch (local finetuned, one-to-one mapping)
# weights: BTCUSDT,ETHUSDT,LTCUSDT -> targets: BTCUSDT,LTCUSDT,XRPUSDT
bash scripts/infer/infer_batch.sh configs/experiment.yaml local BTCUSDT,ETHUSDT,LTCUSDT BTCUSDT,LTCUSDT,XRPUSDT pretrained native latest expA amazon/chronos-2 ft_batch

# Inference batch (HF zero-shot, one-to-one mapping lists still required)
sbatch scripts/infer/sbatch_infer_batch.sbatch configs/experiment.yaml hf BTCUSDT,ETHUSDT BTCUSDT,ETHUSDT pretrained native latest "" amazon/chronos-2 zshot_batch
```
