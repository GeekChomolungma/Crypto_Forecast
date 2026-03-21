# Crypto_Forecast

Chronos-2 workflow for crypto daily log-return forecasting.

## What this project provides

- CSV -> log-return conversion for Binance market data
- Full-shot Chronos-2 finetuning entrypoint
- `init_mode` switch (`pretrained` / `random`)
- `loss_mode` switch with 4 concrete options:
  - `native`
  - `weighted_extreme_time_decay`
  - `magnitude_weighted`
  - `directional_hybrid`
- Loss hyper-parameter routing (6 params) into `Chronos2Pipeline.fit(...)`
- Stable `loss_signature` generation for checkpoint naming and selection
- W&B tracking support in `Chronos2Pipeline.fit()`
- Decision-time aligned prediction export schema for downstream testbed

## Quick start

1. Prepare environment

```bash
cd /projects/prjs1859/Crypto_Forecast
pip install -e .
```

If you are modifying Chronos-2 internals, install your local Chronos repository in editable mode first:

```bash
pip install -e /projects/prjs1859/chronos-forecasting
```

2. Convert raw CSV to processed log-return dataset

```bash
python -m crypto_forecast.pipelines.run_convert --config configs/experiment.yaml
```

3. Finetune (single symbol)

```bash
bash scripts/finetuning/finetune_single.sh \
  configs/experiment.yaml \
  BTCUSDT \
  pretrained \
  native \
  expA \
  2.0 2.0 0.8 1.0 0.2 0.1
```

4. Inference (local checkpoint)

```bash
bash scripts/infer/infer_single.sh \
  configs/experiment.yaml \
  local \
  BTCUSDT \
  BTCUSDT \
  pretrained \
  native \
  latest \
  expA \
  amazon/chronos-2 \
  inferA \
  qg0_qp0_td0_ma0_dl0_dt0
```

5. Inference (HF zero-shot)

```bash
bash scripts/infer/infer_single.sh \
  configs/experiment.yaml \
  hf \
  BTCUSDT \
  BTCUSDT \
  pretrained \
  native \
  latest \
  "" \
  amazon/chronos-2 \
  zshot
```

## Loss modes and hyper-parameters

Supported `loss_mode` values:

- `native`
- `weighted_extreme_time_decay`
- `magnitude_weighted`
- `directional_hybrid`

Supported loss hyper-parameters (in this order in finetune scripts):

1. `loss_quantile_extreme_gamma`
2. `loss_quantile_extreme_power`
3. `loss_time_decay`
4. `loss_magnitude_alpha`
5. `loss_directional_lambda`
6. `loss_directional_temperature`

All 6 values are always accepted by scripts. For `native`, the signature is forced to all-zero.

## Loss signature design

`loss_signature` is generated from `loss_mode + 6 hyper-parameters` and used in:

- training run directory name
- `run_manifest.json`
- inference checkpoint filtering (`--loss-signature`)
- prediction output naming

Current signature format:

- `native`: `qg0_qp0_td0_ma0_dl0_dt0`
- other modes: `qg<...>_qp<...>_td<...>_ma<...>_dl<...>_dt<...>`

Example:

- `qg2_qp2_td0p8_ma1_dl0p2_dt0p1`

## Output contracts

### Finetune output (checkpoints)

Finetune scripts generate run names in this format:

`<base_run_name>__<weight_symbol>__<init_mode>__<loss_mode>__lsig_<loss_signature>__tag_<run_tag>`

Example:

`chronos2_logret__BTCUSDT__pretrained__native__lsig_qg0_qp0_td0_ma0_dl0_dt0__tag_expA`

Saved artifacts:

- Checkpoint folder: `outputs/checkpoints/<run_name>/finetuned-ckpt`
- Run manifest: `outputs/checkpoints/<run_name>/run_manifest.json`

Manifest includes at least:

- `loss_mode`
- `loss_type`
- `loss_signature`
- `loss_params`

### Inference output (predictions)

Inference creates a directory under `outputs/predictions/` and one CSV file inside.

For local finetuned inference:

`outputs/predictions/infer__ft__w_<weight_symbol>__i_<init_mode>__l_<loss_mode>__lsig_<loss_signature_or_any>__ckpt_<resolved_ckpt_timestamp>__target_<infer_symbol>__<infer_runtime_timestamp>[__tag_<output_tag>]`

For HF zero-shot inference:

`outputs/predictions/infer__hf__<hf_model_id_slug>__target_<infer_symbol>__<infer_runtime_timestamp>[__tag_<output_tag>]`

CSV filename:

`predictions_decision_aligned__target_<infer_symbol>__init_<init_mode>__loss_<loss_mode>__lsig_<loss_signature_or_any>__tag_<run_tag_or_output_tag>.csv`

## End-to-end workflow

### 1) Training workflow (finetune)

Training is organized around one weight symbol per run:

1. Prepare processed data (`run_convert`)
2. Run finetune single or batch script
3. Each run writes:
- checkpoint: `outputs/checkpoints/<run_name>/finetuned-ckpt`
- metadata: `outputs/checkpoints/<run_name>/run_manifest.json`

Recommended training entrypoints:

- Single symbol: `scripts/finetuning/finetune_single.sh`
- Multi symbol loop: `scripts/finetuning/finetune_batch.sh`

### 2) Inference workflow (single)

Entry: `scripts/infer/infer_single.sh`

Local mode checkpoint selection filters:

- `weight_symbol`
- `init_mode`
- `loss_mode`
- optional `loss_signature`
- optional `run_tag`
- optional `ckpt_timestamp`

If `ckpt_timestamp=latest`, latest matched checkpoint is selected.

### 3) Inference workflow (batch)

Entry: `scripts/infer/infer_batch.sh`

Batch behavior:

1. Parse `weight_symbols_csv` and `infer_symbols_csv` (comma-separated)
2. Enforce one-to-one mapping (`weight_symbols_csv[i] -> infer_symbols_csv[i]`)
3. Loop each pair and call `infer_single.sh`
4. Keep global settings fixed across batch:
- `model_source`, `init_mode`, `loss_mode`, `loss_signature`, `ckpt_timestamp`, `run_tag`, `hf_model_id`

## Scripts and examples

```bash
# Finetuning (single)
bash scripts/finetuning/finetune_single.sh \
  configs/experiment.yaml BTCUSDT pretrained native expA 2.0 2.0 0.8 1.0 0.2 0.1

# Finetuning (batch)
bash scripts/finetuning/finetune_batch.sh \
  configs/experiment.yaml BTCUSDT,ETHUSDT pretrained directional_hybrid expBatch 2.0 2.0 0.8 1.0 0.2 0.1

# Inference (local finetuned, exact signature)
bash scripts/infer/infer_single.sh \
  configs/experiment.yaml local BTCUSDT LTCUSDT pretrained directional_hybrid latest expBatch amazon/chronos-2 transfer_ltc qg2_qp2_td0p8_ma1_dl0p2_dt0p1

# Inference (local finetuned, no signature filter)
bash scripts/infer/infer_single.sh \
  configs/experiment.yaml local BTCUSDT BTCUSDT pretrained native latest expA amazon/chronos-2 infer_native

# Inference (HF zero-shot)
bash scripts/infer/infer_single.sh \
  configs/experiment.yaml hf BTCUSDT BTCUSDT pretrained native latest "" amazon/chronos-2 zshot
```

## Notes

- Keep `configs/experiment.yaml` and your script args aligned.
- For reproducibility, record both repository commit hashes (`Crypto_Forecast` and `chronos-forecasting`) together with selected `loss_signature`.
