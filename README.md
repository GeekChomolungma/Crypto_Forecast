# Crypto_Forecast

Chronos-2 finetuning and inference pipeline for crypto log-return forecasting.

## Data Preprocessing

Raw Binance kline data must be converted to processed parquet files before running this pipeline.
That step is handled by a separate library:

**[kline-preprocess](https://github.com/GeekChomolungma/kline-preprocess)**

Place the output parquet files under `data/processed/` following the naming convention:

```
<SYMBOL>_<INTERVAL>_Binance_with_indicators.parquet
```

Each parquet must contain at minimum the columns listed in `data.keep_feature_cols` in `configs/experiment.yaml`, plus `symbol`, `datetime` (datetime64, UTC), and `target_logreturn`.

## Quick start

```bash
pip install -e .
# If modifying Chronos-2 internals, also:
pip install -e /projects/prjs1859/chronos-forecasting
```

### Finetune (batch, Slurm)

`sbatch_finetune_batch.sbatch` loops over a comma-separated symbol list and calls `finetune_single.sh` for each.

```bash
sbatch scripts/finetuning/sbatch_finetune_batch.sbatch \
  configs/experiment.yaml \   # $1 config
  BTCUSDT,ETHUSDT,LTCUSDT \   # $2 symbols (csv)
  4h \                         # $3 interval
  pretrained \                 # $4 init_mode
  native \                     # $5 loss_mode
  expA \                       # $6 run_tag
  2.0 2.0 0.8 1.0 0.2 0.1      # $7-$12 loss params (qg qp td ma dl dt)
```

Each symbol produces a checkpoint at:
`outputs/checkpoints/<base>__<SYMBOL>__<INTERVAL>__<INIT_MODE>__<LOSS_MODE>__lsig_<SIG>__tag_<TAG>/finetuned-ckpt`

### Inference (batch, Slurm)

`sbatch_infer_batch.sbatch` iterates over paired symbol lists (`weight[i] → infer[i]`) and calls `infer_single.sh` for each pair.

```bash
sbatch scripts/infer/sbatch_infer_batch.sbatch \
  configs/experiment.yaml \             # $1  config
  local \                               # $2  model_source (local|hf)
  BTCUSDT,ETHUSDT,LTCUSDT \             # $3  weight_symbols (csv)
  BTCUSDT,ETHUSDT,LTCUSDT \             # $4  infer_symbols  (csv, 1-to-1)
  4h \                                  # $5  interval
  pretrained \                          # $6  init_mode
  native \                              # $7  loss_mode
  latest \                              # $8  ckpt_timestamp (latest|YYYYmmdd_HHMMSS)
  expA \                                # $9  run_tag
  amazon/chronos-2 \                    # $10 hf_model_id
  inferA \                              # $11 output_tag_base
  qg0_qp0_td0_ma0_dl0_dt0              # $12 loss_signature filter
```

Predictions land in:
`outputs/predictions/infer__ft__int_<INTERVAL>__w_<WEIGHT>__...__target_<INFER>__<TS>/`

## Loss modes

| `loss_mode` | loss params used |
|---|---|
| `native` | none (signature forced to `qg0_qp0_td0_ma0_dl0_dt0`) |
| `weighted_extreme_time_decay` | qg, qp, td |
| `magnitude_weighted` | qg, qp, ma |
| `directional_hybrid` | qg, qp, dl, dt |

The 6 loss params passed to scripts in order: `qg` `qp` `td` `ma` `dl` `dt`
(`loss_quantile_extreme_gamma`, `_power`, `loss_time_decay`, `loss_magnitude_alpha`, `loss_directional_lambda`, `loss_directional_temperature`).

## Notes

- Keep `configs/experiment.yaml` and script args aligned (interval, init_mode, loss_mode, run_tag).
- For reproducibility, record both repo commit hashes (`Crypto_Forecast` + `chronos-forecasting`) together with `loss_signature`.
