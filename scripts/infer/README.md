# Infer Scripts

Primary Slurm entrypoint:

```bash
sbatch scripts/infer/sbatch_infer_batch.sbatch \
  configs/experiment.yaml \
  local \
  BTCUSDT,BTCUSDT \
  ETHUSDT,LTCUSDT \
  4h \
  pretrained \
  weighted_extreme_time_decay \
  latest \
  batch \
  amazon/chronos-2 \
  transfer \
  qg2_qp2_td0p8_ma1_dl0p2_dt0p1
```

Direct single-pair entrypoint:

```bash
bash scripts/infer/infer_single.sh \
  configs/experiment.yaml local BTCUSDT ETHUSDT 4h pretrained native latest batch amazon/chronos-2 manual qg0_qp0_td0_ma0_dl0_dt0
```
