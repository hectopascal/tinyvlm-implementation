#!/bin/bash
# fsdp_study/run_sweep.sh
set -e

mkdir -p fsdp_study/logs

for ckpt in nockpt_config config; do
  for n in 8 4 2; do  # start with 8 — most likely to fit, fail fast on smaller
    NAME="${n}gpu_${ckpt}"
    echo "=== Running $NAME ==="
    accelerate launch \
      --config_file configs/${NAME}.yaml \
      train_fsdp.py \
      configs/scale_run_${n}gpu.yaml \
      2>&1 | tee fsdp_study/logs/${NAME}.log
    echo "=== Done $NAME ==="
    sleep 5  # let GPU memory settle between runs
  done
done

echo "All runs complete. Results in fsdp_study/results.csv"