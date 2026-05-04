#!/bin/bash
# fsdp_study/multi_sweep.sh
set -e

mkdir -p fsdp_study/logs

N_REPEATS=3

for run_idx in $(seq 1 $N_REPEATS); do
  for ckpt in nockpt_config config; do
    for n in 8 4 2; do
      NAME="${n}gpu_${ckpt}"
      echo "=== Run $run_idx/$N_REPEATS: $NAME ==="
      accelerate launch \
        --config_file configs/${NAME}.yaml \
        train_fsdp.py \
        configs/scale_run_${n}gpu.yaml \
        --label "scale_run${run_idx}" \
        2>&1 | tee fsdp_study/logs/${NAME}_run${run_idx}.log
      sleep 5
    done
  done
done

# Add prefetch variant — only at 8 GPU since that's the interesting case
for run_idx in $(seq 1 $N_REPEATS); do
  echo "=== Run $run_idx/$N_REPEATS: 8gpu_nockpt_prefetch ==="
  accelerate launch \
    --config_file fsdp_study/fsdp_config_8gpu_nockpt_prefetch.yaml \
    fsdp_study/train_fsdp.py \
    configs/scale_run_8gpu.yaml \
    --label "8gpu_nockpt_prefetch_run${run_idx}" \
    2>&1 | tee fsdp_study/logs/8gpu_nockpt_prefetch_run${run_idx}.log
  sleep 5
done

echo "All runs complete."