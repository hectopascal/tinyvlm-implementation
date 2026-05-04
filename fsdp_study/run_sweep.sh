#!/bin/bash
set -e

# Strong scaling: effective batch = 32 across all configs
# 2 GPU: per_gpu=4, accum=4 -> 4*4*2 = 32
# 4 GPU: per_gpu=4, accum=2 -> 4*2*4 = 32
# 8 GPU: per_gpu=4, accum=1 -> 4*1*8 = 32

# You'll need 6 config files total (3 GPU counts x 2 ckpt settings)
# OR override via env vars — accelerate doesn't make this super clean.
# Easiest: maintain 6 yaml files, run them in sequence.

for ckpt in "true" "false"; do
  for n in 2 4 8; do
    echo "=== Running $n GPUs, activation_ckpt=$ckpt ==="
    # Edit the yaml in place or maintain separate copies
    accelerate launch \
      --config_file fsdp_study/fsdp_config_${n}gpu_ckpt${ckpt}.yaml \
      fsdp_study/train_fsdp.py \
      configs/scale_7b_${n}gpu.yaml
  done
done