#!/usr/bin/env bash
set -euo pipefail

python eval/vlm_mr_markit.py \
  --backend qwen2_5_vl \
  --model_path "${QWEN25_VL_MODEL_PATH:-/models/local/Qwen2.5-VL-7B-Instruct}" \
  --overlay_root /data/MarkIt/Ball \
  --raw_video_root /data/MarkIt/Raw/Ball-Videos \
  --test_path /data/MarkIt/Ball/test.json \
  --save_path outputs/predictions/ball_markit_qwen25vl.json \
  --max_samples -1 \
  --num_sampled_frames 32 \
  --video_ext .mov
