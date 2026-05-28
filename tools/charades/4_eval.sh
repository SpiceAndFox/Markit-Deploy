#!/usr/bin/env bash
set -euo pipefail

python eval/vlm_mr_markit.py \
  --backend qwen2_5_vl \
  --model_path /models/local/Qwen2.5-VL-7B-Instruct \
  --overlay_root /data/MarkIt/Charades/Videos \
  --raw_video_root /data/MarkIt/Raw/Charades-Videos \
  --test_path /data/MarkIt/Charades/test.json \
  --save_path outputs/predictions/charades_markit_qwen25vl.json \
  --max_samples -1 \
  --num_sampled_frames 32
