#!/usr/bin/env bash
set -euo pipefail

python eval/vlm_mr_markit.py \
  --backend qwen2_5_vl \
  --model_path "${QWEN25_VL_MODEL_PATH:-/models/local/Qwen2.5-VL-7B-Instruct}" \
  --overlay_root "${OVERLAY_ROOT:-/data/markit_overlay_videos}" \
  --raw_video_root "${RAW_VIDEO_ROOT:-/data/charades/videos}" \
  --test_path "${TEST_PATH:-/data/charades/test.json}" \
  --save_path "${SAVE_PATH:-outputs/predictions/charades_markit_qwen25vl.json}" \
  --max_samples "${MAX_SAMPLES:--1}" \
  --num_sampled_frames "${NUM_SAMPLED_FRAMES:-32}"
