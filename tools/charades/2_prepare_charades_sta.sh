#!/usr/bin/env bash
set -euo pipefail

python tools/prepare_charades_sta.py \
  --input_txt /data/MarkIt/Raw/Charades-STA/raw/charades_sta_test.txt \
  --video_root "${RAW_VIDEO_ROOT:-/data/MarkIt/Raw/Charades-Videos}" \
  --output_json "${TEST_PATH:-/data/MarkIt/test.json}" \
  --summary_json /data/MarkIt/test.summary.json \
  --nouns_json /data/MarkIt/nouns_qwen.json \
  --strict_videos
