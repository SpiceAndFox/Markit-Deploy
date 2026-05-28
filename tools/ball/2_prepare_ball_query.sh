#!/usr/bin/env bash
set -euo pipefail

python tools/prepare_charades_sta.py \
  --input_txt /data/MarkIt/Raw/Ball-Query/raw/ball_query_test.txt \
  --video_root /data/MarkIt/Raw/Ball-Videos \
  --video_ext .mov \
  --output_json /data/MarkIt/Ball/test.json \
  --summary_json /data/MarkIt/Ball/test.summary.json \
  --nouns_json /data/MarkIt/Ball/nouns_qwen.json \
  --strict_videos
