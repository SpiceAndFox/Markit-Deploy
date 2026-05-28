#!/usr/bin/env bash
set -euo pipefail

python tools/extract_nouns_qwen.py \
  --input_txt /data/MarkIt/Raw/Ball-Query/raw/ball_query_test.txt \
  --model_path "${SUBJECT_LLM_MODEL_PATH:-/models/local/Qwen2.5-7B-Instruct}" \
  --output_json /data/MarkIt/Ball/nouns_qwen.json
