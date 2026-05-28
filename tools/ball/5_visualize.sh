#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python3 "$PROJECT_ROOT/tools/visualize_predictions.py" \
  --predictions "$PROJECT_ROOT/outputs/predictions/ball_markit_qwen25vl.json" \
  --raw_video_root /data/MarkIt/Raw/Ball-Videos \
  --output_dir "$PROJECT_ROOT/outputs/visualizations/ball" \
  --max_samples -1 \
  --font_scale 0.55 \
  --border_thickness 6 \
  --timeline_height 50 \
  "$@"
