#!/usr/bin/env bash
set -euo pipefail

python3 "tools/visualize_predictions.py" \
  --predictions outputs/predictions/charades_markit_qwen25vl.json \
  --raw_video_root /data/MarkIt/Raw/Charades-Videos \
  --output_dir "outputs/visualizations/charades" \
  --max_samples -1 \
  --font_scale 0.55 \
  --border_thickness 6 \
  --timeline_height 50 \
  "$@"
