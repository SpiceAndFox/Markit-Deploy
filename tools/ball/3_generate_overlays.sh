#!/usr/bin/env bash
set -euo pipefail

python tools/generate_markit_overlays.py \
  --test_path /data/MarkIt/Ball/test.json \
  --raw_video_root /data/MarkIt/Raw/Ball-Videos \
  --overlay_root /data/MarkIt/Ball/Videos \
  --yoloe_weights "${YOLOE_WEIGHTS_PATH:-/models/local/YOLOE-Large/yoloe-v8l-seg.pt}" \
  --mobileclip_weights "${MOBILECLIP_WEIGHTS_PATH:-/models/local/MobileCLIP/mobileclip_blt.pt}" \
  --device cuda:0 \
  --batch_size 32 \
  --video_ext .mov \
  --conf 0.05 \
  --imgsz 1280
