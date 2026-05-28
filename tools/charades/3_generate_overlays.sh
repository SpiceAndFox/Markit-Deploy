#!/usr/bin/env bash
set -euo pipefail

pip install open-clip-torch==2.24.0 timm==0.9.12 \
  'git+https://github.com/ultralytics/CLIP.git@81ff68ed7ffcac3b40484c914f104f816757308d'
pip install --no-deps 'git+https://github.com/apple/ml-mobileclip.git@aecfb5453d022e9deff12f81a150ea8f35194baa'

python tools/generate_markit_overlays.py \
  --test_path /data/MarkIt/Charades/test.json \
  --raw_video_root /data/MarkIt/Raw/Charades-Videos \
  --overlay_root /data/MarkIt/Charades/Videos \
  --yoloe_weights "${YOLOE_WEIGHTS_PATH:-/models/local/YOLOE-Large/yoloe-v8l-seg.pt}" \
  --mobileclip_weights "${MOBILECLIP_WEIGHTS_PATH:-/models/local/MobileCLIP/mobileclip_blt.pt}" \
  --device cuda:0 \
  --batch_size 64 \
  --conf 0.05 \
  --imgsz 1280
