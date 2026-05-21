#!/usr/bin/env python3
# Usage:
#   python tools/generate_markit_overlays.py \
#     --test_path /data/MarkIt/test.json \
#     --raw_video_root /data/Raw/Charades-Video \
#     --overlay_root /data/MarkIt/Charades \
#     --yoloe_weights /models/local/YOLOE-Large/yoloe-v8l-seg.pt \
#     --device cuda:0
#
# Docker Compose smoke test:
#   docker compose run --rm markit bash -lc '
#   python tools/generate_markit_overlays.py \
#     --test_path "$TEST_PATH" \
#     --raw_video_root "$RAW_VIDEO_ROOT" \
#     --overlay_root "$OVERLAY_ROOT" \
#     --yoloe_weights "$YOLOE_WEIGHTS_PATH" \
#     --device cuda:0 \
#     --max_records 5 \
#     --summary_json "$OVERLAY_ROOT/overlay_smoke.summary.json" \
#     --overwrite'
#
# Output:
#   Writes <overlay_root>/<id>_overlay.mp4 (i.e. /data/MarkIt/Charades/<id>_overlay.mp4),
#   which is consumed by eval/vlm_mr_markit.py. This script does not write .npz mask caches.
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable


COLOR_MAP_BGR = {
    1: (0, 0, 255),       # red
    2: (0, 255, 255),     # yellow
    3: (255, 0, 0),       # blue
    4: (0, 255, 0),       # green
    5: (255, 255, 0),     # cyan
    6: (255, 0, 255),     # magenta
    7: (0, 165, 255),     # orange
    8: (203, 192, 255),   # pink
    9: (255, 255, 255),   # white
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def ensure_video_name(video_name: str, suffix: str = ".mp4") -> str:
    video_name = str(video_name)
    return video_name if video_name.endswith(suffix) else f"{video_name}{suffix}"


def coerce_float(value: Any) -> float:
    return float(value)


def flatten_grouped_video_dataset(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for video_id, sample in payload.items():
        if not isinstance(sample, dict):
            raise ValueError(f"{video_id}: expected object payload")

        sentences = sample.get("sentences")
        timestamps = sample.get("timestamps")
        duration = sample.get("video_duration", sample.get("duration"))
        if not isinstance(sentences, list) or not isinstance(timestamps, list):
            raise ValueError(f"{video_id}: expected list fields 'sentences' and 'timestamps'")
        if len(sentences) != len(timestamps):
            raise ValueError(f"{video_id}: sentences and timestamps lengths differ")
        if duration is None:
            raise ValueError(f"{video_id}: missing duration/video_duration")

        nouns_payload = sample.get("nouns")
        for idx, (query, timestamp) in enumerate(zip(sentences, timestamps), start=1):
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                raise ValueError(f"{video_id}: invalid timestamp at index {idx}: {timestamp}")
            if isinstance(nouns_payload, list) and len(nouns_payload) >= idx:
                nouns = nouns_payload[idx - 1]
            elif isinstance(nouns_payload, dict):
                nouns = nouns_payload
            else:
                nouns = {}
            records.append(
                {
                    "id": f"{Path(str(video_id)).stem}_{idx}",
                    "video": ensure_video_name(str(video_id)),
                    "query": normalize_text(query),
                    "start_time": coerce_float(timestamp[0]),
                    "end_time": coerce_float(timestamp[1]),
                    "duration": coerce_float(duration),
                    "nouns": nouns if isinstance(nouns, dict) else {},
                }
            )
    return records


def flatten_list_video_dataset(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(payload, start=1):
        query = sample.get("query", sample.get("sentence"))
        if query is None:
            raise ValueError(f"List sample {idx}: missing query/sentence")

        if "start_time" in sample and "end_time" in sample:
            start_time, end_time = sample["start_time"], sample["end_time"]
        else:
            timestamp = sample.get("timestamp", sample.get("timestamps"))
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                raise ValueError(f"List sample {idx}: missing valid timestamp")
            start_time, end_time = timestamp

        duration = sample.get("duration", sample.get("video_duration"))
        if duration is None:
            raise ValueError(f"List sample {idx}: missing duration/video_duration")

        video_name = sample.get("video", sample.get("video_id", sample.get("clip_id")))
        if video_name is None:
            raise ValueError(f"List sample {idx}: missing video/video_id/clip_id")

        record_id = sample.get("id", f"{Path(str(video_name)).stem}_{idx}")
        nouns = sample.get("nouns", {})
        records.append(
            {
                "id": str(record_id),
                "video": ensure_video_name(str(video_name)),
                "query": normalize_text(query),
                "start_time": coerce_float(start_time),
                "end_time": coerce_float(end_time),
                "duration": coerce_float(duration),
                "nouns": nouns if isinstance(nouns, dict) else {},
            }
        )
    return records


def load_test_records(test_path: str) -> list[dict[str, Any]]:
    with open(test_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return flatten_list_video_dataset(payload)
    if isinstance(payload, dict):
        return flatten_grouped_video_dataset(payload)
    raise TypeError(f"Unsupported test JSON root type: {type(payload).__name__}")


def sorted_nouns(nouns: dict[str, Any], max_nouns: int) -> list[tuple[int, str]]:
    parsed: list[tuple[int, str]] = []
    for key, value in nouns.items():
        text = str(value).strip()
        if not text:
            continue
        try:
            idx = int(key)
        except (TypeError, ValueError):
            idx = len(parsed) + 1
        parsed.append((idx, text))
    parsed.sort(key=lambda item: item[0])
    if max_nouns > 0:
        parsed = parsed[:max_nouns]
    return parsed


def load_yoloe(weights_path: str, device: str):
    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Rebuild the Docker image or install it with "
            "'python -m pip install ultralytics'."
        ) from exc

    model = YOLOE(weights_path)
    if hasattr(model, "to"):
        model.to(device)
    return model


def require_cv_deps() -> None:
    if cv2 is None:
        raise SystemExit("Missing dependency: opencv-python. Install it or run inside the Docker image.")
    if np is None:
        raise SystemExit("Missing dependency: numpy. Install it or run inside the Docker image.")
    if importlib.util.find_spec("clip") is None:
        raise SystemExit(
            "Missing dependency: clip. Rebuild the Docker image, or install the Ultralytics CLIP fork:\n"
            "  python -m pip install "
            "'git+https://github.com/ultralytics/CLIP.git@81ff68ed7ffcac3b40484c914f104f816757308d'"
        )


def set_text_prompts(model: Any, class_names: list[str]) -> None:
    if not class_names:
        return
    text_embeddings = model.get_text_pe(class_names)
    model.set_classes(class_names, text_embeddings)


def build_label_map_from_result(result: Any, size: tuple[int, int], max_nouns: int) -> np.ndarray:
    width, height = size
    label_map = np.zeros((height, width), dtype=np.uint8)
    if getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return label_map

    masks = result.masks.data
    boxes = result.boxes
    if masks is None or boxes is None or len(masks) == 0:
        return label_map

    masks_np = masks.detach().cpu().numpy()
    cls_np = boxes.cls.detach().cpu().numpy().astype(int)
    if getattr(boxes, "conf", None) is not None:
        conf_np = boxes.conf.detach().cpu().numpy()
        order = np.argsort(conf_np)
    else:
        order = np.arange(len(cls_np))

    for det_idx in order:
        noun_id = int(cls_np[det_idx]) + 1
        if noun_id < 1 or noun_id > max_nouns:
            continue
        mask = masks_np[det_idx]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        label_map[mask > 0.5] = noun_id

    return label_map


def render_overlay(frame_bgr: np.ndarray, label_map: np.ndarray, alpha: float, contour_width: int) -> np.ndarray:
    rendered = frame_bgr.astype(np.float32).copy()
    for noun_id, color_bgr in COLOR_MAP_BGR.items():
        mask = label_map == noun_id
        if not np.any(mask):
            continue

        color = np.asarray(color_bgr, dtype=np.float32)
        rendered[mask] = rendered[mask] * (1.0 - alpha) + color * alpha

    rendered_u8 = np.clip(rendered, 0, 255).astype(np.uint8)
    if contour_width > 0:
        for noun_id, color_bgr in COLOR_MAP_BGR.items():
            mask = (label_map == noun_id).astype(np.uint8)
            if not np.any(mask):
                continue
            contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rendered_u8, contours, -1, color_bgr, contour_width)
    return rendered_u8


def probe_video(cap: cv2.VideoCapture, video_path: Path) -> tuple[float, int, int, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = 1.0
    if total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return fps, total_frames, width, height


def generate_overlay_for_record(model: Any, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    nouns = sorted_nouns(record.get("nouns", {}), args.max_nouns)
    if not nouns:
        return {"id": record["id"], "status": "skipped_no_nouns"}

    class_names = [noun for _, noun in nouns]
    set_text_prompts(model, class_names)

    raw_video_path = Path(args.raw_video_root) / record["video"]
    if not raw_video_path.exists():
        if args.skip_missing:
            return {"id": record["id"], "status": "skipped_missing_video", "video": str(raw_video_path)}
        raise FileNotFoundError(f"Missing raw video for {record['id']}: {raw_video_path}")

    overlay_root = Path(args.overlay_root)
    overlay_root.mkdir(parents=True, exist_ok=True)
    output_path = overlay_root / f"{record['id']}_overlay.mp4"
    if output_path.exists() and not args.overwrite:
        return {"id": record["id"], "status": "exists", "output": str(output_path)}

    cap = cv2.VideoCapture(str(raw_video_path))
    if not cap.isOpened():
        cap.release()
        if args.skip_missing:
            return {"id": record["id"], "status": "skipped_unreadable_video", "video": str(raw_video_path)}
        raise RuntimeError(f"Failed to open video for {record['id']}: {raw_video_path}")

    fps, total_frames, _, _ = probe_video(cap, raw_video_path)
    frame_stride = max(1, int(args.frame_stride))
    output_fps = max(0.5, fps / frame_stride)
    render_size = int(args.render_size)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (render_size, render_size))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    written = 0
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            frame_small = cv2.resize(frame_bgr, (render_size, render_size), interpolation=cv2.INTER_LINEAR)
            result = model.predict(
                frame_small,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            label_map = build_label_map_from_result(
                result=result,
                size=(render_size, render_size),
                max_nouns=len(class_names),
            )
            overlay_frame = render_overlay(
                frame_bgr=frame_small,
                label_map=label_map,
                alpha=args.mask_alpha,
                contour_width=args.contour_width,
            )
            writer.write(overlay_frame)
            written += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if written == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"No frames were written for {record['id']}: {raw_video_path}")

    return {
        "id": record["id"],
        "status": "written",
        "output": str(output_path),
        "video": str(raw_video_path),
        "nouns": dict((str(idx), noun) for idx, noun in nouns),
        "source_fps": fps,
        "source_frames": total_frames,
        "output_fps": output_fps,
        "output_frames": written,
    }


def write_json(path: str | None, payload: Any) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MarkIt overlay videos named <id>_overlay.mp4 with YOLOE masks."
    )
    parser.add_argument("--test_path", required=True, help="MarkIt test JSON")
    parser.add_argument("--raw_video_root", required=True, help="Directory containing raw videos")
    parser.add_argument("--overlay_root", required=True, help="Output directory for <id>_overlay.mp4")
    parser.add_argument(
        "--yoloe_weights",
        default="/models/local/YOLOE-Large/yoloe-v8l-seg.pt",
        help="Path to YOLOE segmentation weights",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--render_size", type=int, default=336)
    parser.add_argument("--mask_alpha", type=float, default=0.3)
    parser.add_argument("--contour_width", type=int, default=3)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_nouns", type=int, default=3)
    parser.add_argument("--max_records", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--summary_json", help="Optional generation summary JSON")
    args = parser.parse_args()

    require_cv_deps()

    if not Path(args.yoloe_weights).exists():
        raise FileNotFoundError(f"Missing YOLOE weights: {args.yoloe_weights}")

    records = load_test_records(args.test_path)
    if args.start_index > 0:
        records = records[args.start_index :]
    if args.max_records > 0:
        records = records[: args.max_records]

    model = load_yoloe(args.yoloe_weights, args.device)
    results = []
    for record in tqdm(records, desc="overlays"):
        result = generate_overlay_for_record(model, record, args)
        results.append(result)

    summary = {
        "test_path": args.test_path,
        "raw_video_root": args.raw_video_root,
        "overlay_root": args.overlay_root,
        "yoloe_weights": args.yoloe_weights,
        "num_records": len(records),
        "statuses": {},
        "results": results,
    }
    for result in results:
        status = result["status"]
        summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
    write_json(args.summary_json, summary)

    print("overlay generation summary:")
    for status, count in sorted(summary["statuses"].items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
