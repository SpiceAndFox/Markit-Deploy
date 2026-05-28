import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        for i, item in enumerate(iterable):
            print(f"\r[{i + 1}]", end="", flush=True, file=sys.stderr)
            yield item
        print(file=sys.stderr)


LEGEND_ITEMS = [
    ("GT", (0, 200, 0)),
    ("Pred", (0, 0, 200)),
    ("Both", (0, 200, 200)),
]

FALLBACK_EXTENSIONS = [".mov", ".mp4", ".avi", ".webm"]


def _format_seconds(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def _find_video(video_name: str, root: str) -> str | None:
    path = os.path.join(root, video_name)
    if os.path.exists(path):
        return path
    stem = os.path.splitext(video_name)[0]
    for ext in FALLBACK_EXTENSIONS:
        alt = os.path.join(root, stem + ext)
        if os.path.exists(alt):
            return alt
    return None


def _draw_rounded_filled_rect(img, x1, y1, x2, y2, color, radius=4):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius - 1, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius - 1), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius - 1, y2 - radius - 1), (radius, radius), 0, 0, 90, color, -1)
    return cv2.addWeighted(overlay, 0.85, img, 0.15, 0)


def _draw_timeline(canvas, width, video_h, frame_idx, total_frames, gt_from, gt_to, pred_from, pred_to,
                   fps_val, font_scale, timeline_height):
    pad = 10
    bar_y = video_h + pad
    bar_h = timeline_height - 2 * pad
    bar_w = width - 2 * pad

    cv2.rectangle(canvas, (pad, bar_y), (pad + bar_w, bar_y + bar_h), (60, 60, 60), -1)

    def to_x(f):
        return pad + int(f / max(1, total_frames) * bar_w)

    gt_x1 = to_x(gt_from)
    gt_x2 = to_x(gt_to)
    cv2.rectangle(canvas, (gt_x1, bar_y), (max(gt_x1 + 1, gt_x2), bar_y + bar_h), (0, 200, 0), -1)

    pred_x1 = to_x(pred_from)
    pred_x2 = to_x(pred_to)
    pred_h = int(bar_h * 0.6)
    pred_y = bar_y + int((bar_h - pred_h) / 2)
    cv2.rectangle(canvas, (pred_x1, pred_y), (max(pred_x1 + 1, pred_x2), pred_y + pred_h), (0, 0, 240), -1)

    cursor_x = to_x(frame_idx)
    cv2.line(canvas, (cursor_x, bar_y), (cursor_x, bar_y + bar_h), (255, 255, 255), 2)

    cv2.putText(canvas, "GT", (pad + 4, bar_y + bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Pred", (pad + 4, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    duration = total_frames / max(1.0, fps_val)
    ts_text = _format_seconds(frame_idx / max(1.0, duration) * duration)
    (tw, th), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.45, 1)
    cv2.putText(canvas, ts_text, (width - tw - pad - 4, bar_y + bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_legend(canvas, width, video_h, font_scale):
    lw, lh = 130, 90
    x0 = width - lw - 10
    y0 = video_h - lh - 10
    legend_img = np.zeros((lh, lw, 3), dtype=np.uint8)
    legend_img[:] = (40, 40, 40)
    legend_img = _draw_rounded_filled_rect(legend_img, 0, 0, lw, lh, (40, 40, 40), radius=8)

    for i, (label, color) in enumerate(LEGEND_ITEMS):
        yy = 12 + i * 22
        cv2.rectangle(legend_img, (10, yy), (28, yy + 14), color, -1)
        cv2.putText(legend_img, label, (34, yy + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    mask = legend_img > 0
    roi = canvas[y0:y0 + lh, x0:x0 + lw]
    roi[mask] = legend_img[mask]


def _draw_info_overlay(canvas, width, video_h, pred, fps, pred_from, pred_to, gt_from, gt_to, font_scale):
    query = pred.get("query", "")
    if len(query) > 70:
        query = query[:67] + "..."

    gt_str = f"GT: {pred['gt_start']:.1f}s - {pred['gt_end']:.1f}s"
    pred_str = f"Pred: f{pred_from}-{pred_to}  ({pred_from / max(1, fps):.1f}s - {pred_to / max(1, fps):.1f}s)"

    text_margin = 12
    y = text_margin
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 2))

    for text in [query, gt_str, pred_str]:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        bg_y1 = y - 2
        bg_y2 = y + th + 4
        canvas[bg_y1:bg_y2, text_margin:text_margin + tw + 8] = (40, 40, 40)
        cv2.putText(canvas, text, (text_margin + 4, y + th), font, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        y += th + 10


def _get_border_color(frame_idx, pred_from, pred_to, gt_from, gt_to):
    in_pred = pred_from <= frame_idx <= pred_to
    in_gt = gt_from <= frame_idx <= gt_to
    if in_pred and in_gt:
        return (0, 220, 220)
    if in_pred:
        return (0, 0, 220)
    if in_gt:
        return (0, 220, 0)
    return None


def _draw_border(frame, border_thickness, color):
    h, w = frame.shape[:2]
    t = border_thickness
    frame[:t, :] = color
    frame[h - t:, :] = color
    frame[:, :t] = color
    frame[:, w - t:] = color


def process_entry(pred, raw_video_root, output_dir, font_scale, border_thickness, timeline_height):
    video_name = pred["video"]
    video_path = _find_video(video_name, raw_video_root)
    if video_path is None:
        print(f"[skip] video not found: {video_name}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[skip] cannot open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_stride = int(pred.get("frame_stride", 1))
    pred_from_raw = int(pred["pred_start"]) * frame_stride
    pred_to_raw = int(pred["pred_end"]) * frame_stride

    gt_from_raw = int(pred["gt_start"] * fps)
    gt_to_raw = int(pred["gt_end"] * fps)
    gt_from_raw = max(0, min(gt_from_raw, total_frames - 1))
    gt_to_raw = max(0, min(gt_to_raw, total_frames - 1))

    canvas_w = width
    canvas_h = height + timeline_height

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pred['id']}_vis.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    disp_pred_from = int(pred["pred_start"])
    disp_pred_to = int(pred["pred_end"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:height, :width] = frame

        border_color = _get_border_color(frame_idx, pred_from_raw, pred_to_raw, gt_from_raw, gt_to_raw)
        if border_color is not None:
            _draw_border(canvas, border_thickness, border_color)

        _draw_timeline(canvas, width, height, frame_idx, total_frames,
                       gt_from_raw, gt_to_raw, pred_from_raw, pred_to_raw,
                       fps, font_scale, timeline_height)
        _draw_info_overlay(canvas, width, height, pred, fps,
                           disp_pred_from, disp_pred_to, gt_from_raw, gt_to_raw, font_scale)
        _draw_legend(canvas, width, height, font_scale)

        out.write(canvas)
        frame_idx += 1

    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MarkIt/numpro predictions on raw videos with GT vs prediction overlays."
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON file.")
    parser.add_argument("--raw_video_root", type=str, required=True,
                        help="Root directory containing raw videos.")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations",
                        help="Output directory for annotated videos.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max number of predictions to process (-1 = all).")
    parser.add_argument("--font_scale", type=float, default=0.55,
                        help="Base font scale for text overlays (default: 0.55).")
    parser.add_argument("--border_thickness", type=int, default=6,
                        help="Border thickness in pixels (default: 6).")
    parser.add_argument("--timeline_height", type=int, default=50,
                        help="Height of the bottom timeline bar in pixels (default: 50).")
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if args.max_samples > 0:
        predictions = predictions[: args.max_samples]

    for pred in tqdm(predictions, desc="Visualizing"):
        process_entry(pred, args.raw_video_root, args.output_dir,
                      args.font_scale, args.border_thickness, args.timeline_height)

    print(f"Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
