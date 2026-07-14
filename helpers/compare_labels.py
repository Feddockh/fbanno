#!/usr/bin/env python3
"""
Side-by-side comparison of predicted vs ground-truth YOLO labels on NIR frames.

For each frame, shows three panels on the demosaiced (false-color) Ximea image:
predicted boxes | ground-truth boxes | both overlaid (pred=orange, GT=lime).
The title reports the per-frame mean IoU of greedily matched box pairs.

Both label sets must be normalized to the raw mosaic frame (as produced by
main.py and stored in datasets/field_dataset_nir/labels).

Usage:
    # Interactive click-through (Right/n = next, Left/p = prev, q/Esc = quit)
    python helpers/compare_labels.py \
        --pred-dir datasets/rivendale_v6/ximea/labels/train \
        --gt-dir datasets/field_dataset_nir/labels/val \
        --image-dir datasets/field_dataset_nir/images/val

    # Batch-render PNGs instead (sorted worst-first by mean IoU with --sort)
    python helpers/compare_labels.py ... --save-dir outputs/comparisons/val [--sort]
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from batch_demosaic import demosaic_ximea_5x5, _convert_bbox  # noqa: E402
from evaluate import load_yolo_file, iou_matrix, greedy_match  # noqa: E402

DEFAULT_RGB_BANDS = (886, 793, 743)
PRED_COLOR = "orange"
GT_COLOR = "lime"


def load_nir_frame(image_path):
    """Demosaic a raw Ximea mosaic frame into a false-color RGB uint8 image."""
    cube = demosaic_ximea_5x5(image_path)
    bands = [b for b in DEFAULT_RGB_BANDS if b in cube] or list(cube.keys())[:3]
    img = cv2.merge([cv2.normalize(cube[b], None, 0, 255, cv2.NORM_MINMAX) for b in bands[:3]])
    return img.astype("uint8")


def remap_boxes(labels, boxes_xyxy, raw_w, raw_h):
    """Convert normalized raw-frame xyxy boxes to normalized demosaiced xywh."""
    remapped = []
    for cls, (x1, y1, x2, y2) in zip(labels, boxes_xyxy):
        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        remapped.append((int(cls), *_convert_bbox(xc, yc, w, h, img_w=raw_w, img_h=raw_h)))
    return remapped


def draw_boxes(ax, boxes, img_w, img_h, color):
    for cls, xc, yc, bw, bh in boxes:
        x1, y1 = (xc - bw / 2) * img_w, (yc - bh / 2) * img_h
        ax.add_patch(patches.Rectangle((x1, y1), bw * img_w, bh * img_h,
                                       linewidth=1.5, edgecolor=color, facecolor="none"))
        ax.text(x1, max(y1 - 3, 0), str(cls), color="black", fontsize=7,
                bbox=dict(facecolor=color, edgecolor="none", pad=0.5))


def frame_mean_iou(pred_boxes, gt_boxes):
    """Mean IoU of greedily matched pred/GT pairs (class-agnostic); None if no boxes."""
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return None
    matches = greedy_match(iou_matrix(pred_boxes, gt_boxes), threshold=1e-6)
    n = max(len(pred_boxes), len(gt_boxes))
    return sum(m[2] for m in matches) / n if n else None


def render(fig, axes, image_path, pred_path, gt_path):
    img = load_nir_frame(image_path)
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    raw_h, raw_w = raw.shape[:2]
    h, w = img.shape[:2]

    pl, pb = load_yolo_file(pred_path if os.path.exists(pred_path) else None)
    gl, gb = load_yolo_file(gt_path)
    miou = frame_mean_iou(pb, gb)

    pred_boxes = remap_boxes(pl, pb, raw_w, raw_h)
    gt_boxes = remap_boxes(gl, gb, raw_w, raw_h)

    panels = [
        (f"predicted ({len(pred_boxes)})", [(pred_boxes, PRED_COLOR)]),
        (f"ground truth ({len(gt_boxes)})", [(gt_boxes, GT_COLOR)]),
        ("overlay", [(gt_boxes, GT_COLOR), (pred_boxes, PRED_COLOR)]),
    ]
    for ax, (title, layers) in zip(axes, panels):
        ax.clear()
        ax.imshow(img)
        for boxes, color in layers:
            draw_boxes(ax, boxes, w, h, color)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    name = os.path.basename(image_path)
    miou_str = f"mean IoU {miou:.3f}" if miou is not None else "no boxes"
    missing = "" if os.path.exists(pred_path) else "  [NO PREDICTION FILE]"
    fig.suptitle(f"{name}  —  {miou_str}{missing}", fontsize=10)
    return miou


class CompareViewer:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4.5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def draw(self):
        render(self.fig, self.axes, *self.frames[self.idx])
        self.fig.suptitle(f"[{self.idx + 1}/{len(self.frames)}] " + self.fig._suptitle.get_text(),
                          fontsize=10)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ("right", "n"):
            self.idx = (self.idx + 1) % len(self.frames)
            self.draw()
        elif event.key in ("left", "p"):
            self.idx = (self.idx - 1) % len(self.frames)
            self.draw()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


def main():
    p = argparse.ArgumentParser(description="Compare predicted vs GT YOLO labels on NIR frames")
    p.add_argument("--pred-dir", required=True, help="Directory of predicted label files")
    p.add_argument("--gt-dir", required=True, help="Directory of ground-truth label files")
    p.add_argument("--image-dir", required=True, help="Directory of raw Ximea mosaic images")
    p.add_argument("--save-dir", default=None, help="Render PNGs here instead of interactive view")
    p.add_argument("--sort", action="store_true",
                   help="With --save-dir: prefix filenames with mean IoU so worst frames sort first")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N frames")
    args = p.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.txt")))
    frames = []
    for gt_path in gt_files:
        stem = os.path.splitext(os.path.basename(gt_path))[0]
        image_path = os.path.join(args.image_dir, stem + ".png")
        if not os.path.exists(image_path):
            continue
        frames.append((image_path, os.path.join(args.pred_dir, stem + ".txt"), gt_path))
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit("No frames with both a GT label and an image found.")

    if args.save_dir:
        matplotlib.use("Agg")
        os.makedirs(args.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for image_path, pred_path, gt_path in frames:
            miou = render(fig, axes, image_path, pred_path, gt_path)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            prefix = f"iou{miou:.3f}_" if (args.sort and miou is not None) else ""
            out = os.path.join(args.save_dir, f"{prefix}{stem}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight")
            print(out)
        plt.close(fig)
    else:
        viewer = CompareViewer(frames)
        print("Controls: Right/n = next, Left/p = prev, q/Esc = quit")
        plt.show()


if __name__ == "__main__":
    main()
