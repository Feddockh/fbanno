#!/usr/bin/env python3
"""
Simple click-through viewer for YOLO-format datasets (field_dataset / field_dataset_nir /
rivendale_v5 / rivendale_v6). Draws the image with its label boxes overlaid and lets you
step through the split with the keyboard.

Handles two image kinds transparently:
  - Regular viewable images (RGB or already-demosaiced Ximea) — shown as-is, boxes
    already in normalized coords matching the image.
  - Raw un-demosaiced Ximea 5x5 mosaic frames (single-channel, ~1088x2048) — demosaiced
    on the fly into a false-color image, boxes remapped into the demosaiced frame.

Usage:
    python helpers/view_dataset.py datasets/field_dataset --split train
    python helpers/view_dataset.py datasets/field_dataset_nir --split val
    python helpers/view_dataset.py datasets/rivendale_v6/ximea --split train
    python helpers/view_dataset.py datasets/rivendale_v5/ximea_demosaic --split train

Controls: Right/Left (or n/p) = next/prev image, q/Esc = quit.
"""

import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_demosaic import demosaic_ximea_5x5, _convert_bbox  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".pgm", ".tif", ".tiff")

# Ximea raw mosaic frames are single-channel and this specific resolution.
RAW_MOSAIC_SHAPE = (1088, 2048)
DEFAULT_RGB_BANDS = (886, 793, 743)


def find_data_yaml(start_dir):
    """Walk upward from start_dir looking for a data.yaml (dataset root config)."""
    d = os.path.abspath(start_dir)
    for _ in range(4):
        candidate = os.path.join(d, "data.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def load_class_names(start_dir):
    path = find_data_yaml(start_dir)
    if path is None:
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    return {int(k): v for k, v in names.items()}


def load_label(label_path):
    """Return list of (cls, xc, yc, w, h) normalized boxes."""
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, xc, yc, w, h = line.split()
            boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def load_frame(image_path, label_path):
    """Load an image (demosaicing raw Ximea frames if needed) and its boxes,
    returning an RGB uint8 array and boxes normalized to that array."""
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(image_path)

    boxes = load_label(label_path)

    if raw.ndim == 2 and raw.shape == RAW_MOSAIC_SHAPE:
        # Raw un-demosaiced Ximea mosaic: demosaic + remap boxes.
        cube = demosaic_ximea_5x5(image_path)
        bands = [b for b in DEFAULT_RGB_BANDS if b in cube] or list(cube.keys())[:3]
        img = cv2.merge([cv2.normalize(cube[b], None, 0, 255, cv2.NORM_MINMAX) for b in bands[:3]])
        img = img.astype("uint8")
        h_raw, w_raw = raw.shape
        boxes = [
            (cls, *_convert_bbox(xc, yc, w, h, img_w=w_raw, img_h=h_raw))
            for cls, xc, yc, w, h in boxes
        ]
    else:
        img = raw if raw.ndim == 2 else cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if raw.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img, boxes


class DatasetViewer:
    def __init__(self, root, split, class_names):
        self.image_dir = os.path.join(root, "images", split)
        self.label_dir = os.path.join(root, "labels", split)
        self.class_names = class_names

        self.image_paths = sorted(
            p for p in glob.glob(os.path.join(self.image_dir, "*"))
            if os.path.splitext(p)[1].lower() in IMG_EXTS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.idx = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def label_path_for(self, image_path):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.label_dir, f"{stem}.txt")

    def draw(self):
        image_path = self.image_paths[self.idx]
        img, boxes = load_frame(image_path, self.label_path_for(image_path))
        h, w = img.shape[:2]

        self.ax.clear()
        self.ax.imshow(img)
        for cls, xc, yc, bw, bh in boxes:
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            rect = patches.Rectangle(
                (x1, y1), bw * w, bh * h,
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            self.ax.add_patch(rect)
            label = self.class_names.get(cls, str(cls))
            self.ax.text(
                x1, max(y1 - 4, 0), label, color="black", fontsize=9,
                bbox=dict(facecolor="lime", edgecolor="none", pad=1),
            )

        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.image_paths)}] {os.path.basename(image_path)}"
            f"  ({len(boxes)} box{'es' if len(boxes) != 1 else ''})"
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ("right", "n"):
            self.idx = (self.idx + 1) % len(self.image_paths)
            self.draw()
        elif event.key in ("left", "p"):
            self.idx = (self.idx - 1) % len(self.image_paths)
            self.draw()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def show(self):
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Click-through viewer for YOLO field/rivendale datasets")
    p.add_argument("root", help="Dataset root containing images/<split> and labels/<split> "
                                 "(e.g. datasets/field_dataset, datasets/rivendale_v6/ximea)")
    p.add_argument("--split", default="train", help="Split to view (train/val/test/...)")
    return p.parse_args()


def main():
    args = parse_args()
    class_names = load_class_names(args.root)
    viewer = DatasetViewer(args.root, args.split, class_names)
    print("Controls: Right/n = next, Left/p = prev, q/Esc = quit")
    viewer.show()


if __name__ == "__main__":
    main()
