#!/usr/bin/env python3
# Author: Hayden Feddock

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

UNCROPPED_H = 1088  # original image height
UNCROPPED_W = 2048  # original image width

CROP_TOP  = 3        # pixels removed before demosaic
CROP_LEFT = 0
DOWNSAMPLE = 5       # 5×5 pattern (each band is 1/5 resolution)

CROPPED_H = 1080
CROPPED_W = 2045
DEMOSAIC_H = CROPPED_H // DOWNSAMPLE   # 216
DEMOSAIC_W = CROPPED_W // DOWNSAMPLE   # 409


def demosaic_ximea_5x5(image_path: str, sort_bands: bool = True):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    cropped_image = image[CROP_TOP:1083, CROP_LEFT:2045]          # H=1080, W=2045
    h, w = cropped_image.shape
    if h % 5 or w % 5:
        raise ValueError("Cropped dims not divisible by 5")

    block_rows, block_cols = h // 5, w // 5
    bw = [
        [886, 896, 877, 867, 951],
        [793, 806, 782, 769, 675],
        [743, 757, 730, 715, 690],
        [926, 933, 918, 910, 946],
        [846, 857, 836, 824, 941],
    ]
    cube = {}
    for ro in range(5):
        for co in range(5):
            band = cropped_image[ro::5, co::5]
            if band.shape != (block_rows, block_cols):
                raise ValueError("Band shape mismatch")
            cube[bw[ro][co]] = band
    return dict(sorted(cube.items())) if sort_bands else cube


def hypercube_dict_to_array(hypercube_dict):
    first = next(iter(hypercube_dict.values()))
    cube = np.empty((len(hypercube_dict), *first.shape), dtype=first.dtype)
    for i, b in enumerate(hypercube_dict.values()):
        cube[i] = b
    return cube


# ------------------------ bbox conversion utils ------------------------ #



def _convert_bbox(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Takes *normalised* YOLO bbox in the RAW image reference frame and
    returns a *normalised* bbox in the final demosaiced PNG frame.
    """
    # 1) to pixels in RAW frame
    xc *= img_w
    yc *= img_h
    w  *= img_w
    h  *= img_h

    # 2) crop offset                 (x offset is 0)
    xc -= CROP_LEFT
    yc -= CROP_TOP

    # 3) down-sample
    xc /= DOWNSAMPLE
    yc /= DOWNSAMPLE
    w  /= DOWNSAMPLE
    h  /= DOWNSAMPLE

    # 4) back to normalised coords for new size
    xc /= DEMOSAIC_W
    yc /= DEMOSAIC_H
    w  /= DEMOSAIC_W
    h  /= DEMOSAIC_H
    return xc, yc, w, h


def _clip_bbox(xc, yc, w, h):
    """
    Clip bbox to [0,1] range; drop if it vanishes. Returns None to drop.
    """
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = xc + w / 2, yc + h / 2
    x1, y1, x2, y2 = map(lambda v: max(0.0, min(1.0, v)), (x1, y1, x2, y2))
    w_new, h_new = x2 - x1, y2 - y1
    if w_new <= 0 or h_new <= 0:
        return None
    xc_new, yc_new = x1 + w_new / 2, y1 + h_new / 2
    return xc_new, yc_new, w_new, h_new
# ----------------------------------------------------------------------- #


# ---------------------------- batch driver ----------------------------- #
class SetType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

def process_dataset(
    src_root: Path,
    set_type: SetType,
    dst_root: Path,
    bands_rgb: List[int] = (886, 793, 743),
    img_exts: Tuple[str, ...] = (".pgm", ".png", ".tif", ".tiff"),
):
    src_images = (src_root / "images" / set_type).glob("*")
    src_images = [p for p in src_images if p.suffix.lower() in img_exts]
    if not src_images:
        raise RuntimeError(f"No images with {img_exts} found in {src_root / 'images' / set_type}")

    dst_img_dir = dst_root / "images" / set_type
    dst_lbl_dir = dst_root / "labels" / set_type
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in src_images:
        print(f"→ {img_path.name}")

        # ---------- demosaic & build 3-channel PNG ----------
        cube = demosaic_ximea_5x5(img_path)
        try:
            png_rgb = np.stack(
                [cube[b] for b in bands_rgb], axis=-1
            )  # H×W×3, uint8
        except KeyError as e:
            raise KeyError(
                f"Requested band {e} not found in demosaiced cube."
            ) from e

        # simple per-band normalisation to uint8 for visualization / ML
        png_rgb = np.stack(
            [
                cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
                for c in cv2.split(png_rgb)
            ],
            axis=-1,
        ).astype(np.uint8)

        out_png = dst_img_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_png), cv2.cvtColor(png_rgb, cv2.COLOR_RGB2BGR))

        # ---------- label conversion (if any) ----------
        src_lbl = src_root / "labels" / set_type / f"{img_path.stem}.txt"
        if not src_lbl.is_file():
            print(f"No label file {src_lbl} found, skipping.")
            continue

        new_lines = []
        with open(src_lbl) as f:
            for ln in f:
                cls, xc, yc, w, h = map(float, ln.strip().split())
                xc, yc, w, h = _convert_bbox(
                    xc, yc, w, h, img_w=UNCROPPED_W, img_h=UNCROPPED_H
                )

                clipped = _clip_bbox(xc, yc, w, h)
                if clipped is None:
                    continue  # box vanished after crop/down-sample
                xc, yc, w, h = clipped
                new_lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        if new_lines:
            out_lbl = dst_lbl_dir / src_lbl.name
            out_lbl.write_text("".join(new_lines))


# ---------------------------- CLI wrapper ------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Batch-demosaic YOLO dataset")
    parser.add_argument("--src", required=True, type=Path, help="source dataset root")
    parser.add_argument(
        "--dst", type=Path, default=None, help="output dir (default src+'_demosaic')"
    )
    parser.add_argument(
        "--bands",
        nargs=3,
        type=int,
        metavar=("B1", "B2", "B3"),
        default=(886, 793, 743),
        # default=(757, 769, 793),  # RGB bands to use in output PNG
        help="three bandwidths to map → R,G,B",
    )
    parser.add_argument(
        "--set",
        type=str,
        choices=[SetType.TRAIN, SetType.VAL, SetType.TEST],
        default=SetType.TRAIN,
        help="dataset split to process (train/val/test)",
    )
    args = parser.parse_args()
    dst = args.dst or args.src.with_name(args.src.name + "_demosaic")
    process_dataset(args.src, args.set, dst, list(args.bands))
    print(f"\nFinished. Demosaiced dataset written to {dst}")


if __name__ == "__main__":
    main()
