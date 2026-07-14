# fb_reproject

Transfers YOLO bounding-box annotations from the RGB camera (firefly_left) to
the NIR camera (ximea) using calibrated intrinsics/extrinsics, FoundationStereo
depth (firefly_left/right pair), and SAM2 masks.

## Usage

```
python3 main.py --rgb-dir <dir> --rgb-labels-dir <dir> \
                --right-dir <dir> --nir-dir <dir> \
                [--output-dir outputs/reprojected_labels] \
                [--calib-dir calibration_files] \
                [--limit N]
```

Each modality is an independent flat directory of images: `--rgb-dir` (RGB
source images), `--rgb-labels-dir` (YOLO labels for the RGB boxes to
reproject), `--right-dir` (RGB stereo-right images, needed for depth), and
`--nir-dir` (NIR target images). Frames are matched across directories by
image basename. Only the RGB camera needs input labels. Reprojected NIR YOLO
labels are written to `--output-dir` (never back into an input directory).
`--calib-dir` overrides where per-camera calibration YAMLs (`firefly_left.yaml`,
`firefly_right.yaml`, `ximea.yaml`) are loaded from; defaults to
`calibration_files/` in this repo.

## Evaluation

`evaluate.py` scores a directory of predicted YOLO labels against ground truth
(P/R/F1 at IoU thresholds, mean IoU, and human-fix workload; see `RESULTS.md`
for current numbers):

```
python3 evaluate.py --pred-dir outputs/reprojected_labels \
                    --gt-dir datasets/field_dataset_nir/labels/val \
                    --class-names "Shepherds_Crook" "Canker"
```

Pass `--image-dir` to also generate side-by-side ground-truth/predicted
comparison images (ground truth left, predictions right; handles both raw
Ximea mosaic frames and regular images), each filename prefixed with its
per-frame mean IoU. Only the first `--vis-limit` frames are rendered (default
50; `0` = no limit); `--sort` orders frames worst-first by mean IoU before
that limit is applied. With `--save-dir`, PNGs are batch rendered; without
it, an interactive click-through viewer opens instead:

```
python3 evaluate.py --pred-dir outputs/reprojected_labels \
                    --gt-dir datasets/field_dataset_nir/labels/val \
                    --image-dir datasets/field_dataset_nir/images/val \
                    --save-dir outputs/comparisons/val --sort
```

Ground truth: `datasets/field_dataset_nir` holds hand-labeled NIR boxes
(train/val/test splits whose frames exactly partition rivendale_v6's train
set); `datasets/field_dataset` is the matching RGB dataset (identical to
`rivendale_v6/firefly_left`).

## Legacy / known-stale

- `demo.py` references classes that no longer exist (`MultiCamDataset`,
  `SetType.ALL`, `save_annos_coco`) and will not run.
- `dataset.py` module constants `DATA_DIR`/`YOLO_DATA_DIR` and the
  `YoloV5MultiCamDataset` / `YoloMultiCamDataset` / `COCOMultiCamDataset`
  classes target older multi-cam dataset layouts (a shared dataset root with
  per-camera subdirs, e.g. `rivendale_v6`) and are unused by `main.py`, which
  now uses `ReprojectionDataset` (independent flat per-modality directories).
- `helpers/` contains one-off scripts with hardcoded machine-specific paths.

## Misc

Symbolic link: `ln -s /path/to/target_dir /path/to/link_name`
