# fb_reproject

Transfers YOLO bounding-box annotations from the RGB camera (firefly_left) to
the NIR camera (ximea) using calibrated intrinsics/extrinsics, FoundationStereo
depth (firefly_left/right pair), and SAM2 masks.

## Usage

```
python3 main.py [--data-dir datasets/rivendale_v6] \
                [--output-dir outputs/reprojected_labels] \
                [--limit N]
```

Reads the multi-cam YOLO dataset at `--data-dir` (per-camera
`<cam>/images/<split>/`, `<cam>/labels/<split>/` layout) and writes reprojected
ximea YOLO labels to `--output-dir` (never back into the input dataset).

## Evaluation

`evaluate.py` scores a directory of predicted YOLO labels against ground truth
(P/R/F1 at IoU thresholds + mean IoU; see `RESULTS.md` for current numbers):

```
python3 evaluate.py --pred-dir outputs/reprojected_labels \
                    --gt-dir datasets/field_dataset_nir/labels/val \
                    --class-names "Shepherds_Crook" "Canker"
```

Ground truth: `datasets/field_dataset_nir` holds hand-labeled NIR boxes
(train/val/test splits whose frames exactly partition rivendale_v6's train
set); `datasets/field_dataset` is the matching RGB dataset (identical to
`rivendale_v6/firefly_left`).

## Legacy / known-stale

- `demo.py` references classes that no longer exist (`MultiCamDataset`,
  `SetType.ALL`, `save_annos_coco`) and will not run.
- `dataset.py` module constants `DATA_DIR`/`YOLO_DATA_DIR` and the
  `YoloMultiCamDataset` / `COCOMultiCamDataset` classes target older dataset
  layouts (rivendale COCO / flat YOLO) and are unused by the pipeline.
- `helpers/` contains one-off scripts with hardcoded machine-specific paths.

## Misc

Symbolic link: `ln -s /path/to/target_dir /path/to/link_name`
