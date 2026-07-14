# RGB → NIR annotation reprojection: evaluation

Reprojected ximea (NIR) bounding boxes produced by `main.py` (firefly_left RGB
labels → FoundationStereo depth → SAM2 masks → 3D → ximea plane) scored against
the hand-labeled NIR ground truth in `datasets/field_dataset_nir/labels/{split}`.

Both prediction and GT labels are normalized YOLO boxes in the raw 2048×1088
ximea image space, matched by filename (timestamps are unique across splits).
The reprojected boxes carry no confidence scores, so mAP is degenerate; metrics
are precision / recall / F1 at IoU thresholds plus mean IoU over matched pairs
(greedy one-to-one matching, per class). Computed with:

```
python3 evaluate.py \
    --pred-dir <labels dir> \
    --gt-dir datasets/field_dataset_nir/labels/<split> \
    --class-names "Shepherds_Crook" "Canker"
```

Notes on reading the tables:

- `agnostic` ignores class ids (localization only). It is essentially identical
  to `all` everywhere, meaning class labels transfer cleanly (expected — they
  are copied from the RGB annotation) and all error is localization/coverage.
- Missing prediction files are frames the pipeline skipped (no RGB boxes, or no
  valid masks survived reprojection filtering); their GT boxes count as FN.

## Baseline: pipeline outputs from 2025-09-09

Predictions: `datasets/rivendale_v6/ximea/labels/train` (1231 files, unmodified
output of the previous pipeline run).

### F1 summary (all classes)

| Split | Frames (missing preds) | GT boxes | Pred boxes | P/R/F1 @0.25 | P/R/F1 @0.50 | P/R/F1 @0.75 | mIoU @0.50 |
|---|---|---|---|---|---|---|---|
| train | 983 (34) | 1245 | 1363 | 0.809 / 0.885 / **0.845** | 0.711 / 0.778 / **0.743** | 0.504 / 0.552 / 0.527 | 0.853 |
| val   | 155 (6)  | 210  | 233  | 0.760 / 0.843 / **0.799** | 0.670 / 0.743 / **0.704** | 0.481 / 0.533 / 0.506 | 0.863 |
| test  | 137 (4)  | 177  | 192  | 0.891 / 0.966 / **0.927** | 0.740 / 0.802 / **0.770** | 0.536 / 0.582 / 0.558 | 0.954 |

Per-class detail lives in `outputs/eval/baseline_{split}.json`.

### Takeaways

- The transfer recovers **~78% of NIR GT boxes at IoU 0.5** (recall 0.74–0.80),
  with precision ~0.67–0.74 — the process clearly produces usable annotations.
- At the loose IoU 0.25 threshold (i.e. "a box landed on the right object"),
  F1 is 0.80–0.93, so gross misprojections are rare; most loss between 0.25
  and 0.75 is box tightness, not placement.
- Matched boxes are tight: mean IoU of matches at the 0.5 threshold is ~0.86.
- ~3% of frames were skipped by the pipeline entirely (no valid masks).

## Fresh rerun (this branch)

Predictions: `outputs/reprojected_labels/` from rerunning `main.py` on this
branch. Spot checks showed the rerun reproduces the 2025-09-09 outputs almost
byte-for-byte (sub-pixel GPU nondeterminism on isolated boxes), so numbers are
expected to match the baseline closely.

_(pending — run in progress)_
