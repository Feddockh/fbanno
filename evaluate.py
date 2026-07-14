"""Score predicted YOLO labels against ground-truth YOLO labels.

Compares two directories of YOLO-format label files (normalized
``cls xc yc w h``) that live in the same image coordinate space, e.g. the
reprojected ximea labels produced by main.py vs the hand-labeled NIR ground
truth in datasets/field_dataset_nir/labels/<split>.

The predicted boxes carry no confidence scores, so mAP is degenerate; instead
this reports precision / recall / F1 at a set of IoU thresholds (per class,
overall, and class-agnostic) plus the mean IoU over matched pairs.

Example:
    python evaluate.py \
        --pred-dir datasets/rivendale_v6/ximea/labels/train \
        --gt-dir datasets/field_dataset_nir/labels/val \
        --iou-thresholds 0.25 0.5 0.75 \
        --json results_val.json
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_yolo_file(path):
    """Parse a YOLO label file into (labels [N], boxes [N,4] xyxy normalized)."""
    labels, boxes = [], []
    if path is not None and os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = map(float, parts)
                labels.append(int(cls))
                boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return np.array(labels, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a, b):
    """Pairwise IoU between boxes a [N,4] and b [M,4] in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(ious, threshold):
    """One-to-one greedy matching by descending IoU. Returns list of (pred_i, gt_j, iou)."""
    matches = []
    if ious.size == 0:
        return matches
    ious = ious.copy()
    while True:
        i, j = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[i, j] < threshold:
            break
        matches.append((int(i), int(j), float(ious[i, j])))
        ious[i, :] = -1
        ious[:, j] = -1
    return matches


def evaluate(frames, class_ids, thresholds):
    """Accumulate TP/FP/FN over frames.

    frames: list of (pred_labels, pred_boxes, gt_labels, gt_boxes).
    Returns {threshold: {class_key: {tp, fp, fn, ious: [...]}}} where class_key
    is an int class id, "all" (micro over classes), or "agnostic" (labels ignored).
    """
    results = {t: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ious": []}) for t in thresholds}

    for pl, pb, gl, gb in frames:
        for t in thresholds:
            # Per-class matching (predictions only count within their class)
            for c in class_ids:
                p_idx, g_idx = np.where(pl == c)[0], np.where(gl == c)[0]
                matches = greedy_match(iou_matrix(pb[p_idx], gb[g_idx]), t)
                stats = results[t][c]
                stats["tp"] += len(matches)
                stats["fp"] += len(p_idx) - len(matches)
                stats["fn"] += len(g_idx) - len(matches)
                stats["ious"].extend(m[2] for m in matches)
            # Class-agnostic matching (localization only)
            matches = greedy_match(iou_matrix(pb, gb), t)
            stats = results[t]["agnostic"]
            stats["tp"] += len(matches)
            stats["fp"] += len(pb) - len(matches)
            stats["fn"] += len(gb) - len(matches)
            stats["ious"].extend(m[2] for m in matches)

    for t in thresholds:
        all_stats = results[t]["all"]
        for c in class_ids:
            for k in ("tp", "fp", "fn"):
                all_stats[k] += results[t][c][k]
            all_stats["ious"].extend(results[t][c]["ious"])
    return results


def workload(frames, thresholds):
    """Human-fix workload: classify every annotation at each IoU threshold.

    Predictions and GT are first paired greedily by IoU regardless of threshold
    (class-agnostic, IoU > 0), then each annotation lands in one bucket:
      ok      — matched pair with IoU >= t (usable as-is)
      adjust  — matched pair with IoU < t (right object, box needs resizing/moving)
      add     — unmatched GT box (human must draw it)
      delete  — unmatched prediction (human must remove it)
    A matched pair counts once (one existing annotation to touch or not).
    Returns {threshold: {ok, adjust, add, delete}}.
    """
    counts = {t: {"ok": 0, "adjust": 0, "add": 0, "delete": 0} for t in thresholds}
    for _, pb, _, gb in frames:
        matches = greedy_match(iou_matrix(pb, gb), threshold=1e-9)
        for t in thresholds:
            good = sum(1 for m in matches if m[2] >= t)
            counts[t]["ok"] += good
            counts[t]["adjust"] += len(matches) - good
            counts[t]["add"] += len(gb) - len(matches)
            counts[t]["delete"] += len(pb) - len(matches)
    return counts


def prf(stats):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    miou = float(np.mean(stats["ious"])) if stats["ious"] else 0.0
    return p, r, f1, miou


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", required=True, help="Directory of predicted YOLO label files")
    parser.add_argument("--gt-dir", required=True, help="Directory of ground-truth YOLO label files")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--class-names", type=str, nargs="+", default=None,
                        help="Optional class names indexed by class id")
    parser.add_argument("--json", type=str, default=None, help="Optional path to dump results as JSON")
    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.txt")))
    if not gt_files:
        raise SystemExit(f"No ground-truth label files found in {args.gt_dir}")

    frames = []
    class_ids = set()
    n_missing_pred, n_gt_boxes, n_pred_boxes = 0, 0, 0
    for gt_path in gt_files:
        pred_path = os.path.join(args.pred_dir, os.path.basename(gt_path))
        if not os.path.exists(pred_path):
            n_missing_pred += 1
            pred_path = None
        gl, gb = load_yolo_file(gt_path)
        pl, pb = load_yolo_file(pred_path)
        class_ids.update(gl.tolist())
        class_ids.update(pl.tolist())
        n_gt_boxes += len(gb)
        n_pred_boxes += len(pb)
        frames.append((pl, pb, gl, gb))

    class_ids = sorted(class_ids)
    results = evaluate(frames, class_ids, args.iou_thresholds)

    def class_name(c):
        if args.class_names and c < len(args.class_names):
            return args.class_names[c]
        return f"class {c}"

    print(f"GT frames: {len(gt_files)}  (missing prediction files: {n_missing_pred})")
    print(f"GT boxes: {n_gt_boxes}  predicted boxes: {n_pred_boxes}")
    header = f"{'IoU':>5}  {'class':<20} {'P':>6} {'R':>6} {'F1':>6} {'mIoU':>6} {'TP':>5} {'FP':>5} {'FN':>5}"
    print(header)
    print("-" * len(header))
    summary = {}
    for t in args.iou_thresholds:
        for key in class_ids + ["all", "agnostic"]:
            stats = results[t][key]
            p, r, f1, miou = prf(stats)
            name = class_name(key) if isinstance(key, int) else key
            print(f"{t:>5.2f}  {name:<20} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {miou:>6.3f} "
                  f"{stats['tp']:>5} {stats['fp']:>5} {stats['fn']:>5}")
            summary[f"{t}/{name}"] = {"precision": p, "recall": r, "f1": f1, "mean_iou": miou,
                                      "tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"]}
        print("-" * len(header))

    fix_counts = workload(frames, args.iou_thresholds)
    print("\nHuman-fix workload (annotations needing manual action to reach GT):")
    header2 = (f"{'IoU':>5}  {'total':>6} {'ok':>6} {'adjust':>7} {'add':>5} {'delete':>7} "
               f"{'% needs fixing':>15}")
    print(header2)
    print("-" * len(header2))
    for t in args.iou_thresholds:
        c = fix_counts[t]
        total = c["ok"] + c["adjust"] + c["add"] + c["delete"]
        fix = c["adjust"] + c["add"] + c["delete"]
        pct = 100.0 * fix / total if total else 0.0
        print(f"{t:>5.2f}  {total:>6} {c['ok']:>6} {c['adjust']:>7} {c['add']:>5} {c['delete']:>7} "
              f"{pct:>14.1f}%")
        summary[f"{t}/workload"] = {**c, "total": total,
                                    "pct_needs_fixing": round(pct, 2)}

    if args.json:
        summary["meta"] = {"pred_dir": args.pred_dir, "gt_dir": args.gt_dir,
                           "gt_frames": len(gt_files), "missing_pred_files": n_missing_pred,
                           "gt_boxes": n_gt_boxes, "pred_boxes": n_pred_boxes}
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
