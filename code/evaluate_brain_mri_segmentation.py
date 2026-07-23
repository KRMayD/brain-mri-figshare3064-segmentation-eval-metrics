#!/usr/bin/env python3
"""Evaluate predicted binary masks against Brain MRI GT masks with DSC and NSD."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = BASE_DIR / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from SurfaceDice import compute_dice_coefficient, compute_surface_dice_at_tolerance, compute_surface_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--tolerance-mm", type=float, default=2.0)
    parser.add_argument("--spacing-mm", type=float, nargs="+", default=(1.0, 1.0, 1.0))
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing, extra, or unreadable prediction files.",
    )
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 127


def safe_nsd(gt: np.ndarray, pred: np.ndarray, spacing_mm, tolerance_mm: float) -> float:
    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0
    if gt.ndim == 2:
        gt = gt[None, :, :]
    if pred.ndim == 2:
        pred = pred[None, :, :]
    spacing = tuple(spacing_mm)
    if len(spacing) == 2:
        spacing = (1.0, spacing[0], spacing[1])
    surface_distances = compute_surface_distances(gt, pred, spacing_mm=spacing)
    return float(compute_surface_dice_at_tolerance(surface_distances, tolerance_mm))


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    gt_paths = sorted(path for path in gt_dir.iterdir() if path.is_file())
    pred_paths = sorted(path for path in pred_dir.iterdir() if path.is_file())
    if args.expected_count is not None and len(gt_paths) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} GT masks, found {len(gt_paths)} in {gt_dir}"
        )

    gt_names = {path.name for path in gt_paths}
    pred_names = {path.name for path in pred_paths}
    if args.strict and gt_names != pred_names:
        missing = sorted(gt_names - pred_names)
        extra = sorted(pred_names - gt_names)
        raise RuntimeError(
            "GT/prediction filename mismatch: "
            f"missing={missing[:10]} ({len(missing)} total), "
            f"extra={extra[:10]} ({len(extra)} total)"
        )

    rows = []
    for gt_path in gt_paths:
        pred_path = pred_dir / gt_path.name
        gt = load_mask(gt_path)
        pred = load_mask(pred_path)
        if gt is None:
            raise RuntimeError(f"Could not read GT mask: {gt_path}")
        if pred is None:
            if args.strict:
                raise RuntimeError(f"Could not read prediction mask: {pred_path}")
            rows.append({
                "sample": gt_path.name,
                "dsc": 0.0,
                "nsd": 0.0,
                "gt_area": int(gt.sum()),
                "pred_area": 0,
                "missing_pred": True,
            })
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        dsc = float(compute_dice_coefficient(gt, pred))
        nsd = safe_nsd(gt, pred, spacing_mm=tuple(args.spacing_mm), tolerance_mm=args.tolerance_mm)
        rows.append({
            "sample": gt_path.name,
            "dsc": dsc,
            "nsd": nsd,
            "gt_area": int(gt.sum()),
            "pred_area": int(pred.sum()),
            "missing_pred": False,
        })

    if args.expected_count is not None and len(rows) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} evaluated samples, got {len(rows)}"
        )

    dsc_values = [row["dsc"] for row in rows]
    nsd_values = [row["nsd"] for row in rows]
    payload = {
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "num_samples": len(rows),
        "num_missing_pred": int(sum(bool(row["missing_pred"]) for row in rows)),
        "mean_dsc": float(np.mean(dsc_values)) if dsc_values else None,
        "std_dsc": float(np.std(dsc_values)) if dsc_values else None,
        "mean_nsd": float(np.mean(nsd_values)) if nsd_values else None,
        "std_nsd": float(np.std(nsd_values)) if nsd_values else None,
        "tolerance_mm": args.tolerance_mm,
        "spacing_mm": list(args.spacing_mm),
        "strict": bool(args.strict),
        "expected_count": args.expected_count,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["sample", "dsc", "nsd", "gt_area", "pred_area", "missing_pred"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
