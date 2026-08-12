"""Validation-fitted trait-wise affine calibration (DCS) for the final student.

Protocol
--------
1. Fit one affine mapping per OCEAN trait on validation predictions only.
2. Freeze the five slopes and five intercepts.
3. Apply the frozen mapping to test/deployment predictions.
4. Clip outputs to [0, 1].

The test split is intentionally loaded only after the calibration parameters have
been fitted on validation data. Test labels are never used for parameter fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


TRAITS = ("O", "C", "E", "A", "N")
METRICS = ("mae", "racc", "r2", "pcc")


def load_prediction_csv(
    path: str | Path,
    require_targets: bool = True,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray | None]:
    """Load raw predictions; targets are required for calibration fitting/evaluation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    pred_columns = [f"pred_{trait}" for trait in TRAITS]
    target_columns = [f"target_{trait}" for trait in TRAITS]
    required = ["sample_id", *pred_columns]
    if require_targets:
        required.extend(target_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")

    sample_ids = frame["sample_id"].astype(str).str.strip().tolist()
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"{path} contains empty sample_id values")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{path} contains duplicate sample_id values")

    predictions = frame[pred_columns].to_numpy(dtype=np.float64)
    targets = (
        frame[target_columns].to_numpy(dtype=np.float64)
        if all(column in frame.columns for column in target_columns)
        else None
    )
    if predictions.ndim != 2 or predictions.shape[1] != 5:
        raise ValueError(f"Expected predictions [N,5], got {predictions.shape}")
    if require_targets and targets is None:
        raise KeyError(f"{path} is missing one or more target_* columns")
    if targets is not None and predictions.shape != targets.shape:
        raise ValueError(
            f"Expected matching [N,5] arrays, got {predictions.shape} and {targets.shape}"
        )
    if not np.isfinite(predictions).all() or (targets is not None and not np.isfinite(targets).all()):
        raise ValueError(f"{path} contains NaN or Inf")
    if np.min(predictions) < 0.0 or np.max(predictions) > 1.0:
        raise ValueError(f"{path} predictions fall outside [0,1]")
    if targets is not None and (np.min(targets) < 0.0 or np.max(targets) > 1.0):
        raise ValueError(f"{path} targets fall outside [0,1]")

    return frame, sample_ids, predictions, targets


def fit_traitwise_affine(
    validation_predictions: np.ndarray,
    validation_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit y ~= a*x+b independently for the five OCEAN traits by OLS."""
    pred = np.asarray(validation_predictions, dtype=np.float64)
    target = np.asarray(validation_targets, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 5:
        raise ValueError(f"Expected matching [N,5] arrays, got {pred.shape} and {target.shape}")

    slopes = np.zeros(5, dtype=np.float64)
    intercepts = np.zeros(5, dtype=np.float64)
    for index, trait in enumerate(TRAITS):
        x = pred[:, index]
        y = target[:, index]
        if float(np.std(x)) <= 0.0:
            raise ValueError(f"Validation predictions are constant for trait {trait}")
        design = np.column_stack([x, np.ones_like(x)])
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        slopes[index] = float(solution[0])
        intercepts[index] = float(solution[1])

    if not np.isfinite(slopes).all() or not np.isfinite(intercepts).all():
        raise ValueError("Calibration parameters contain NaN or Inf")
    return slopes, intercepts


def apply_traitwise_affine(
    predictions: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    """Apply frozen trait-wise affine mappings followed by clip[0,1]."""
    pred = np.asarray(predictions, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64).reshape(-1)
    intercepts = np.asarray(intercepts, dtype=np.float64).reshape(-1)
    if pred.ndim != 2 or pred.shape[1] != 5:
        raise ValueError(f"Expected predictions [N,5], got {pred.shape}")
    if slopes.shape != (5,) or intercepts.shape != (5,):
        raise ValueError("Trait-wise DCS requires five slopes and five intercepts")
    return np.clip(pred * slopes.reshape(1, -1) + intercepts.reshape(1, -1), 0.0, 1.0)


def metric_block(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Return trait-wise and average MAE/RACC/R2/PCC."""
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    mae = np.mean(np.abs(pred - target), axis=0)
    racc = 1.0 - mae

    target_centered = target - target.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    ss_total = np.sum(target_centered**2, axis=0)
    if np.any(ss_total <= 0.0):
        raise ValueError("At least one target trait is constant")
    r2 = 1.0 - np.sum((target - pred) ** 2, axis=0) / ss_total

    denominator = np.sqrt(
        np.sum(target_centered**2, axis=0) * np.sum(pred_centered**2, axis=0)
    )
    if np.any(denominator <= 0.0):
        raise ValueError("At least one prediction or target trait is constant")
    pcc = np.sum(target_centered * pred_centered, axis=0) / denominator

    arrays = {"mae": mae, "racc": racc, "r2": r2, "pcc": pcc}
    return {
        "trait_metrics": {
            trait: {metric: float(arrays[metric][index]) for metric in METRICS}
            for index, trait in enumerate(TRAITS)
        },
        "average": {metric: float(np.mean(arrays[metric])) for metric in METRICS},
    }


def build_parameter_report(
    slopes: np.ndarray,
    intercepts: np.ndarray,
    fit_samples: int | None = None,
    test_samples: int | None = None,
) -> dict[str, Any]:
    return {
        "fit_split": "validation",
        "fit_samples": fit_samples,
        "test_samples": test_samples,
        "method": "ordinary least squares, independently per trait",
        "formula": "clip(a_t * raw_prediction_t + b_t, 0, 1)",
        "clip_range": [0.0, 1.0],
        "parameters": {
            trait: {
                "slope_a": float(slopes[index]),
                "intercept_b": float(intercepts[index]),
            }
            for index, trait in enumerate(TRAITS)
        },
        "test_labels_used_for_parameter_fitting": False,
    }


def save_parameters(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_parameters(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate a DCS calibration_parameters.json file."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)

    if payload.get("fit_split") != "validation":
        raise ValueError("DCS parameters must be fitted on the validation split")
    if payload.get("test_labels_used_for_parameter_fitting") is not False:
        raise ValueError("DCS metadata must confirm that test labels were not used")
    if payload.get("clip_range") != [0.0, 1.0]:
        raise ValueError("DCS clip_range must be [0.0, 1.0]")

    params = payload.get("parameters")
    if not isinstance(params, dict):
        raise TypeError("DCS parameters must be a mapping")
    slopes = np.asarray([float(params[t]["slope_a"]) for t in TRAITS], dtype=np.float64)
    intercepts = np.asarray([float(params[t]["intercept_b"]) for t in TRAITS], dtype=np.float64)
    if not np.isfinite(slopes).all() or not np.isfinite(intercepts).all():
        raise ValueError("DCS parameters contain NaN or Inf")
    return slopes, intercepts, payload


class TraitWiseDCS(nn.Module):
    """Deployment wrapper: Raw Student -> trait-wise affine -> clip[0,1]."""

    def __init__(self, student: nn.Module, slopes: np.ndarray, intercepts: np.ndarray) -> None:
        super().__init__()
        slopes = np.asarray(slopes, dtype=np.float32).reshape(-1)
        intercepts = np.asarray(intercepts, dtype=np.float32).reshape(-1)
        if slopes.shape != (5,) or intercepts.shape != (5,):
            raise ValueError("Trait-wise DCS requires five slopes and five intercepts")
        self.student = student
        self.register_buffer("dcs_slopes", torch.from_numpy(slopes.copy()))
        self.register_buffer("dcs_intercepts", torch.from_numpy(intercepts.copy()))

    def forward(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        raw_prediction = self.student(features, masks)
        return torch.clamp(
            raw_prediction * self.dcs_slopes + self.dcs_intercepts,
            min=0.0,
            max=1.0,
        )


def _add_calibrated_columns(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    for index, trait in enumerate(TRAITS):
        output[f"calibrated_pred_{trait}"] = values[:, index]
    return output


def fit_command(valid_csv: Path, test_csv: Path, output_dir: Path) -> None:
    # Protocol-critical ordering: fit and freeze validation parameters first.
    valid_frame, valid_ids, valid_pred, valid_target = load_prediction_csv(valid_csv, require_targets=True)
    assert valid_target is not None
    slopes, intercepts = fit_traitwise_affine(valid_pred, valid_target)
    frozen_slopes = slopes.copy()
    frozen_intercepts = intercepts.copy()
    valid_calibrated = apply_traitwise_affine(valid_pred, frozen_slopes, frozen_intercepts)

    # Only now load test data. Test labels are evaluation-only.
    test_frame, test_ids, test_pred, test_target = load_prediction_csv(test_csv, require_targets=True)
    assert test_target is not None
    overlap = sorted(set(valid_ids).intersection(test_ids))
    if overlap:
        raise ValueError(
            f"Validation/test sample_id leakage detected: count={len(overlap)}, examples={overlap[:5]}"
        )
    test_calibrated = apply_traitwise_affine(test_pred, frozen_slopes, frozen_intercepts)

    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_report = build_parameter_report(
        frozen_slopes,
        frozen_intercepts,
        fit_samples=len(valid_ids),
        test_samples=len(test_ids),
    )
    save_parameters(output_dir / "calibration_parameters.json", parameter_report)

    _add_calibrated_columns(valid_frame, valid_calibrated).to_csv(
        output_dir / "valid_predictions_calibrated.csv", index=False, encoding="utf-8-sig"
    )
    _add_calibrated_columns(test_frame, test_calibrated).to_csv(
        output_dir / "test_predictions_calibrated.csv", index=False, encoding="utf-8-sig"
    )

    metrics = {
        "protocol": {
            "fit_split": "validation",
            "evaluation_split": "test",
            "test_labels_used_for_parameter_fitting": False,
            "formula": "clip(a_t * raw_prediction_t + b_t, 0, 1)",
        },
        "validation": {
            "raw": metric_block(valid_pred, valid_target),
            "calibrated": metric_block(valid_calibrated, valid_target),
        },
        "test": {
            "raw": metric_block(test_pred, test_target),
            "calibrated": metric_block(test_calibrated, test_target),
        },
    }
    with (output_dir / "calibration_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2, allow_nan=False)

    print("[DCS] Validation-fitted trait-wise affine parameters")
    for index, trait in enumerate(TRAITS):
        print(f"  {trait}: a={frozen_slopes[index]:.9f}, b={frozen_intercepts[index]:.9f}")
    before = metrics["test"]["raw"]["average"]
    after = metrics["test"]["calibrated"]["average"]
    print(
        f"[TEST] Raw RACC={before['racc']:.9f}, R2={before['r2']:.9f} | "
        f"D-DCS RACC={after['racc']:.9f}, R2={after['r2']:.9f}"
    )
    print("[DONE] Test labels were never used to fit calibration parameters.")


def apply_command(input_csv: Path, parameters: Path, output_csv: Path) -> None:
    frame, _, predictions, _ = load_prediction_csv(input_csv, require_targets=False)
    slopes, intercepts, _ = load_parameters(parameters)
    calibrated = apply_traitwise_affine(predictions, slopes, intercepts)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _add_calibrated_columns(frame, calibrated).to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )
    print(f"[SAVED] {output_csv.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-fitted trait-wise DCS calibration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit on validation and apply to test.")
    fit_parser.add_argument("--valid-csv", type=Path, required=True)
    fit_parser.add_argument("--test-csv", type=Path, required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)

    apply_parser = subparsers.add_parser("apply", help="Apply frozen DCS parameters.")
    apply_parser.add_argument("--input-csv", type=Path, required=True)
    apply_parser.add_argument("--parameters", type=Path, required=True)
    apply_parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        fit_command(args.valid_csv, args.test_csv, args.output_dir)
    else:
        apply_command(args.input_csv, args.parameters, args.output_csv)


if __name__ == "__main__":
    main()
