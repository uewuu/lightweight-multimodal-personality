"""Efficiency benchmark for Raw Student and D-DCS.

The timed region covers fusion-and-regression inference from pre-extracted
modality features already resident on the selected device. Feature extraction,
data loading, and host-to-device transfer are excluded. D-DCS timing includes
the five trait-wise affine transformations and clipping to [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from dcs import TraitWiseDCS, load_parameters
from model import (
    DEFAULT_FEATURES,
    DEFAULT_INPUT_DIMS,
    EXPECTED_TRAINABLE_PARAMETERS,
    FinalStudent,
    assert_parameter_count,
)


FEATURES = tuple(DEFAULT_FEATURES)
INPUT_DIMS = tuple(DEFAULT_INPUT_DIMS)
PAPER_BENCHMARK_LENGTHS = {
    "dinov2_face": 450,
    "wavlm": 764,
    "roberta": 56,
    "egemaps_lld": 1500,
}
OFFICIAL_SEED42 = {
    "Raw Student": {"racc": 0.916190, "r2": 0.481810},
    "D-DCS": {"racc": 0.916363, "r2": 0.485638},
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    nested = payload.get("config")
    if isinstance(nested, dict):
        merged = {key: value for key, value in payload.items() if key != "config"}
        merged.update(nested)
        return merged
    return payload


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location="cpu")


def load_student(config: dict[str, Any], checkpoint: Path | None) -> FinalStudent:
    """Instantiate the student and optionally load checkpoint weights."""
    student = FinalStudent(config)
    assert_parameter_count(student)
    if checkpoint is None:
        return student
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    payload = _torch_load(checkpoint)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    else:
        raise TypeError(
            "Checkpoint must be a Lightning checkpoint with state_dict or a raw state_dict mapping"
        )
    student.load_compatible_state_dict(state_dict)
    return student


def make_inputs(
    batch_size: int,
    device: torch.device,
    lengths: dict[str, int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    features: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for name, dim in zip(FEATURES, INPUT_DIMS):
        length = int(lengths[name])
        if length <= 0:
            raise ValueError(f"Invalid benchmark length for {name}: {length}")
        features.append(torch.rand(batch_size, length, dim, device=device))
        masks.append(torch.ones(batch_size, length, dtype=torch.bool, device=device))
    return features, masks


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_output(output: Any, batch_size: int) -> None:
    if not torch.is_tensor(output):
        raise TypeError(f"Expected tensor output, got {type(output)}")
    if tuple(output.shape) != (batch_size, 5):
        raise ValueError(f"Expected output [{batch_size},5], got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise FloatingPointError("Model output contains NaN or Inf")


def measure_latency(
    model: torch.nn.Module,
    features: list[torch.Tensor],
    masks: list[torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(features, masks)
        synchronize(device)
        validate_output(output, int(features[0].shape[0]))

        all_latencies: list[float] = []
        repeat_means: list[float] = []
        for repeat_index in range(repeats):
            values: list[float] = []
            for _ in range(iterations):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = model(features, masks)
                    end.record()
                    end.synchronize()
                    elapsed_ms = float(start.elapsed_time(end))
                else:
                    start_time = time.perf_counter()
                    output = model(features, masks)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                values.append(elapsed_ms)
            validate_output(output, int(features[0].shape[0]))
            repeat_mean = statistics.fmean(values)
            repeat_means.append(repeat_mean)
            all_latencies.extend(values)
            print(
                f"  [repeat {repeat_index + 1}/{repeats}] mean latency={repeat_mean:.6f} ms"
            )

    batch_size = int(features[0].shape[0])
    repeat_throughputs = [batch_size * 1000.0 / value for value in repeat_means]
    return {
        "warmup_iterations": warmup,
        "measurement_iterations_per_repeat": iterations,
        "measurement_repeats": repeats,
        "measurements": len(all_latencies),
        "mean_ms": statistics.fmean(repeat_means),
        "repeat_sample_std_ms": statistics.stdev(repeat_means) if len(repeat_means) >= 2 else 0.0,
        "repeat_mean_ms": repeat_means,
        "throughput_mean_samples_per_second": statistics.fmean(repeat_throughputs),
        "throughput_sample_std_samples_per_second": (
            statistics.stdev(repeat_throughputs) if len(repeat_throughputs) >= 2 else 0.0
        ),
        "repeat_throughput_samples_per_second": repeat_throughputs,
        "median_ms_all_iterations_diagnostic": statistics.median(all_latencies),
        "p95_ms_all_iterations_diagnostic": float(np.percentile(all_latencies, 95)),
    }


def measure_memory(
    model: torch.nn.Module,
    features: list[torch.Tensor],
    masks: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda":
        return {"available": False, "reason": "CUDA is required for peak GPU memory"}
    synchronize(device)
    allocated_before_forward = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = model(features, masks)
    synchronize(device)
    validate_output(output, int(features[0].shape[0]))
    peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "available": True,
        "allocated_before_forward_bytes": allocated_before_forward,
        "peak_allocated_bytes": peak,
        "peak_allocated_mib": peak / (1024.0**2),
        "forward_incremental_peak_mib": max(0, peak - allocated_before_forward) / (1024.0**2),
    }


def benchmark_one(
    name: str,
    factory: Callable[[], torch.nn.Module],
    device: torch.device,
    batch_sizes: list[int],
    lengths: dict[str, int],
    warmup: int,
    iterations: int,
    repeats: int,
    fixed_calibration_scalars: int,
) -> dict[str, Any]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)

    model = factory().eval()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    if total_params != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"{name} parameter mismatch: expected {EXPECTED_TRAINABLE_PARAMETERS:,}, got {total_params:,}"
        )
    model.requires_grad_(False)
    model.to(device)
    synchronize(device)

    result: dict[str, Any] = {
        "name": name,
        "parameters": total_params,
        "fixed_calibration_scalars": fixed_calibration_scalars,
        "batches": {},
    }
    print(f"\n[MODEL] {name}: parameters={total_params:,}")

    for batch_size in batch_sizes:
        features, masks = make_inputs(batch_size, device, lengths)
        print(f"[BENCHMARK] {name}, batch={batch_size}")
        memory = measure_memory(model, features, masks, device)
        latency = measure_latency(model, features, masks, device, warmup, iterations, repeats)
        result["batches"][str(batch_size)] = {
            "batch_size": batch_size,
            "input_shapes": [list(x.shape) for x in features],
            "memory": memory,
            "latency": latency,
        }
        print(
            f"[RESULT] latency={latency['mean_ms']:.6f} ± {latency['repeat_sample_std_ms']:.6f} ms, "
            f"throughput={latency['throughput_mean_samples_per_second']:.2f} ± "
            f"{latency['throughput_sample_std_samples_per_second']:.2f} samples/s"
        )
        del features, masks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)
    return result


def save_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_summary.csv"
    md_path = output_dir / "benchmark_summary.md"

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)

    rows: list[dict[str, Any]] = []
    for model in report["models"]:
        for batch_key, batch in model["batches"].items():
            latency = batch["latency"]
            rows.append(
                {
                    "model": model["name"],
                    "batch_size": int(batch_key),
                    "network_parameters": model["parameters"],
                    "fixed_calibration_scalars": model["fixed_calibration_scalars"],
                    "latency_mean_ms": latency["mean_ms"],
                    "latency_sample_sd_ms": latency["repeat_sample_std_ms"],
                    "throughput_mean_samples_s": latency["throughput_mean_samples_per_second"],
                    "throughput_sample_sd_samples_s": latency[
                        "throughput_sample_std_samples_per_second"
                    ],
                    "peak_gpu_memory_mib": batch["memory"].get("peak_allocated_mib"),
                }
            )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    raw = next(item for item in report["models"] if item["name"] == "Raw Student")
    dcs = next(item for item in report["models"] if item["name"] == "D-DCS")
    lines = [
        "# Student Efficiency Benchmark",
        "",
        "Scope: fusion-and-regression inference from pre-extracted features already on device.",
        "Feature extraction, data loading, and host-to-device transfer are excluded.",
        "D-DCS includes the student forward, five trait-wise affine transforms, and clip[0,1].",
        "",
        "| Model | Batch | Latency mean ± sample SD (ms) | Throughput mean ± sample SD (samples/s) | Peak GPU memory (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in report["models"]:
        for batch_key, batch in model["batches"].items():
            latency = batch["latency"]
            memory = batch["memory"].get("peak_allocated_mib")
            memory_text = "N/A" if memory is None else f"{memory:.3f}"
            lines.append(
                f"| {model['name']} | {batch_key} | "
                f"{latency['mean_ms']:.6f} ± {latency['repeat_sample_std_ms']:.6f} | "
                f"{latency['throughput_mean_samples_per_second']:.2f} ± "
                f"{latency['throughput_sample_std_samples_per_second']:.2f} | {memory_text} |"
            )
    for batch_key in raw["batches"]:
        raw_ms = raw["batches"][batch_key]["latency"]["mean_ms"]
        dcs_ms = dcs["batches"][batch_key]["latency"]["mean_ms"]
        lines.append(
            f"- Batch {batch_key} D-DCS latency overhead vs Raw Student: "
            f"{(dcs_ms / raw_ms - 1.0) * 100.0:.3f}%."
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[SAVED] {json_path.resolve()}")
    print(f"[SAVED] {csv_path.resolve()}")
    print(f"[SAVED] {md_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Raw Student and D-DCS.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional student checkpoint. If omitted, random weights are used for timing only.",
    )
    parser.add_argument(
        "--calibration-parameters",
        type=Path,
        default=Path("calibration_parameters.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value <= 0 for value in args.batch_sizes):
        raise ValueError("All batch sizes must be positive")
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        raise ValueError("warmup, iterations, and repeats must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Use --allow-cpu only for a functional smoke test.")

    config = load_yaml(args.config)
    config["input_feature_dim"] = list(INPUT_DIMS)
    checkpoint = args.checkpoint
    slopes, intercepts, calibration_metadata = load_parameters(args.calibration_parameters)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    def raw_factory() -> torch.nn.Module:
        torch.manual_seed(42)
        return load_student(dict(config), checkpoint)

    def dcs_factory() -> torch.nn.Module:
        torch.manual_seed(42)
        student = load_student(dict(config), checkpoint)
        return TraitWiseDCS(student, slopes, intercepts)

    models = [
        benchmark_one(
            "Raw Student",
            raw_factory,
            device,
            args.batch_sizes,
            PAPER_BENCHMARK_LENGTHS,
            args.warmup,
            args.iterations,
            args.repeats,
            fixed_calibration_scalars=0,
        ),
        benchmark_one(
            "D-DCS",
            dcs_factory,
            device,
            args.batch_sizes,
            PAPER_BENCHMARK_LENGTHS,
            args.warmup,
            args.iterations,
            args.repeats,
            fixed_calibration_scalars=10,
        ),
    ]

    raw_by_batch = models[0]["batches"]
    dcs_by_batch = models[1]["batches"]
    overhead = {
        batch: (
            dcs_by_batch[batch]["latency"]["mean_ms"]
            / raw_by_batch[batch]["latency"]["mean_ms"]
            - 1.0
        )
        * 100.0
        for batch in raw_by_batch
    }
    report = {
        "protocol": {
            "feature_extraction_included": False,
            "data_loading_included": False,
            "host_to_device_transfer_included": False,
            "dtype": "float32",
            "input_lengths": PAPER_BENCHMARK_LENGTHS,
            "input_length_source": "medians archived from the first 64 deterministic validation samples in the paper benchmark",
            "warmup": args.warmup,
            "iterations_per_repeat": args.iterations,
            "measurement_repeats": args.repeats,
            "batch_sizes": args.batch_sizes,
            "dcs_calibration_timed": True,
            "peak_memory_definition": "torch.cuda.max_memory_allocated",
            "checkpoint_loaded": checkpoint is not None,
            "random_weights_note": (
                None
                if checkpoint is not None
                else "No checkpoint supplied: timing uses the exact architecture with deterministic random initialization and is not a reproduction of the paper's archived numeric benchmark."
            ),
        },
        "environment": {
            "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "official_seed42_performance": OFFICIAL_SEED42,
        "calibration": {
            "fit_split": calibration_metadata.get("fit_split"),
            "formula": calibration_metadata.get("formula"),
            "test_labels_used_for_parameter_fitting": calibration_metadata.get(
                "test_labels_used_for_parameter_fitting"
            ),
            "fixed_scalars": 10,
        },
        "models": models,
        "dcs_latency_overhead_vs_raw_percent": overhead,
    }
    save_outputs(args.output_dir, report)
    print("[DONE] Raw Student / D-DCS benchmark completed.")


if __name__ == "__main__":
    main()
