"""Feature-level efficiency benchmark for Full, Raw Student, and D-DCS.

The timed region starts from pre-extracted modality representations already on
one device. Upstream feature extraction, data loading, and host-to-device
transfer are excluded. D-DCS timing includes trait-wise affine calibration.
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
from model import DEFAULT_FEATURES, DEFAULT_INPUT_DIMS, EXPECTED_TRAINABLE_PARAMETERS, FinalStudent, assert_parameter_count

FEATURES = tuple(DEFAULT_FEATURES)
INPUT_DIMS = tuple(DEFAULT_INPUT_DIMS)
PAPER_BENCHMARK_LENGTHS = {"dinov2_face": 450, "wavlm": 764, "roberta": 56, "egemaps_lld": 1500}
OFFICIAL_SEED42 = {
    "Full": {"racc": 0.916474, "r2": 0.485587, "pcc": 0.697162},
    "Raw Student": {"racc": 0.916190, "r2": 0.481810, "pcc": 0.695661},
    "D-DCS": {"racc": 0.916363, "r2": 0.485638, "pcc": 0.695661},
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    nested = payload.get("config")
    if isinstance(nested, dict):
        merged = {k: v for k, v in payload.items() if k != "config"}
        merged.update(nested)
        return merged
    return payload


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_student(config: dict[str, Any], checkpoint: Path | None) -> FinalStudent:
    student = FinalStudent(config)
    assert_parameter_count(student)
    if checkpoint is None:
        return student
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = _torch_load(checkpoint)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise TypeError("Student checkpoint must contain a state_dict mapping")
    student.load_compatible_state_dict(state_dict)
    return student


def load_full(config: dict[str, Any], checkpoint: Path | None):
    from full_model import FullModel, assert_full_parameter_count, load_lightning_checkpoint
    model = FullModel(config)
    assert_full_parameter_count(model)
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        load_lightning_checkpoint(model, checkpoint, map_location="cpu")
    return model


def make_inputs(batch_size: int, device: torch.device, lengths: dict[str, int]):
    features, masks = [], []
    for name, dim in zip(FEATURES, INPUT_DIMS):
        length = int(lengths[name])
        features.append(torch.rand(batch_size, length, dim, device=device))
        masks.append(torch.ones(batch_size, length, dtype=torch.bool, device=device))
    return features, masks


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_output(output: Any, batch_size: int) -> None:
    if not torch.is_tensor(output) or tuple(output.shape) != (batch_size, 5):
        raise ValueError(f"Expected tensor output [{batch_size},5], got {type(output)} / {getattr(output, 'shape', None)}")
    if not torch.isfinite(output).all():
        raise FloatingPointError("Model output contains NaN or Inf")


def measure_latency(model, features, masks, device, warmup, iterations, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(features, masks)
        synchronize(device)
        validate_output(output, int(features[0].shape[0]))
        all_latencies, repeat_means = [], []
        for repeat_index in range(repeats):
            values = []
            for _ in range(iterations):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                    start.record(); output = model(features, masks); end.record(); end.synchronize()
                    elapsed_ms = float(start.elapsed_time(end))
                else:
                    start_time = time.perf_counter(); output = model(features, masks)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                values.append(elapsed_ms)
            validate_output(output, int(features[0].shape[0]))
            repeat_mean = statistics.fmean(values)
            repeat_means.append(repeat_mean); all_latencies.extend(values)
            print(f"  [repeat {repeat_index + 1}/{repeats}] mean latency={repeat_mean:.6f} ms")
    batch_size = int(features[0].shape[0])
    repeat_throughputs = [batch_size * 1000.0 / value for value in repeat_means]
    return {
        "mean_ms": statistics.fmean(repeat_means),
        "repeat_sample_std_ms": statistics.stdev(repeat_means) if len(repeat_means) >= 2 else 0.0,
        "throughput_mean_samples_per_second": statistics.fmean(repeat_throughputs),
        "throughput_sample_std_samples_per_second": statistics.stdev(repeat_throughputs) if len(repeat_throughputs) >= 2 else 0.0,
        "warmup_iterations": warmup, "measurement_iterations_per_repeat": iterations, "measurement_repeats": repeats,
        "median_ms_all_iterations_diagnostic": statistics.median(all_latencies),
        "p95_ms_all_iterations_diagnostic": float(np.percentile(all_latencies, 95)),
    }


def measure_memory(model, features, masks, device):
    if device.type != "cuda":
        return {"available": False, "reason": "CUDA is required for peak GPU memory"}
    synchronize(device); before = int(torch.cuda.memory_allocated(device)); torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = model(features, masks)
    synchronize(device); validate_output(output, int(features[0].shape[0])); peak = int(torch.cuda.max_memory_allocated(device))
    return {"available": True, "peak_allocated_mib": peak / (1024.0**2), "forward_incremental_peak_mib": max(0, peak-before)/(1024.0**2)}


def benchmark_one(name: str, factory: Callable[[], torch.nn.Module], expected_parameters: int, device: torch.device,
                  batch_sizes: list[int], warmup: int, iterations: int, repeats: int, fixed_calibration_scalars: int):
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache(); synchronize(device)
    model = factory().eval()
    total_params = sum(p.numel() for p in model.parameters())
    if total_params != expected_parameters:
        raise ValueError(f"{name} parameter mismatch: expected {expected_parameters:,}, got {total_params:,}")
    model.requires_grad_(False).to(device); synchronize(device)
    result = {"name": name, "parameters": total_params, "fixed_calibration_scalars": fixed_calibration_scalars, "batches": {}}
    print(f"\n[MODEL] {name}: parameters={total_params:,}")
    for batch_size in batch_sizes:
        features, masks = make_inputs(batch_size, device, PAPER_BENCHMARK_LENGTHS)
        print(f"[BENCHMARK] {name}, batch={batch_size}")
        result["batches"][str(batch_size)] = {"memory": measure_memory(model, features, masks, device),
                                             "latency": measure_latency(model, features, masks, device, warmup, iterations, repeats)}
        del features, masks; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    return result


def save_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for model in report["models"]:
        for batch_key, batch in model["batches"].items():
            lat = batch["latency"]
            rows.append({"model": model["name"], "batch_size": int(batch_key), "network_parameters": model["parameters"],
                         "fixed_calibration_scalars": model["fixed_calibration_scalars"], "latency_mean_ms": lat["mean_ms"],
                         "latency_sample_sd_ms": lat["repeat_sample_std_ms"], "throughput_mean_samples_s": lat["throughput_mean_samples_per_second"],
                         "throughput_sample_sd_samples_s": lat["throughput_sample_std_samples_per_second"],
                         "peak_gpu_memory_mib": batch["memory"].get("peak_allocated_mib")})
    with (output_dir / "benchmark_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    print(f"[SAVED] {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Full, Raw Student, and D-DCS.")
    p.add_argument("--config", type=Path, default=Path("config.yaml")); p.add_argument("--full-config", type=Path, default=Path("full_config.yaml"))
    p.add_argument("--checkpoint", type=Path, default=None); p.add_argument("--full-checkpoint", type=Path, default=None)
    p.add_argument("--calibration-parameters", type=Path, default=Path("calibration_parameters.json")); p.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-sizes", nargs="+", type=int, default=[1,8])
    p.add_argument("--warmup", type=int, default=50); p.add_argument("--iterations", type=int, default=200); p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--allow-cpu", action="store_true"); return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Use --allow-cpu only for a functional smoke test.")
    student_config = load_yaml(args.config); student_config["input_feature_dim"] = list(INPUT_DIMS)
    full_config = load_yaml(args.full_config); full_config["input_feature_dim"] = list(INPUT_DIMS)
    slopes, intercepts, calibration_metadata = load_parameters(args.calibration_parameters)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); torch.set_float32_matmul_precision("medium")
    if device.type == "cuda": torch.cuda.manual_seed_all(42); torch.backends.cudnn.benchmark = True
    def full_factory(): torch.manual_seed(42); return load_full(dict(full_config), args.full_checkpoint)
    def raw_factory(): torch.manual_seed(42); return load_student(dict(student_config), args.checkpoint)
    def dcs_factory(): torch.manual_seed(42); return TraitWiseDCS(load_student(dict(student_config), args.checkpoint), slopes, intercepts)
    from full_model import EXPECTED_FULL_TRAINABLE_PARAMETERS
    models = [
        benchmark_one("Full", full_factory, EXPECTED_FULL_TRAINABLE_PARAMETERS, device, args.batch_sizes, args.warmup, args.iterations, args.repeats, 0),
        benchmark_one("Raw Student", raw_factory, EXPECTED_TRAINABLE_PARAMETERS, device, args.batch_sizes, args.warmup, args.iterations, args.repeats, 0),
        benchmark_one("D-DCS", dcs_factory, EXPECTED_TRAINABLE_PARAMETERS, device, args.batch_sizes, args.warmup, args.iterations, args.repeats, 10),
    ]
    full_b, raw_b, dcs_b = [m["batches"] for m in models]
    report = {
        "protocol": {"feature_extraction_included": False, "data_loading_included": False, "host_to_device_transfer_included": False,
                     "dtype": "float32", "input_lengths": PAPER_BENCHMARK_LENGTHS, "warmup": args.warmup, "iterations_per_repeat": args.iterations,
                     "measurement_repeats": args.repeats, "batch_sizes": args.batch_sizes, "dcs_calibration_timed": True,
                     "random_weights_note": "Omitted checkpoints use deterministic random weights for architecture/timing smoke tests only."},
        "environment": {"timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": str(device),
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"},
        "official_seed42_performance": OFFICIAL_SEED42,
        "calibration": {"fit_split": calibration_metadata.get("fit_split"), "formula": calibration_metadata.get("formula"),
                        "test_labels_used_for_parameter_fitting": calibration_metadata.get("test_labels_used_for_parameter_fitting"), "fixed_scalars": 10},
        "models": models,
        "ddcs_speedup_vs_full": {b: full_b[b]["latency"]["mean_ms"] / dcs_b[b]["latency"]["mean_ms"] for b in full_b},
        "dcs_latency_overhead_vs_raw_percent": {b: (dcs_b[b]["latency"]["mean_ms"] / raw_b[b]["latency"]["mean_ms"] - 1.0)*100 for b in raw_b},
    }
    save_outputs(args.output_dir, report); print("[DONE] Full / Raw Student / D-DCS benchmark completed.")


if __name__ == "__main__":
    main()
