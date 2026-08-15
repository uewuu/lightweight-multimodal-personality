# High-Capacity and Lightweight Multimodal Learning for Apparent Personality Recognition

Minimal public code release for the paper:

**High-Capacity and Lightweight Multimodal Learning for Apparent Personality Recognition via Trait-Conditioned Routing and Distillation–Calibration**

This repository covers the downstream **fusion-and-regression stage** and assumes DINOv2-face, WavLM, RoBERTa, and eGeMAPS representations are pre-extracted.

## Included methods

- **Full**: LinMulT -> Modality Token Fusion (MTF) with Behavior-Reliability Summary Token (BRST) -> trait-conditioned modality selection (TCMS) -> trait-specific residual regression. B-ARCL and Agreeableness-aware contrastive learning are training-only objectives.
- **D-DCS**: mask-aware pooling -> 96-D modality projections -> Projected Concat+MLP -> Trait-Interactive TCMS-Lite -> trait-specific residual regression. Prediction KD is training-only; validation-fitted trait-wise affine calibration completes DCS.

DCS uses `y'_t = clip(a_t * y_t + b_t, 0, 1)`, fitted independently for the five OCEAN traits on validation predictions only.

## Paper results (seed 42)

| System | Neural params | Fixed affine scalars | Avg. RACC | Avg. R² | Avg. PCC |
|---|---:|---:|---:|---:|---:|
| Full | 5,476,680 | 0 | 0.916474 | 0.485587 | 0.697162 |
| Raw Student | 500,107 | 0 | 0.916190 | 0.481810 | 0.695661 |
| D-DCS | 500,107 | 10 | 0.916363 | 0.485638 | 0.695661 |

Feature-level efficiency (RTX A4000, FP32; pre-extracted representations already on GPU):

| System | Batch-1 latency (ms) | Batch-8 throughput (samples/s) | Peak GPU memory, B=8 (MiB) |
|---|---:|---:|---:|
| Full | 92.285 ± 1.328 | 81.51 ± 0.66 | 245.964 |
| Raw Student | 1.885 ± 0.098 | 4,254.93 ± 33.04 | 54.383 |
| D-DCS | 1.952 ± 0.076 | 4,099.77 ± 84.82 | 54.384 |

D-DCS therefore provides 10.95x neural compression and a 47.29x batch-1 feature-level speed-up relative to Full. Upstream feature extraction and data loading are excluded from this timing boundary.

## Files

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── fi.py
├── model.py
├── train.py
├── config.yaml
├── dcs.py
├── calibration_parameters.json
├── full_model.py
├── full_config.yaml
├── benchmark.py
└── tools/
    └── check_feature_helpers.py
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
```

`linmult==1.5.2` is the external LinMulT dependency. `exordium` supplies the FI normalization/padding helpers. Install the PyTorch build appropriate for the local CUDA environment when using a GPU.

## Data layout

FIv2 and pre-extracted features are not redistributed. Expected layout:

```text
data/fi_features/
├── cache/
│   ├── split_train/*.pkl
│   ├── split_valid/*.pkl
│   └── split_test/*.pkl
└── standardization/
    └── egemaps_lld.npz
```

Each sample contains `dinov2_face [T,384]`, `wavlm [T,768]`, `roberta [T,1024]`, `egemaps_lld [T,25]`, and `ocean [5]`. OpenGraphAU statistics are loaded only when OpenGraphAU is explicitly requested; they are not needed for the paper's four-stream configuration.

## Prediction KD

Offline teacher predictions are an NPZ with `sample_ids` and `predictions` (`[N,5]`, O/C/E/A/N). The teacher is not used by the deployed student.

## Train / evaluate the student

```bash
python train.py --config config.yaml --db-root ./data/fi_features --teacher-predictions ./data/teacher_predictions/full_teacher_train_predictions.npz
```

Evaluate an existing selected checkpoint:

```bash
python train.py --config config.yaml --db-root ./data/fi_features --checkpoint ./checkpoints/checkpoint_valid_racc.ckpt
```

The selected-checkpoint evaluation writes both `valid_predictions.csv` and `test_predictions.csv`. Early stopping monitors validation R²; the reported final student uses the validation-RACC checkpoint.

## D-DCS

Apply the supplied fixed seed-42 parameters:

```bash
python dcs.py apply --input-csv raw_predictions.csv --parameters calibration_parameters.json --output-csv calibrated_predictions.csv
```

Refit on a newly trained student's validation predictions:

```bash
python dcs.py fit --valid-csv ./results/trait_interactive_tcms_predkd/valid_predictions.csv --test-csv ./results/trait_interactive_tcms_predkd/test_predictions.csv --output-dir dcs_results
```

All affine parameters are fitted from validation only; test labels are not used for fitting.

## Full model

`full_model.py` is the minimal public implementation of the locked Full path. Historical ablation branches and experiment-management plumbing are omitted. The LinMulT backbone remains an external dependency, while the hidden-state interface used by the formal Full run is included in `full_model.py`.

Parameter check:

```bash
python -c "import yaml; from full_model import FullModel, assert_full_parameter_count; c=yaml.safe_load(open('full_config.yaml', encoding='utf-8')); m=FullModel(c); print(assert_full_parameter_count(m))"
```

Expected output: `5476680`.

## Efficiency benchmark

The benchmark measures Full, Raw Student, and D-DCS under the same feature-level timing boundary:

```bash
python benchmark.py --config config.yaml --full-config full_config.yaml --checkpoint ./checkpoints/checkpoint_valid_racc.ckpt --full-checkpoint ./checkpoints/full_checkpoint_valid_r2.ckpt --calibration-parameters calibration_parameters.json --device cuda:0
```

If a checkpoint is omitted, that architecture uses deterministic random weights for a functional/timing smoke test only. CPU smoke test:

```bash
python benchmark.py --allow-cpu --warmup 2 --iterations 5 --repeats 1
```

Default benchmark settings are batch sizes 1/8, 50 warm-up iterations, 200 timed iterations per repeat, and 3 repeats.

## Notes

- Final student: **500,107** trainable neural parameters.
- Full: **5,476,680** trainable neural parameters.
- D-DCS adds **10 fixed non-trainable affine scalars**.
- `calibration_parameters.json` contains the reported seed-42 validation-fitted parameters.
- FIv2, pre-extracted features, checkpoints, and offline teacher prediction targets are not redistributed.

## Citation

Citation information will be added after publication.

## License

MIT License. See `LICENSE`.
