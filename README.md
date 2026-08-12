# Efficient Multimodal Apparent Personality Recognition

Code for the lightweight multimodal personality recognition model described in:

**Efficient Multimodal Apparent Personality Recognition with Trait-Interactive Fusion and Validation-Fitted Calibration**

The repository covers the **fusion-and-regression stage** and assumes that modality features have already been extracted.

## Model

Four modality representations are used:

| Modality | Representation | Dim. |
|---|---|---:|
| Visual | DINOv2-face | 384 |
| Speech | WavLM | 768 |
| Text | RoBERTa | 1024 |
| Acoustic descriptors | eGeMAPS LLD | 25 |

The student architecture is:

```text
modality sequences
    -> mask-aware temporal pooling
    -> modality-specific 96-d projections
    -> projected concat + MLP
    -> Trait-Interactive TCMS-Lite
    -> trait-specific regression heads
    -> O/C/E/A/N predictions
```

Trait-Interactive TCMS-Lite performs sample- and trait-conditioned modality routing. Training uses offline Prediction KD together with the regression and regularization terms described in the paper.

D-DCS applies one validation-fitted affine transformation to each trait:

```text
y'_t = clip(a_t * y_t + b_t, 0, 1)
```

The ten calibration scalars are fitted on validation predictions and then fixed for test and deployment inference.

## Results

Seed-42 results reported in the paper:

| System | Fusion/regression params | Fixed calibration scalars | Avg. RACC | Avg. R² |
|---|---:|---:|---:|---:|
| Full Teacher | 5,476,680 | 0 | 0.916474 | 0.485587 |
| Raw Student | 500,107 | 0 | 0.916190 | 0.481810 |
| D-DCS | 500,107 | 10 | 0.916363 | 0.485638 |

Parameter counts refer to the fusion-and-regression network. Pretrained feature extractors are not included.

Efficiency measurements were obtained on an NVIDIA RTX A4000 with FP32 inputs. Pre-extracted features were already on the GPU, and feature extraction/data loading were excluded from timing.

| System | Batch-1 latency (ms) | Batch-8 throughput (samples/s) | Peak GPU memory, B=8 (MiB) |
|---|---:|---:|---:|
| Full Teacher | 92.285 ± 1.328 | 81.51 ± 0.66 | 245.964 |
| Raw Student | 1.885 ± 0.098 | 4,254.93 ± 33.04 | 54.383 |
| D-DCS | 1.952 ± 0.076 | 4,099.77 ± 84.82 | 54.384 |

`benchmark.py` measures Raw Student and D-DCS. The Full Teacher numbers above are included for reference.

## Files

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── model.py
├── train.py
├── fi.py
├── dcs.py
├── benchmark.py
├── config.yaml
├── calibration_parameters.json
└── tools/
    └── check_feature_helpers.py
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build that matches the local CUDA environment. The efficiency measurements in the paper used PyTorch 2.5.1+cu118 and CUDA 11.8.

## Data layout

The ChaLearn First Impressions dataset and extracted features are not included.

Expected layout:

```text
data/fi_features/
├── cache/
│   ├── split_train/*.pkl
│   ├── split_valid/*.pkl
│   └── split_test/*.pkl
└── standardization/
    └── egemaps_lld.npz
```

Each sample should contain:

```python
{
    "dinov2_face": ...,   # [T, 384]
    "wavlm": ...,         # [T, 768]
    "roberta": ...,       # [T, 1024]
    "egemaps_lld": ...,   # [T, 25]
    "ocean": ...,         # [5], O/C/E/A/N
}
```

The loader also supports `cache/fi_train.pkl`, `cache/fi_valid.pkl`, and `cache/fi_test.pkl`.

## Prediction KD

Offline teacher predictions are stored in an NPZ file containing:

```text
sample_ids
predictions
```

`predictions` must have shape `[N, 5]` in O/C/E/A/N order.

The default path in `config.yaml` is:

```text
./data/teacher_predictions/full_teacher_train_predictions.npz
```

The Teacher is required only to generate the distillation targets; it is not used by the deployed student.

## Training

```bash
python train.py   --config config.yaml   --db-root ./data/fi_features   --teacher-predictions ./data/teacher_predictions/full_teacher_train_predictions.npz
```

Training saves validation-RACC and validation-R² checkpoints. Early stopping monitors validation R², while final model selection uses validation RACC.

Evaluate a saved checkpoint with:

```bash
python train.py   --config config.yaml   --db-root ./data/fi_features   --checkpoint ./checkpoints/checkpoint_valid_racc.ckpt
```

## D-DCS

Apply the supplied seed-42 calibration parameters:

```bash
python dcs.py apply   --input-csv raw_predictions.csv   --parameters calibration_parameters.json   --output-csv calibrated_predictions.csv
```

For a newly trained student, refit D-DCS using that model's validation predictions:

```bash
python dcs.py fit   --valid-csv valid_predictions.csv   --test-csv test_predictions.csv   --output-dir dcs_results
```

The fitting routine estimates all calibration parameters from the validation split before loading the test file for evaluation.

## Efficiency benchmark

```bash
python benchmark.py   --config config.yaml   --checkpoint ./checkpoints/checkpoint_valid_racc.ckpt   --calibration-parameters calibration_parameters.json   --device cuda:0
```

Default settings match the paper protocol for Raw Student and D-DCS: batch sizes 1 and 8, 50 warm-up iterations, 200 timed iterations per repeat, and 3 repeats.

A CPU smoke test can be run with:

```bash
python benchmark.py --allow-cpu --warmup 2 --iterations 5 --repeats 1
```

Without a checkpoint, the benchmark uses deterministic random weights for a functional timing check; those values should not be compared with the paper results.

## Notes

- Student trainable parameters: **500,107**.
- D-DCS adds **10 fixed non-trainable scalars**.
- Compression results refer to the fusion-and-regression stage rather than raw-media feature extraction.
- `calibration_parameters.json` contains the validation-fitted parameters for the reported seed-42 model.
- `tools/check_feature_helpers.py` can be used to compare local normalization/padding behavior with `exordium`.

## Citation

Citation information will be added after publication.

## License

MIT License. See `LICENSE`.
