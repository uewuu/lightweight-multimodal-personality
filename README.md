# Factorized Low-Rank Sample-Trait Conditioning for Compact Multimodal Fusion in Apparent Personality Regression

GitHub Public Release v1.0 for the downstream **multimodal fusion-and-regression** experiments associated with the manuscript of the same title.

The released system assumes **pre-extracted** DINOv2-face, WavLM, RoBERTa, and eGeMAPS representations. The compactness and latency claims therefore apply to the downstream fusion/regression stack, not to an end-to-end sensing pipeline that includes the upstream encoders.

## Locked seed-42 endpoints

| Endpoint | Neural params | Avg. RACC | Avg. R² | Avg. PCC |
|---|---:|---:|---:|---:|
| Full Teacher | 5,476,680 | 0.916474 | 0.485587 | 0.697162 |
| D-KD Raw Student | 500,107 | 0.916190 | 0.481810 | 0.695661 |
| D-DCS | 500,107 | 0.916363 | 0.485638 | 0.695661 |
| F: Trait-Interactive student, no KD/no calibration | 500,107 | 0.915532 | 0.472860 | 0.690358 |
| Generic-Joint direct control, no KD/no calibration | 500,875 | 0.915422 | 0.471153 | 0.688460 |

**Complete DCS** is one deployment strategy: Prediction KD is used during student training, then five trait-wise affine mappings are fitted on validation predictions only. DCS is not described here as a new distillation principle, and the calibration step is not treated as an independent new method.

## What is included

- public student/full model code and configurations;
- checkpoint fingerprints (expected filenames, endpoint rules, parameter counts, and SHA-256 hashes) for Full, D-KD, F, and Generic-Joint; binary `.ckpt` files are intentionally not redistributed;
- FI processed-split manifests (5997/1999/1999);
- public-safe seed-42 prediction CSVs containing only `sample_id` and `pred_*` columns;
- validation-fitted D-DCS coefficients and F-Cal diagnostic coefficients;
- target-free teacher prediction NPZ for offline Prediction KD;
- scripts for label preparation, evaluation, calibration, DCS application, source-video-cluster bootstrap, and release verification;
- selected preprocessing scripts for the four final streams;
- a cleaned reference environment snapshot.

## What is intentionally NOT included

This release does **not** redistribute FI/FIv2 raw media, transcripts, annotations/ground-truth labels, processed feature caches, FI/FIv2-derived `.ckpt` binaries, or any prediction CSV containing `target_*` / `GroundTruth` columns. Users must obtain FI/FIv2 through the dataset's authorized access route and comply with its terms.

Checkpoint omission is a conservative redistribution choice rather than a claim that derived checkpoints are legally prohibited. The exact checkpoint fingerprints remain available in `checkpoints/README.md`.

The processed cache used by the paper contains 5997 train, 1999 validation, and 1999 test clips. The exact historical preprocessing failure causes for the five clips absent from the finalized cache were not preserved in the archived logs. The exact included samples are specified by `artifacts/manifests/*.csv`.

## Important row-order note

Do **not** infer prediction alignment from manifest row order. All released prediction files carry explicit `sample_id` values, and all evaluation/bootstrap scripts join by ID. During the release audit, the legacy Full and D-KD trait-separated exports were linked to IDs through a later unified F export whose five ground-truth vectors matched the legacy exports exactly row-by-row; the F sample-ID set also matched the released test manifest exactly. Ground-truth vectors used for that audit are not redistributed.

## Repository layout

```text
.
├── model.py / train.py / fi.py / config.yaml
├── full_model.py / full_config.yaml
├── dcs.py / benchmark.py
├── checkpoints/  # fingerprint/usage README only; no binary weights
├── configs/
├── artifacts/
│   ├── calibration/
│   ├── manifests/
│   ├── metrics/
│   ├── predictions/
│   ├── provenance/
│   └── teacher_predictions/
├── scripts/
├── preprocessing/
├── environment/
├── provenance/
├── FEATURE_EXTRACTION.md
├── LICENSE_SCOPE.md
├── RELEASE_AUDIT.md
└── SHA256SUMS.txt
```

## Environment

Python 3.11 is recommended. A minimal environment can be installed with:

```bash
pip install -r requirements.txt
```

`environment/` contains a broader **reference environment captured on 2026-08-16**. It is not claimed to be an exact frozen copy of every historical training-time package version.

## Data layout

The public downstream loader expects split shards such as:

```text
data/fi_features/cache/split_train/*.pkl
data/fi_features/cache/split_valid/*.pkl
data/fi_features/cache/split_test/*.pkl
```

Each processed sample must provide the four final streams with feature widths 384/768/1024/25 and a local five-dimensional OCEAN target for training/evaluation. See `FEATURE_EXTRACTION.md` and `preprocessing/README.md`.

## Checkpoint fingerprints and local evaluation

Binary FI/FIv2-derived checkpoints are intentionally **not** redistributed in this public release. Exact SHA-256 fingerprints, expected filenames, endpoint-selection rules, and parameter counts are preserved in `checkpoints/README.md` and `artifacts/provenance/checkpoint_metadata.json`.

If you hold a compatible checkpoint locally, evaluate the D-KD student with:

```bash
python train.py \
  --config config.yaml \
  --db-root ./data/fi_features \
  --checkpoint /path/to/student_dkd_seed42_checkpoint_valid_racc.ckpt
```

Equivalent local commands for F and Generic-Joint are documented in `checkpoints/README.md`. If no checkpoint is supplied, the public student training script can train from the corresponding configuration and an authorized local FI/FIv2 feature cache.

The public `train.py` may export local validation/test CSVs containing targets because metrics are computed locally. Before sharing any export publicly, remove labels with:

```bash
python scripts/export_predictions.py \
  --input ./results/trait_interactive_tcms_predkd/test_predictions.csv \
  --output ./results/public_test_predictions.csv
```

## Prepare local labels from a lawfully obtained FI annotation pickle

```bash
python scripts/build_labels_from_fi_annotation.py \
  --annotation /path/to/annotation_test.pkl \
  --manifest artifacts/manifests/test_manifest.csv \
  --output local_private/test_labels.csv
```

The generated label file is for local evaluation only and should not be committed to the public repository.

## Recompute metrics from released prediction artifacts

```bash
python scripts/evaluate_predictions.py \
  --predictions artifacts/predictions/student_dkd_seed42_test_predictions.csv \
  --labels local_private/test_labels.csv
```

## Reproduce D-DCS

Apply the released seed-42 coefficients:

```bash
python scripts/reproduce_dcs.py \
  --predictions artifacts/predictions/student_dkd_seed42_test_predictions.csv \
  --parameters artifacts/calibration/dcs_seed42_parameters.json \
  --output local_private/student_ddcs_test_predictions.csv
```

Or refit the five affine mappings on locally regenerated validation predictions:

```bash
python scripts/fit_calibration.py \
  --predictions local_private/student_dkd_valid_predictions.csv \
  --labels local_private/valid_labels.csv \
  --output local_private/refit_dcs_parameters.json
```

The test labels are never used to fit calibration parameters.

## Source-video-cluster bootstrap

```bash
python scripts/reproduce_cluster_bootstrap.py \
  --baseline artifacts/predictions/full_seed42_test_predictions.csv \
  --candidate artifacts/predictions/student_ddcs_seed42_test_predictions.csv \
  --labels local_private/test_labels.csv \
  --manifest artifacts/manifests/test_manifest.csv \
  --iterations 10000 --seed 42
```

The positive-improvement convention is baseline-minus-candidate for MAE and candidate-minus-baseline for RACC/R²/PCC. The paper interprets intervals that include zero conservatively; they are not equivalence tests. The locked reference output is included as `artifacts/metrics/full_vs_ddcs_source_video_cluster_bootstrap_seed42.json`.


## Direct structural control: Generic-Joint vs factorized multiplicative conditioning

The release includes the formal seed-42 **Generic-Joint Conditioning (Concat+GELU)** control. It keeps the projected modality tokens, sample and trait factors, query/key/value routing, residual fusion, trait-specific heads, and training protocol aligned with F, while replacing the factorized Hadamard interaction with `GELU(concat(sample_factor, trait_factor))`. The public `model.py` exposes this through `concat_tcms_interaction_mode: generic_concat_mlp`; the matched control configuration is `configs/generic_joint_seed42.yaml`.

| Conditioning | Params | Avg. RACC | Avg. R² | Avg. PCC |
|---|---:|---:|---:|---:|
| Generic-Joint (Concat+GELU) | 500,875 | 0.915422 | 0.471153 | 0.688460 |
| F: factorized multiplicative | 500,107 | 0.915532 | 0.472860 | 0.690358 |

The seed-42 point estimates favor F by +0.000110 RACC, +0.001707 R², and +0.001899 PCC. However, all paired source-video cluster-bootstrap 95% intervals include zero. This release therefore treats Generic-Joint as a direct structural control and does **not** claim statistically resolved superiority. The locked artifacts are `artifacts/metrics/direct_structural_control_seed42_metrics.json` and `artifacts/metrics/F_vs_generic_joint_source_video_cluster_bootstrap_seed42.json`.

Recompute the paired bootstrap locally with authorized labels:

```bash
python scripts/reproduce_cluster_bootstrap.py \
  --baseline artifacts/predictions/generic_joint_seed42_test_predictions.csv \
  --candidate artifacts/predictions/ablation_F_seed42_test_predictions.csv \
  --labels local_private/test_labels.csv \
  --manifest artifacts/manifests/test_manifest.csv \
  --iterations 10000 --seed 42
```

## F and the DCS 2×2 diagnostic

`artifacts/metrics/dcs_2x2_attribution_seed42.json` contains F, F-Cal, D-KD, and D-DCS. The diagnostic supports the statement that Prediction KD and validation-only calibration both contribute, with complementary but partially overlapping gains. It does not support a synergy claim.

## Verify the package

```bash
python scripts/verify_release.py
```

The verifier checks SHA-256 hashes, confirms that public prediction CSVs contain no target columns, confirms that the public teacher NPZ contains no ground-truth target array, confirms that no `.ckpt` binaries are present, and validates the 500,107-parameter multiplicative student and 500,875-parameter Generic-Joint control architectures.

## License and third-party data

The MIT license in this repository applies to the authors' released code/documentation. It does not grant rights to FI/FIv2, upstream pretrained models, or third-party dependencies. See `LICENSE_SCOPE.md`.

## Citation

The associated manuscript title is:

**Factorized Low-Rank Sample-Trait Conditioning for Compact Multimodal Fusion in Apparent Personality Regression**

A finalized `CITATION.cff` should be added only after the author list / publication metadata are confirmed; this public release does not invent those fields.
