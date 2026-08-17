# Checkpoint fingerprints (binary weights not redistributed)

The public GitHub release intentionally omits all FI/FIv2-derived `.ckpt` binaries. This is a conservative redistribution choice because the applicable FI/FIv2 access terms do not explicitly establish permission to redistribute derived model checkpoints. This statement does not assert that redistribution is prohibited.

Exact endpoint fingerprints are retained below so privately retained or independently regenerated checkpoints can be verified byte-for-byte.

| Endpoint | Expected filename | Selection endpoint | Neural params | SHA-256 |
|---|---|---|---:|---|
| Full Teacher | `full_seed42_checkpoint_valid_r2.ckpt` | valid-R² | 5,476,680 | `7771bf086cf6e740e89bea7ad83af164e00df3fc777104659ae6539edbe8edc5` |
| D-KD Raw Student | `student_dkd_seed42_checkpoint_valid_racc.ckpt` | valid-RACC | 500,107 | `7f34cfe4ab8d4961eaea0ccaa2c1239670a5c57ad152025c648d664f6e655346` |
| F: multiplicative, no KD/calibration | `ablation_F_seed42_checkpoint_valid_racc.ckpt` | valid-RACC | 500,107 | `d41315fd8b3297cc10822d599a3438dd7beab8ba27166cb20f4e0851254bb48a` |
| Generic-Joint direct control | `generic_joint_seed42_checkpoint_valid_racc.ckpt` | valid-RACC | 500,875 | `ecac4d7ede8f9c78e28be3dc2349d81f4a19b849653f7dfcca81e79c77716cce` |

## Local evaluation with a privately retained checkpoint

For the D-KD student:

```bash
python train.py \
  --config config.yaml \
  --db-root ./data/fi_features \
  --checkpoint /path/to/student_dkd_seed42_checkpoint_valid_racc.ckpt
```

For F (multiplicative, no KD/calibration):

```bash
python train.py \
  --config configs/ablation_F_seed42.yaml \
  --db-root ./data/fi_features \
  --checkpoint /path/to/ablation_F_seed42_checkpoint_valid_racc.ckpt
```

For the Generic-Joint control:

```bash
python train.py \
  --config configs/generic_joint_seed42.yaml \
  --db-root ./data/fi_features \
  --checkpoint /path/to/generic_joint_seed42_checkpoint_valid_racc.ckpt
```

If no local checkpoint is supplied, the student training script can train from the corresponding public configuration and authorized local FI/FIv2 feature cache. The released prediction/metric artifacts remain sufficient to recompute the paper's public evidence without distributing model weights.

For the Full Teacher, `full_model.py`, `full_config.yaml`, the sanitized formal configuration, and the archived training source are retained for architectural/provenance reconstruction. The public release does not claim a one-command byte-identical retraining path for the historical Full checkpoint.
