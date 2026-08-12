"""Training and evaluation for the lightweight student.

The training objective includes trait-weighted SmoothL1 regression,
extreme-label regularization, variance matching, Agreeableness label
contrastive learning, feature-level adversarial training, and offline
prediction distillation. Optimization uses Adam with OneCycleLR; early
stopping monitors validation R2 and final model selection uses validation RACC.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from lightning.fabric import seed_everything

from model import (
    EXPECTED_TRAINABLE_PARAMETERS,
    FinalStudent,
    TRAITS,
    assert_parameter_count,
)


def load_config(path: str | Path) -> dict[str, Any]:
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


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute the paper's average RACC/R2/PCC metrics over five OCEAN traits."""
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2 or pred.shape[1] != 5:
        raise ValueError(f"Expected matching [N,5] arrays, got {pred.shape} and {true.shape}")

    pred = np.clip(pred, 0.0, 1.0)
    mae = np.mean(np.abs(pred - true), axis=0)
    racc = 1.0 - mae

    centered_true = true - true.mean(axis=0, keepdims=True)
    ss_total = np.sum(centered_true**2, axis=0)
    r2 = 1.0 - np.sum((true - pred) ** 2, axis=0) / np.maximum(ss_total, 1e-12)

    centered_pred = pred - pred.mean(axis=0, keepdims=True)
    pcc_den = np.sqrt(
        np.sum(centered_true**2, axis=0) * np.sum(centered_pred**2, axis=0)
    )
    pcc = np.sum(centered_true * centered_pred, axis=0) / np.maximum(pcc_den, 1e-12)

    result: dict[str, float] = {
        "mae": float(mae.mean()),
        "racc": float(racc.mean()),
        "r2": float(r2.mean()),
        "pcc": float(pcc.mean()),
    }
    for idx, trait in enumerate(TRAITS):
        result[f"{trait}_racc"] = float(racc[idx])
        result[f"{trait}_r2"] = float(r2[idx])
        result[f"{trait}_pcc"] = float(pcc[idx])
    return result


def agreeableness_label_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    agreeableness_index: int = 3,
    top_k: int = 3,
    temperature: float = 0.2,
    label_sigma: float = 0.08,
) -> torch.Tensor:
    """Continuous-label contrastive loss for Agreeableness."""
    if features.size(0) <= 1:
        return features.sum() * 0.0
    if features.ndim == 3:
        features = features.mean(dim=1)

    features = F.normalize(features, dim=1)
    a_labels = labels[:, agreeableness_index].view(-1, 1)
    similarity = torch.matmul(features, features.T) / max(float(temperature), 1e-8)
    distance = torch.abs(a_labels - a_labels.T)

    batch_size = features.size(0)
    eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    similarity = similarity.masked_fill(eye, -1e9)
    distance = distance.masked_fill(eye, 1e9)

    k = min(int(top_k), batch_size - 1)
    if k <= 0:
        return features.sum() * 0.0

    positive_indices = torch.topk(distance, k=k, largest=False, dim=1).indices
    positive_mask = torch.zeros_like(distance, dtype=torch.bool)
    positive_mask.scatter_(1, positive_indices, True)
    positive_weight = (
        torch.exp(-distance / max(float(label_sigma), 1e-8))
        * positive_mask.float()
    )

    exp_similarity = torch.exp(similarity) * (~eye).float()
    numerator = (exp_similarity * positive_weight).sum(dim=1)
    denominator = exp_similarity.sum(dim=1).clamp_min(1e-8)
    valid = numerator > 0
    if valid.sum() == 0:
        return features.sum() * 0.0
    return -torch.log((numerator[valid] / denominator[valid]).clamp_min(1e-8)).mean()


class StudentTrainingModule(L.LightningModule):
    """Lightning training wrapper around the exact 500,107-parameter student."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.save_hyperparameters({"config": self.config})

        self.model = FinalStudent(self.config)
        assert_parameter_count(self.model)

        self.feature_list = list(
            self.config.get(
                "feature_list",
                ["dinov2_face", "wavlm", "roberta", "egemaps_lld"],
            )
        )
        self.pretrain_epochs = int(self.config.get("pretrain_epochs", 5))
        self.cl_start_epoch = int(self.config.get("cl_start_epoch", 5))
        self.cl_warmup_epochs = int(self.config.get("cl_warmup_epochs", 5))
        self.adversarial_start_epoch = int(
            self.config.get("adversarial_start_epoch", 10)
        )
        self.feat_eps = float(self.config.get("feat_eps", 0.005))

        self.agreeableness_index = int(self.config.get("agreeableness_index", 3))
        self.trait_loss_alpha_a = float(self.config.get("trait_loss_alpha_a", 1.5))
        self.use_agreeableness_cl = bool(
            self.config.get("use_agreeableness_cl", True)
        )
        self.lambda_agreeableness_cl = float(
            self.config.get("lambda_agreeableness_cl", 0.05)
        )
        self.agreeableness_top_k = int(self.config.get("agreeableness_top_k", 3))
        self.agreeableness_temperature = float(
            self.config.get("agreeableness_temperature", 0.2)
        )
        self.agreeableness_label_sigma = float(
            self.config.get("agreeableness_label_sigma", 0.08)
        )

        self.lambda_extreme = float(self.config.get("lambda_extreme", 0.05))
        self.extreme_weight_strength = float(
            self.config.get("extreme_weight_strength", 1.0)
        )
        self.lambda_var = float(self.config.get("lambda_var", 0.005))

        self.use_prediction_kd = bool(self.config.get("use_prediction_kd", True))
        self.teacher_predictions_path = str(
            self.config.get("teacher_predictions_path", "")
        ).strip()
        self.lambda_prediction_kd = float(
            self.config.get("lambda_prediction_kd", 0.2)
        )
        self.kd_start_epoch = int(self.config.get("kd_start_epoch", 0))
        self.kd_warmup_epochs = int(self.config.get("kd_warmup_epochs", 5))
        self.kd_smooth_l1_beta = float(
            self.config.get("kd_smooth_l1_beta", 1.0)
        )
        self.kd_on_clean_prediction_only = bool(
            self.config.get("kd_on_clean_prediction_only", True)
        )
        if not self.kd_on_clean_prediction_only:
            raise ValueError("Prediction KD is defined on clean predictions only")

        self._teacher_prediction_lookup: dict[str, np.ndarray] | None = None

        self.train_predictions: list[torch.Tensor] = []
        self.train_targets: list[torch.Tensor] = []
        self.valid_predictions: list[torch.Tensor] = []
        self.valid_targets: list[torch.Tensor] = []
        self.test_predictions: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []
        self.test_sample_ids: list[str] = []

    def forward(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.model(features, masks)

    def on_fit_start(self) -> None:
        self._load_teacher_predictions_if_needed()

    @staticmethod
    def _normalize_sample_id(value: Any) -> str:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("Tensor sample_id must be scalar")
            value = value.detach().cpu().item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        sample_id = str(value).strip()
        if not sample_id:
            raise ValueError("Empty sample_id encountered")
        return sample_id

    def _load_teacher_predictions_if_needed(self) -> None:
        if not self.use_prediction_kd:
            return
        if self._teacher_prediction_lookup is not None:
            return
        if not self.teacher_predictions_path:
            raise ValueError("Prediction KD requires teacher_predictions_path")

        teacher_path = Path(self.teacher_predictions_path).expanduser().resolve()
        if not teacher_path.is_file():
            raise FileNotFoundError(
                "Teacher prediction NPZ not found. Set teacher_predictions_path "
                f"in the configuration. Resolved path: {teacher_path}"
            )

        with np.load(teacher_path, allow_pickle=False) as payload:
            if not {"sample_ids", "predictions"}.issubset(payload.files):
                raise KeyError("Teacher NPZ must contain sample_ids and predictions")
            ids = payload["sample_ids"]
            predictions = np.asarray(payload["predictions"], dtype=np.float32)

        if predictions.ndim != 2 or predictions.shape[1] != 5:
            raise ValueError(
                f"Teacher predictions must be [N,5], got {predictions.shape}"
            )
        normalized_ids = [self._normalize_sample_id(value) for value in ids.tolist()]
        if len(normalized_ids) != predictions.shape[0]:
            raise ValueError("Teacher ID/prediction counts do not match")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("Teacher prediction NPZ contains duplicate sample IDs")
        if not np.isfinite(predictions).all():
            raise ValueError("Teacher predictions contain NaN/Inf")

        self._teacher_prediction_lookup = {
            sample_id: predictions[index].copy()
            for index, sample_id in enumerate(normalized_ids)
        }

    def _lookup_teacher_predictions(
        self,
        sample_ids: list[Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self._load_teacher_predictions_if_needed()
        if self._teacher_prediction_lookup is None:
            raise RuntimeError("Teacher lookup is not initialized")

        normalized = [self._normalize_sample_id(value) for value in sample_ids]
        missing = [x for x in normalized if x not in self._teacher_prediction_lookup]
        if missing:
            raise KeyError(f"Missing teacher predictions for IDs: {missing[:10]}")
        values = np.stack(
            [self._teacher_prediction_lookup[x] for x in normalized], axis=0
        )
        return torch.as_tensor(values, device=device, dtype=dtype).detach()

    def _build_masks(
        self,
        batch: dict[str, Any],
    ) -> list[torch.Tensor]:
        masks: list[torch.Tensor] = []
        for feature_name in self.feature_list:
            key = feature_name + "_mask"
            if key in batch:
                masks.append(batch[key])
            else:
                feature = batch[feature_name]
                masks.append(
                    torch.ones(
                        feature.shape[:2],
                        dtype=torch.bool,
                        device=feature.device,
                    )
                )
        return masks

    def _decayed_lambda(
        self,
        base_value: float,
        start_epoch: int,
        warmup_epochs: int,
    ) -> float:
        base_value = float(base_value)
        if base_value <= 0 or self.current_epoch < start_epoch:
            return 0.0
        if warmup_epochs > 0 and self.current_epoch < start_epoch + warmup_epochs:
            progress = (self.current_epoch - start_epoch + 1) / float(warmup_epochs)
            return base_value * max(0.0, min(1.0, progress))

        max_epochs = int(getattr(self.trainer, "max_epochs", 100) or 100)
        decay_epochs = max_epochs - (start_epoch + warmup_epochs)
        if decay_epochs <= 0:
            return base_value
        decay_progress = (
            self.current_epoch - (start_epoch + warmup_epochs)
        ) / float(decay_epochs)
        min_factor = 0.01
        decay_factor = min_factor + 0.5 * (1.0 - min_factor) * (
            1.0 + math.cos(math.pi * decay_progress)
        )
        return base_value * decay_factor

    def _prediction_kd_weight(self) -> float:
        if not self.use_prediction_kd or self.lambda_prediction_kd <= 0:
            return 0.0
        if self.current_epoch < self.kd_start_epoch:
            return 0.0
        if self.kd_warmup_epochs <= 0:
            return self.lambda_prediction_kd
        elapsed = self.current_epoch - self.kd_start_epoch + 1
        progress = min(1.0, max(0.0, elapsed / float(self.kd_warmup_epochs)))
        return self.lambda_prediction_kd * progress

    def _regression_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        trait_loss = F.smooth_l1_loss(prediction, target, reduction="none")
        trait_weights = torch.ones(5, device=prediction.device, dtype=prediction.dtype)
        trait_weights[self.agreeableness_index] = self.trait_loss_alpha_a
        loss = (trait_loss * trait_weights.view(1, -1)).mean()

        if self.extreme_weight_strength > 0 and self.lambda_extreme > 0:
            weight = 1.0 + self.extreme_weight_strength * torch.abs(target - 0.5)
            weight[:, self.agreeableness_index] = (
                weight[:, self.agreeableness_index] * self.trait_loss_alpha_a
            )
            element_loss = F.smooth_l1_loss(prediction, target, reduction="none")
            loss = loss + self.lambda_extreme * (weight.detach() * element_loss).mean()

        if self.lambda_var > 0 and prediction.size(0) > 1:
            prediction_std = prediction.std(dim=0, unbiased=False)
            target_std = target.std(dim=0, unbiased=False)
            loss = loss + self.lambda_var * F.mse_loss(
                prediction_std, target_std.detach()
            )
        return loss

    def _feature_adversarial_perturbation(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor],
        target: torch.Tensor,
    ) -> list[torch.Tensor]:
        epsilon = self._decayed_lambda(
            self.feat_eps,
            start_epoch=self.adversarial_start_epoch,
            warmup_epochs=0,
        )
        if epsilon <= 0:
            return [feature.detach() for feature in features]

        seeds = [feature.detach().clone().requires_grad_(True) for feature in features]
        with torch.enable_grad():
            seed_prediction = self.model(seeds, masks)
            seed_loss = self._regression_loss(seed_prediction, target)
        gradients = torch.autograd.grad(
            seed_loss,
            seeds,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        adversarial: list[torch.Tensor] = []
        for idx, (seed, gradient) in enumerate(zip(seeds, gradients)):
            if gradient is None:
                adversarial.append(seed.detach())
                continue
            perturbation = epsilon * torch.sign(gradient.detach())
            if masks is not None and idx < len(masks) and masks[idx] is not None:
                valid = masks[idx].bool().unsqueeze(-1).to(perturbation.device)
                perturbation = perturbation * valid
            adversarial.append(seed.detach() + perturbation)
        return adversarial

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        features = [batch[name] for name in self.feature_list]
        masks = self._build_masks(batch)
        target = batch["app"]
        sample_ids = list(batch["sample_id"])

        clean_prediction, clean_repr, _ = self.model(
            features, masks, return_aux=True
        )

        if self.current_epoch < self.pretrain_epochs:
            prediction_for_metrics = clean_prediction
            task_loss = self._regression_loss(clean_prediction, target)
            a_cl_loss = clean_repr.sum() * 0.0
            a_cl_weight = 0.0
        else:
            if self.current_epoch >= self.adversarial_start_epoch:
                adversarial_features = self._feature_adversarial_perturbation(
                    features, masks, target
                )
                adv_prediction, adv_repr, _ = self.model(
                    adversarial_features, masks, return_aux=True
                )
                task_loss = self._regression_loss(adv_prediction, target)
                prediction_for_metrics = adv_prediction
            else:
                adv_repr = clean_repr
                task_loss = self._regression_loss(clean_prediction, target)
                prediction_for_metrics = clean_prediction

            a_cl_weight = self._decayed_lambda(
                self.lambda_agreeableness_cl,
                start_epoch=self.cl_start_epoch,
                warmup_epochs=self.cl_warmup_epochs,
            )
            if self.use_agreeableness_cl and a_cl_weight > 0:
                a_cl_loss = agreeableness_label_contrastive_loss(
                    features=adv_repr,
                    labels=target,
                    agreeableness_index=self.agreeableness_index,
                    top_k=self.agreeableness_top_k,
                    temperature=self.agreeableness_temperature,
                    label_sigma=self.agreeableness_label_sigma,
                )
            else:
                a_cl_loss = clean_repr.sum() * 0.0

        loss = task_loss + a_cl_weight * a_cl_loss

        kd_weight = self._prediction_kd_weight()
        kd_loss = clean_prediction.sum() * 0.0
        if kd_weight > 0:
            teacher_prediction = self._lookup_teacher_predictions(
                sample_ids,
                device=clean_prediction.device,
                dtype=clean_prediction.dtype,
            )
            kd_loss = F.smooth_l1_loss(
                clean_prediction,
                teacher_prediction,
                beta=max(self.kd_smooth_l1_beta, 1e-8),
            )
            loss = loss + kd_weight * kd_loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_task_loss", task_loss, on_step=True, on_epoch=True)
        self.log("train_a_cl_loss", a_cl_loss, on_step=True, on_epoch=False)
        self.log("a_cl_weight", float(a_cl_weight), on_step=True, on_epoch=False)
        self.log("train_prediction_kd_loss", kd_loss, on_step=True, on_epoch=False)
        self.log("prediction_kd_weight", float(kd_weight), on_step=True, on_epoch=False)

        self.train_predictions.append(prediction_for_metrics.detach())
        self.train_targets.append(target.detach())
        return loss

    def on_train_epoch_end(self) -> None:
        if not self.train_predictions:
            return
        predictions = torch.cat(self.train_predictions).cpu().numpy()
        targets = torch.cat(self.train_targets).cpu().numpy()
        metrics = calculate_metrics(predictions, targets)
        self.log("train_racc", metrics["racc"], prog_bar=True)
        self.log("train_r2", metrics["r2"], prog_bar=True)
        self.train_predictions.clear()
        self.train_targets.clear()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        features = [batch[name] for name in self.feature_list]
        masks = self._build_masks(batch)
        target = batch["app"]
        prediction = self.model(features, masks)
        # Validation loss is logging-only. Checkpointing/early stopping use RACC/R2.
        self.log(
            "valid_loss",
            F.l1_loss(prediction, target),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.valid_predictions.append(prediction.detach())
        self.valid_targets.append(target.detach())

    def on_validation_epoch_end(self) -> None:
        if not self.valid_predictions:
            return
        predictions = torch.cat(self.valid_predictions).cpu().numpy()
        targets = torch.cat(self.valid_targets).cpu().numpy()
        metrics = calculate_metrics(predictions, targets)
        self.log("valid_racc", metrics["racc"], prog_bar=True)
        self.log("valid_r2", metrics["r2"], prog_bar=True)
        self.valid_predictions.clear()
        self.valid_targets.clear()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        features = [batch[name] for name in self.feature_list]
        masks = self._build_masks(batch)
        target = batch["app"]
        prediction = self.model(features, masks)
        self.test_predictions.append(prediction.detach())
        self.test_targets.append(target.detach())
        self.test_sample_ids.extend(str(x) for x in batch["sample_id"])

    def on_test_epoch_end(self) -> None:
        predictions = torch.cat(self.test_predictions).cpu().numpy()
        targets = torch.cat(self.test_targets).cpu().numpy()
        metrics = calculate_metrics(predictions, targets)
        self.log("test_racc", metrics["racc"], prog_bar=True)
        self.log("test_r2", metrics["r2"], prog_bar=True)
        self.log("test_pcc", metrics["pcc"], prog_bar=True)

        output_dir = Path(self.config.get("experiment_dir", "results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"sample_id": self.test_sample_ids}
        for index, trait in enumerate(TRAITS):
            payload[f"pred_{trait}"] = np.clip(predictions[:, index], 0.0, 1.0)
            payload[f"target_{trait}"] = targets[:, index]
        pd.DataFrame(payload).to_csv(
            output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
        )
        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)

        self.test_predictions.clear()
        self.test_targets.clear()
        self.test_sample_ids.clear()

    def configure_optimizers(self):
        optimizer_config = self.config.get("optimizer", {})
        base_lr = float(optimizer_config.get("base_lr", 7e-4))
        weight_decay = float(optimizer_config.get("weight_decay", 0.0))
        optimizer = torch.optim.Adam(
            self.parameters(), lr=base_lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        # Keep the return value compact for the training loop.
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def build_callbacks(output_dir: Path, config: dict[str, Any]):
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_racc = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="checkpoint_valid_racc",
        monitor="valid_racc",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
    )
    checkpoint_r2 = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="checkpoint_valid_r2",
        monitor="valid_r2",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
    )
    early_config = config.get("early_stopping", {})
    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor="valid_r2",
        patience=int(early_config.get("patience", 20)),
        mode="max",
        verbose=True,
    )
    return [checkpoint_racc, checkpoint_r2, early_stopping], checkpoint_racc


def run(config: dict[str, Any], test_only_checkpoint: Path | None = None) -> None:
    # Local import keeps model.py independently importable for parameter verification.
    from fi import FiDataModule

    seed_everything(int(config.get("seed", 42)), workers=True)
    torch.set_float32_matmul_precision(
        str(config.get("float32_matmul_precision", "medium"))
    )

    output_dir = Path(config.get("output_dir", "results")) / str(
        config.get("run_name", "public_q3_student")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["experiment_dir"] = str(output_dir)

    data_module = FiDataModule(config=config)
    module = StudentTrainingModule(config)
    parameter_count = assert_parameter_count(module.model)
    print(f"[ModelSize] trainable parameters = {parameter_count:,}")
    if parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("Student parameter-count check failed")

    callbacks, checkpoint_racc = build_callbacks(output_dir, config)
    trainer = L.Trainer(
        accelerator=config.get("accelerator", "gpu"),
        devices=config.get("devices", [0]),
        max_epochs=int(config.get("n_epochs", 100)),
        callbacks=callbacks,
        logger=L.pytorch.loggers.CSVLogger(str(output_dir), name="csv_logs"),
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    if test_only_checkpoint is None:
        trainer.fit(module, datamodule=data_module)
        checkpoint_path = Path(checkpoint_racc.best_model_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError("No validation-RACC checkpoint was produced")
    else:
        checkpoint_path = test_only_checkpoint

    print(f"[FinalSelection] testing validation-RACC checkpoint: {checkpoint_path}")
    module = StudentTrainingModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config=config,
        map_location="cpu",
    )
    trainer.test(module, datamodule=data_module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the final Trait-Interactive TCMS-Lite student."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for test-only evaluation.",
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        default=None,
        help="Override config db_root without editing the YAML.",
    )
    parser.add_argument(
        "--teacher-predictions",
        type=Path,
        default=None,
        help="Override teacher_predictions_path used for offline Prediction KD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.db_root is not None:
        config["db_root"] = str(args.db_root)
    if args.teacher_predictions is not None:
        config["teacher_predictions_path"] = str(args.teacher_predictions)
    run(config, test_only_checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
