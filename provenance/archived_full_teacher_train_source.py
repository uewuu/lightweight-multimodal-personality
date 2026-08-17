from pathlib import Path
from datetime import datetime
import math  # 用于 Cosine Decay 计算
import shutil  # 新增：用于执行文件和目录的复制
import sys
import os

# ================= 🚀 核心劫持逻辑 =================
# 获取 train_app.py 所在的目录，向上推两级获取到项目根目录 (FGSM-SDI-master)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
current_dir = os.path.dirname(os.path.abspath(__file__))

# 强行将项目根目录和当前目录插入到 Python 搜索路径的绝对第一位和第二位！
sys.path.insert(0, project_root)
sys.path.insert(1, current_dir)
# ==================================================

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.fabric import seed_everything
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from exordium.utils.loss import bell_l2_l1_loss
from fi import FiDataModule, custom_collate_fn

from linmult import LinMulT

from personalitylinmult.train.callbacks import TimeTrackingCallback
from personalitylinmult.train.history import History
from personalitylinmult.train.metrics import calculate_app_metrics
from personalitylinmult.train.parser import argparser
def _resolve_output_dim(config: dict, default: int = 5) -> int:
    output_dim_cfg = config.get("output_dim", default)
    if isinstance(output_dim_cfg, (list, tuple)):
        return int(output_dim_cfg[0]) if len(output_dim_cfg) > 0 else int(default)
    return int(output_dim_cfg)


def _infer_label_mean_from_loader(train_loader, max_batches: int | None = None) -> list[float]:
    """Compute OCEAN label mean from the training split only.

    This is used by Residual Prediction Head:
        prediction = train_label_mean + delta

    Using the training split avoids validation/test leakage.
    """
    all_labels = []
    for batch_idx, batch in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if "app" not in batch:
            raise KeyError("Expected batch['app'] to contain five-dimensional OCEAN labels.")
        y = batch["app"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        all_labels.append(y.detach().float().cpu())

    if len(all_labels) == 0:
        raise RuntimeError("No labels found when computing label_mean from train_loader.")

    labels = torch.cat(all_labels, dim=0)
    if labels.ndim != 2 or labels.size(1) != 5:
        raise ValueError(f"Expected OCEAN labels with shape [N, 5], got {tuple(labels.shape)}")

    return labels.mean(dim=0).tolist()


def pseudo_label_contrastive_loss(
        anchor_features: torch.Tensor,
        anchor_ids: list[str],
        anchor_labels: torch.Tensor,
        bank_features: torch.Tensor | None,
        bank_ids,
        bank_labels: torch.Tensor | None,
        temperature: float = 0.5,
) -> torch.Tensor:
    if bank_features is None or bank_labels is None or bank_ids is None:
        return torch.tensor(0.0, device=anchor_features.device)

    valid_anchor_mask = anchor_labels >= 0
    if valid_anchor_mask.sum() == 0:
        return torch.tensor(0.0, device=anchor_features.device)

    anchor_features = anchor_features[valid_anchor_mask]
    anchor_labels = anchor_labels[valid_anchor_mask]
    valid_anchor_ids = [
        anchor_ids[i]
        for i in range(len(anchor_ids))
        if valid_anchor_mask[i].item()
    ]

    bank_features = bank_features.to(anchor_features.device)
    bank_labels = bank_labels.to(anchor_features.device)
    bank_ids = np.asarray(bank_ids)

    anchor_features = F.normalize(anchor_features, dim=1)
    bank_features = F.normalize(bank_features, dim=1)

    logits = torch.matmul(anchor_features, bank_features.T) / temperature

    losses = []
    for i in range(anchor_features.size(0)):
        same_id_mask = torch.tensor(
            bank_ids == valid_anchor_ids[i],
            device=anchor_features.device,
        )
        valid_bank_mask = ~same_id_mask

        if valid_bank_mask.sum() == 0:
            continue

        logits_i = logits[i][valid_bank_mask]
        labels_i = bank_labels[valid_bank_mask]

        pos_mask = labels_i == anchor_labels[i]
        if pos_mask.sum() == 0:
            continue

        log_prob = logits_i - torch.logsumexp(logits_i, dim=0)
        losses.append(-log_prob[pos_mask].mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=anchor_features.device)

    return torch.stack(losses).mean()


def agreeableness_label_contrastive_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        agreeableness_index: int = 3,
        top_k: int = 5,
        temperature: float = 0.2,
        label_sigma: float = 0.08,
) -> torch.Tensor:
    """
    Method 1: Agreeableness-aware label contrastive loss.

    For each anchor, samples with the nearest Agreeableness labels in the same mini-batch
    are treated as multiple positives. Positives are softly weighted by their label distance.
    This is designed for continuous Big Five regression, not discrete classification.
    """
    if features is None or labels is None:
        return torch.tensor(0.0, device=labels.device if labels is not None else "cpu")

    if features.size(0) <= 1:
        return torch.tensor(0.0, device=features.device)

    if labels.ndim != 2 or labels.size(1) <= agreeableness_index:
        return torch.tensor(0.0, device=features.device)

    if features.ndim == 3:
        features = features.mean(dim=1)

    features = F.normalize(features, dim=1)
    a_labels = labels[:, agreeableness_index].view(-1, 1)

    sim = torch.matmul(features, features.T) / max(temperature, 1e-8)
    dist = torch.abs(a_labels - a_labels.T)

    batch_size = features.size(0)
    eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    sim = sim.masked_fill(eye, -1e9)
    dist = dist.masked_fill(eye, 1e9)

    k = min(int(top_k), batch_size - 1)
    if k <= 0:
        return torch.tensor(0.0, device=features.device)

    pos_indices = torch.topk(dist, k=k, largest=False, dim=1).indices
    pos_mask = torch.zeros_like(dist, dtype=torch.bool)
    pos_mask.scatter_(1, pos_indices, True)

    # Soft positive weights: closer Agreeableness labels receive stronger attraction.
    pos_weight = torch.exp(-dist / max(label_sigma, 1e-8)) * pos_mask.float()

    # Numerically stable supervised contrastive form.
    exp_sim = torch.exp(sim) * (~eye).float()
    numerator = (exp_sim * pos_weight).sum(dim=1)
    denominator = exp_sim.sum(dim=1).clamp_min(1e-8)

    valid = numerator > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    loss = -torch.log((numerator[valid] / denominator[valid]).clamp_min(1e-8))
    return loss.mean()


def regression_aware_contrastive_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        top_k: int = 3,
        temperature: float = 0.2,
        label_sigma: float = 0.08,
        trait_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Regression-aware multi-positive contrastive loss for continuous OCEAN labels.

    Different from pseudo-label CL, this loss does not discretize samples into clusters.
    It builds soft positives using continuous label distances:
        closer OCEAN labels -> stronger positive weights.

    This is safer for continuous personality regression because it preserves fine-grained
    differences and reduces the risk of prediction distribution compression.
    """
    if features is None or labels is None:
        device = labels.device if labels is not None else "cpu"
        return torch.tensor(0.0, device=device)

    if features.size(0) <= 1:
        return torch.tensor(0.0, device=features.device)

    if labels.ndim != 2:
        return torch.tensor(0.0, device=features.device)

    if features.ndim == 3:
        features = features.mean(dim=1)

    batch_size = features.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=features.device)

    features = F.normalize(features, dim=1)
    labels = labels.detach()

    sim = torch.matmul(features, features.T) / max(float(temperature), 1e-8)
    eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    sim = sim.masked_fill(eye, -1e9)

    n_traits = labels.size(1)
    if trait_weights is None:
        trait_weights = torch.ones(n_traits, device=features.device, dtype=features.dtype)
    else:
        trait_weights = trait_weights.to(device=features.device, dtype=features.dtype)
        if trait_weights.numel() != n_traits:
            trait_weights = torch.ones(n_traits, device=features.device, dtype=features.dtype)

    pos_weight = torch.zeros(batch_size, batch_size, device=features.device, dtype=features.dtype)
    k = min(int(top_k), batch_size - 1)
    if k <= 0:
        return torch.tensor(0.0, device=features.device)

    for trait_idx in range(n_traits):
        y = labels[:, trait_idx].view(-1, 1)
        dist = torch.abs(y - y.T)
        dist = dist.masked_fill(eye, 1e9)

        pos_indices = torch.topk(dist, k=k, largest=False, dim=1).indices
        pos_mask = torch.zeros_like(dist, dtype=torch.bool)
        pos_mask.scatter_(1, pos_indices, True)

        soft_weight = torch.exp(-dist / max(float(label_sigma), 1e-8)) * pos_mask.float()
        pos_weight = pos_weight + trait_weights[trait_idx] * soft_weight

    # Normalize row-wise so samples with many positives do not dominate.
    pos_weight = pos_weight / pos_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)

    exp_sim = torch.exp(sim) * (~eye).float()
    numerator = (exp_sim * pos_weight).sum(dim=1)
    denominator = exp_sim.sum(dim=1).clamp_min(1e-8)

    valid = numerator > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    loss = -torch.log((numerator[valid] / denominator[valid]).clamp_min(1e-8))
    return loss.mean()


def behavior_aware_regression_contrastive_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        behavior_repr: torch.Tensor | None,
        top_k: int = 3,
        temperature: float = 0.2,
        label_sigma: float = 0.08,
        behavior_sigma: float = 0.5,
        behavior_weight_alpha: float = 0.5,
        trait_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Behavior-aware adversarial regression contrastive loss.

    This is a behavior-guided extension of regression-aware CL for continuous
    OCEAN regression. Soft positive weights are determined by both:
        1) continuous OCEAN label proximity;
        2) behavior/reliability state similarity.

    Compared with pseudo-label ACL, this loss does not discretize continuous
    personality scores into hard clusters. It is designed to be used with
    clean/adv feature pairs:
        features      = concat(clean_repr.detach(), adv_repr)
        labels        = concat(y_true, y_true)
        behavior_repr = concat(clean_behavior.detach(), adv_behavior)

    behavior_weight_alpha controls how strongly behavior similarity modulates
    the label-based positive weights:
        final_weight = label_weight * ((1-alpha) + alpha * behavior_weight)

    A moderate alpha is safer than multiplying by behavior_weight directly,
    because behavior state estimates can be noisy in early training.
    """
    if features is None or labels is None or behavior_repr is None:
        device = features.device if features is not None else (labels.device if labels is not None else "cpu")
        return torch.tensor(0.0, device=device)

    if features.size(0) <= 1:
        return torch.tensor(0.0, device=features.device)

    if features.ndim == 3:
        features = features.mean(dim=1)
    if behavior_repr.ndim == 3:
        behavior_repr = behavior_repr.mean(dim=1)

    if labels.ndim != 2 or behavior_repr.ndim != 2:
        return torch.tensor(0.0, device=features.device)

    if features.size(0) != labels.size(0) or features.size(0) != behavior_repr.size(0):
        return torch.tensor(0.0, device=features.device)

    batch_size = features.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=features.device)

    features = F.normalize(features, dim=1)
    behavior_repr = F.normalize(behavior_repr.detach(), dim=1)
    labels = labels.detach()

    sim = torch.matmul(features, features.T) / max(float(temperature), 1e-8)
    eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    sim = sim.masked_fill(eye, -1e9)

    behavior_sim = torch.matmul(behavior_repr, behavior_repr.T).clamp(-1.0, 1.0)
    behavior_dist = (1.0 - behavior_sim).clamp_min(0.0)
    behavior_weight = torch.exp(-behavior_dist / max(float(behavior_sigma), 1e-8))
    behavior_weight = behavior_weight.masked_fill(eye, 0.0)

    alpha = float(behavior_weight_alpha)
    alpha = max(0.0, min(1.0, alpha))

    n_traits = labels.size(1)
    if trait_weights is None:
        trait_weights = torch.ones(n_traits, device=features.device, dtype=features.dtype)
    else:
        trait_weights = trait_weights.to(device=features.device, dtype=features.dtype)
        if trait_weights.numel() != n_traits:
            trait_weights = torch.ones(n_traits, device=features.device, dtype=features.dtype)

    pos_weight = torch.zeros(batch_size, batch_size, device=features.device, dtype=features.dtype)
    k = min(int(top_k), batch_size - 1)
    if k <= 0:
        return torch.tensor(0.0, device=features.device)

    for trait_idx in range(n_traits):
        y = labels[:, trait_idx].view(-1, 1)
        dist = torch.abs(y - y.T)
        dist = dist.masked_fill(eye, 1e9)

        pos_indices = torch.topk(dist, k=k, largest=False, dim=1).indices
        pos_mask = torch.zeros_like(dist, dtype=torch.bool)
        pos_mask.scatter_(1, pos_indices, True)

        label_weight = torch.exp(-dist / max(float(label_sigma), 1e-8)) * pos_mask.float()
        behavior_modulator = (1.0 - alpha) + alpha * behavior_weight
        soft_weight = label_weight * behavior_modulator
        pos_weight = pos_weight + trait_weights[trait_idx] * soft_weight

    pos_weight = pos_weight / pos_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)

    exp_sim = torch.exp(sim) * (~eye).float()
    numerator = (exp_sim * pos_weight).sum(dim=1)
    denominator = exp_sim.sum(dim=1).clamp_min(1e-8)

    valid = numerator > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    loss = -torch.log((numerator[valid] / denominator[valid]).clamp_min(1e-8))
    return loss.mean()


class TraitAwareFusion(torch.nn.Module):
    """Lightweight trait-aware fusion gate.

    The original LinMulT produces one fused representation for all five traits.
    This lightweight adapter creates five trait-specific gated representations
    without using heavy per-trait adapters.

    Difference from the previous heavy version:
        old: hidden + gate(hidden) * adapter(hidden)
        new: hidden * gate(hidden)

    Input:  hidden [B, D]
    Output: trait_hidden [B, 5, D]
    """

    def __init__(
            self,
            input_dim: int,
            n_traits: int = 5,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.n_traits = int(n_traits)

        # Bottleneck gate: D -> D/4 -> D.
        # This keeps trait-specific fusion expressive but avoids the old heavy
        # adapter blocks with Linear(D, D) x 2 for every trait.
        if hidden_dim is None:
            bottleneck_dim = max(32, int(input_dim // 4))
        else:
            bottleneck_dim = max(8, int(hidden_dim))

        self.trait_gates = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, bottleneck_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(bottleneck_dim, input_dim),
                torch.nn.Sigmoid(),
            )
            for _ in range(self.n_traits)
        ])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError(f"TraitAwareFusion expects [B, D], got {tuple(hidden.shape)}")

        trait_hidden = []
        for gate in self.trait_gates:
            g = gate(hidden)
            h = hidden * g
            trait_hidden.append(h.unsqueeze(1))

        return torch.cat(trait_hidden, dim=1)


class TraitSpecificRegressionHead(torch.nn.Module):
    """
    Method 2: decoupled OCEAN regression heads.

    O/C/E/N use light single-output heads.
    Agreeableness uses a deeper MLP head so that the weak A signal is not suppressed
    by the shared five-dimensional regression layer.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int | None = None,
            dropout: float = 0.0,
            use_sigmoid: bool = False,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or input_dim)
        self.use_sigmoid = use_sigmoid

        self.heads = torch.nn.ModuleList()
        for trait_idx in range(5):
            if trait_idx == 3:
                self.heads.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(input_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_dim, 1),
                    )
                )
            else:
                self.heads.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(input_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_dim, 1),
                    )
                )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        outputs = []

        if hidden.ndim == 3:
            # hidden: [B, 5, D], each trait receives its own representation.
            if hidden.size(1) != len(self.heads):
                raise ValueError(f"Expected hidden shape [B, 5, D], got {tuple(hidden.shape)}")
            for trait_idx, head in enumerate(self.heads):
                outputs.append(head(hidden[:, trait_idx, :]))
        else:
            # hidden: [B, D], all trait heads share the same fused representation.
            for head in self.heads:
                outputs.append(head(hidden))

        pred = torch.cat(outputs, dim=1)
        if self.use_sigmoid:
            pred = torch.sigmoid(pred)
        return pred






class TraitConditionedModalitySelection(torch.nn.Module):
    """
    Trait-conditioned Modality Selection (TCMS).

    This lightweight module lets each Big Five trait attend to modality-level
    tokens separately. It converts shared modality tokens into trait-specific
    hidden representations:
        modality_tokens: [B, M, D]
        global_hidden:   [B, D]
        trait_hidden:    [B, 5, D]
        attn_weights:    [B, 5, M]

    It is designed to be inserted after ModalityTokenFusion, without replacing
    the existing LinMulT + MTF backbone.
    """

    def __init__(
            self,
            dim: int,
            n_traits: int = 5,
            n_modalities: int = 4,
            dropout: float = 0.1,
            use_global_context: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_traits = int(n_traits)
        self.n_modalities = int(n_modalities)
        self.use_global_context = bool(use_global_context)

        self.trait_queries = torch.nn.Parameter(
            torch.randn(self.n_traits, self.dim) * 0.02
        )
        self.query_proj = torch.nn.Linear(self.dim, self.dim)
        self.key_proj = torch.nn.Linear(self.dim, self.dim)
        self.value_proj = torch.nn.Linear(self.dim, self.dim)

        if self.use_global_context:
            self.global_proj = torch.nn.Linear(self.dim, self.dim)
        else:
            self.global_proj = None

        self.dropout = torch.nn.Dropout(float(dropout))
        self.out_norm = torch.nn.LayerNorm(self.dim)
        self.last_trait_modality_weights = None

    def forward(
            self,
            modality_tokens: torch.Tensor,
            global_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modality_tokens.ndim != 3:
            raise ValueError(
                f"TraitConditionedModalitySelection expects modality_tokens [B, M, D], "
                f"got {tuple(modality_tokens.shape)}"
            )

        batch_size, n_modalities, dim = modality_tokens.shape
        if n_modalities != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} modality tokens, got {n_modalities}")
        if dim != self.dim:
            raise ValueError(f"Expected token dim={self.dim}, got {dim}")

        trait_queries = self.trait_queries.unsqueeze(0).expand(batch_size, -1, -1)

        if self.use_global_context and global_hidden is not None:
            if global_hidden.ndim != 2 or global_hidden.size(-1) != self.dim:
                raise ValueError(
                    f"Expected global_hidden [B, {self.dim}], got {tuple(global_hidden.shape)}"
                )
            trait_queries = trait_queries + self.global_proj(global_hidden).unsqueeze(1)

        q = self.query_proj(trait_queries)      # [B, 5, D]
        k = self.key_proj(modality_tokens)      # [B, M, D]
        v = self.value_proj(modality_tokens)    # [B, M, D]

        attn_logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(max(dim, 1))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, 5, M]
        attn_weights = self.dropout(attn_weights)

        trait_hidden = torch.matmul(attn_weights, v)  # [B, 5, D]
        trait_hidden = self.out_norm(trait_hidden)
        self.last_trait_modality_weights = attn_weights.detach()

        return trait_hidden, attn_weights


class BehaviorReliabilitySummaryToken(torch.nn.Module):
    """
    Behavior-Reliability Summary Token (BRST).

    This lightweight module generates one extra sample-level token from projected
    modality tokens and simple reliability / behavior statistics. It does not
    require extra behavior labels.
    """

    def __init__(
            self,
            n_modalities: int,
            fused_dim: int,
            hidden_ratio: float = 0.5,
            dropout: float = 0.1,
            use_stats: bool = True,
            use_gate: bool = True,
            gate_alpha: float = 0.2,
    ):
        super().__init__()
        self.n_modalities = int(n_modalities)
        self.fused_dim = int(fused_dim)
        self.use_stats = bool(use_stats)
        self.use_gate = bool(use_gate)
        self.gate_alpha = float(gate_alpha)

        hidden_dim = max(32, int(self.fused_dim * float(hidden_ratio)))
        n_pairs = self.n_modalities * (self.n_modalities - 1) // 2
        self.stat_dim = self.n_modalities * 5 + n_pairs

        if self.use_stats:
            self.stat_encoder = torch.nn.Sequential(
                torch.nn.LayerNorm(self.stat_dim),
                torch.nn.Linear(self.stat_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, self.fused_dim),
            )
        else:
            self.stat_encoder = None

        self.summary_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(self.fused_dim),
            torch.nn.Linear(self.fused_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, self.fused_dim),
            torch.nn.Dropout(dropout),
        )

        if self.use_gate:
            self.gate = torch.nn.Sequential(
                torch.nn.LayerNorm(self.fused_dim),
                torch.nn.Linear(self.fused_dim, self.fused_dim),
                torch.nn.Sigmoid(),
            )
        else:
            self.gate = None

        self.out_norm = torch.nn.LayerNorm(self.fused_dim)

    def _masked_raw_pool(self, feat: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            valid_ratio = torch.ones(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
            pooled = feat.mean(dim=1)
            return pooled, valid_ratio

        mask = mask.bool().to(device=feat.device)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        valid_ratio = valid_counts.to(dtype=feat.dtype) / max(int(mask.size(1)), 1)
        masked_feat = feat.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = masked_feat.sum(dim=1) / valid_counts.to(dtype=feat.dtype)
        return pooled, valid_ratio

    def _build_stats(
            self,
            features: list[torch.Tensor],
            masks: list[torch.Tensor] | None,
            modality_tokens: torch.Tensor,
    ) -> torch.Tensor:
        stats = []
        eps = 1e-6

        for i, feat in enumerate(features):
            mask_i = masks[i] if masks is not None and i < len(masks) else None
            pooled_raw, valid_ratio = self._masked_raw_pool(feat, mask_i)

            if mask_i is not None:
                mask_bool = mask_i.bool().to(device=feat.device)
                valid_counts = mask_bool.sum(dim=1).clamp(min=1).to(dtype=feat.dtype)
                centered = feat - pooled_raw.unsqueeze(1)
                centered = centered.masked_fill(~mask_bool.unsqueeze(-1), 0.0)
                temporal_var = (
                    centered.pow(2).sum(dim=(1, 2), keepdim=False)
                    / (valid_counts * max(int(feat.size(-1)), 1)).clamp_min(1.0)
                ).view(-1, 1)
            else:
                temporal_var = feat.var(dim=1, unbiased=False).mean(dim=1, keepdim=True)

            raw_norm = pooled_raw.norm(dim=1, keepdim=True) / math.sqrt(max(int(feat.size(-1)), 1))
            token_i = modality_tokens[:, i, :]
            token_norm = token_i.norm(dim=1, keepdim=True) / math.sqrt(max(int(token_i.size(-1)), 1))
            token_std = token_i.std(dim=1, unbiased=False, keepdim=True)

            stats.extend([
                valid_ratio,
                torch.log1p(temporal_var.clamp_min(0.0)),
                torch.log1p(raw_norm.clamp_min(0.0)),
                torch.log1p(token_norm.clamp_min(0.0)),
                torch.log1p(token_std.abs()),
            ])

        norm_tokens = F.normalize(modality_tokens, dim=-1, eps=eps)
        for i in range(self.n_modalities):
            for j in range(i + 1, self.n_modalities):
                cos_ij = (norm_tokens[:, i, :] * norm_tokens[:, j, :]).sum(dim=-1, keepdim=True)
                stats.append(1.0 - cos_ij)

        return torch.cat(stats, dim=1)

    def forward(
            self,
            features: list[torch.Tensor],
            masks: list[torch.Tensor] | None,
            modality_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if modality_tokens.ndim != 3:
            raise ValueError(f"Expected modality_tokens [B, M, D], got {tuple(modality_tokens.shape)}")
        if modality_tokens.size(1) != self.n_modalities:
            raise ValueError(
                f"Expected {self.n_modalities} modality tokens, got {modality_tokens.size(1)}"
            )

        mean_token = modality_tokens.mean(dim=1)

        if self.use_stats and self.stat_encoder is not None:
            stats = self._build_stats(features, masks, modality_tokens)
            stat_emb = self.stat_encoder(stats.to(dtype=modality_tokens.dtype))
            summary_input = mean_token + stat_emb
        else:
            stat_emb = mean_token
            summary_input = mean_token

        candidate = self.summary_encoder(summary_input)

        if self.use_gate and self.gate is not None:
            gate = self.gate(stat_emb)
            token = mean_token + self.gate_alpha * gate * candidate
        else:
            token = mean_token + self.gate_alpha * candidate

        return self.out_norm(token).unsqueeze(1)


class ModalityTokenFusion(torch.nn.Module):
    """
    Modality Token Fusion.

    It builds one token for each modality by masked pooling, adds one fused token
    from LinMulT, and then applies a lightweight TransformerEncoder over these
    modality-level tokens.

    Input:
        features: list of [B, T_i, D_i]
        masks:    list of [B, T_i]
        fused_repr: [B, T, D] or [B, D]
        fused_mask: [B, T] or None

    Output:
        new_fused_hidden: [B, D]
    """

    def __init__(
            self,
            input_dims: list[int],
            fused_dim: int,
            num_layers: int = 1,
            num_heads: int = 4,
            dropout: float = 0.1,
            ffn_ratio: float = 2.0,
            use_residual: bool = False,
            residual_alpha: float = 0.5,
            use_token_dropout: bool = False,
            token_dropout_prob: float = 0.05,
            use_token_layernorm: bool = False,
            use_behavior_state_token: bool = False,
            behavior_state_hidden_ratio: float = 0.5,
            behavior_state_dropout: float = 0.1,
            behavior_state_use_stats: bool = True,
            behavior_state_gate: bool = True,
            behavior_state_gate_alpha: float = 0.2,
    ):
        super().__init__()
        self.input_dims = list(input_dims)
        self.n_modalities = len(self.input_dims)
        self.fused_dim = int(fused_dim)
        self.use_residual = bool(use_residual)
        self.residual_alpha = float(residual_alpha)
        self.use_token_dropout = bool(use_token_dropout)
        self.token_dropout_prob = float(token_dropout_prob)
        self.use_token_layernorm = bool(use_token_layernorm)
        self.use_behavior_state_token = bool(use_behavior_state_token)
        self.behavior_state_hidden_ratio = float(behavior_state_hidden_ratio)
        self.behavior_state_dropout = float(behavior_state_dropout)
        self.behavior_state_use_stats = bool(behavior_state_use_stats)
        self.behavior_state_gate = bool(behavior_state_gate)
        self.behavior_state_gate_alpha = float(behavior_state_gate_alpha)
        self.last_behavior_repr = None

        num_heads = int(num_heads)
        if self.fused_dim % num_heads != 0:
            print(
                f"[WARNING] fused_dim={self.fused_dim} is not divisible by "
                f"modality_token_fusion_heads={num_heads}. Fallback to num_heads=1."
            )
            num_heads = 1
        self.num_heads = num_heads

        self.modality_projs = torch.nn.ModuleList([
            torch.nn.Linear(int(dim), self.fused_dim)
            for dim in self.input_dims
        ])

        # Optional token-level normalization before token type embedding and TransformerEncoder.
        # This is useful when modality tokens come from heterogeneous feature spaces
        # such as AU / wav2vec2 / RoBERTa / eGeMAPS.
        self.modality_token_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(self.fused_dim)
            for _ in self.input_dims
        ])
        self.fused_token_norm = torch.nn.LayerNorm(self.fused_dim)

        # +1 for the LinMulT fused token. Add one extra token when BRST is enabled.
        self.num_fusion_tokens = self.n_modalities + 1 + int(self.use_behavior_state_token)
        self.token_type_embedding = torch.nn.Parameter(
            torch.randn(self.num_fusion_tokens, self.fused_dim) * 0.02
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.fused_dim,
            nhead=self.num_heads,
            dim_feedforward=max(self.fused_dim, int(self.fused_dim * float(ffn_ratio))),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
        )
        self.out_norm = torch.nn.LayerNorm(self.fused_dim)

        if self.use_behavior_state_token:
            self.behavior_state_generator = BehaviorReliabilitySummaryToken(
                n_modalities=self.n_modalities,
                fused_dim=self.fused_dim,
                hidden_ratio=self.behavior_state_hidden_ratio,
                dropout=self.behavior_state_dropout,
                use_stats=self.behavior_state_use_stats,
                use_gate=self.behavior_state_gate,
                gate_alpha=self.behavior_state_gate_alpha,
            )
        else:
            self.behavior_state_generator = None

    def _masked_pool(self, feat: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if feat.ndim != 3:
            raise ValueError(f"Expected modality feature [B, T, D], got {tuple(feat.shape)}")

        if mask is None:
            return feat.mean(dim=1)

        mask = mask.bool().to(device=feat.device)
        masked_feat = feat.masked_fill(~mask.unsqueeze(-1), 0.0)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return masked_feat.sum(dim=1) / valid_counts

    def _pool_fused_repr(
            self,
            fused_repr: torch.Tensor,
            fused_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if fused_repr.ndim == 2:
            return fused_repr

        if fused_repr.ndim != 3:
            raise ValueError(f"Expected fused_repr [B, T, D] or [B, D], got {tuple(fused_repr.shape)}")

        if fused_mask is None:
            return fused_repr.mean(dim=1)

        fused_mask = fused_mask.bool().to(device=fused_repr.device)
        masked_repr = fused_repr.masked_fill(~fused_mask.unsqueeze(-1), 0.0)
        valid_counts = fused_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return masked_repr.sum(dim=1) / valid_counts

    def _apply_modality_token_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """Drop only modality tokens during training.

        tokens:
            [B, 1 + M, D]
            token 0 = LinMulT fused token, always kept
            token 1..M = modality tokens, randomly dropped during training

        This operation adds no learnable parameters and is disabled automatically
        during validation/test because self.training is False.
        """
        if (not self.training) or (not self.use_token_dropout):
            return tokens

        if self.token_dropout_prob <= 0.0:
            return tokens

        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, T, D], got {tuple(tokens.shape)}")

        batch_size, num_tokens, _ = tokens.shape
        if num_tokens <= 1:
            return tokens

        fused_token = tokens[:, :1, :]
        modality_tokens = tokens[:, 1:, :]

        keep_prob = 1.0 - float(self.token_dropout_prob)
        keep_mask = torch.bernoulli(
            torch.full(
                (batch_size, num_tokens - 1, 1),
                keep_prob,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        )

        # Avoid dropping all modality tokens in any sample.
        all_dropped = keep_mask.sum(dim=1).squeeze(-1) == 0
        if all_dropped.any():
            random_keep_idx = torch.randint(
                low=0,
                high=num_tokens - 1,
                size=(batch_size,),
                device=tokens.device,
            )
            batch_idx = torch.arange(batch_size, device=tokens.device)
            keep_mask[batch_idx[all_dropped], random_keep_idx[all_dropped], 0] = 1.0

        modality_tokens = modality_tokens * keep_mask
        return torch.cat([fused_token, modality_tokens], dim=1)

    def forward(
            self,
            features: list[torch.Tensor],
            masks: list[torch.Tensor] | None,
            fused_repr: torch.Tensor,
            fused_mask: torch.Tensor | None = None,
            return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(features) != self.n_modalities:
            raise ValueError(
                f"Expected {self.n_modalities} modality features, got {len(features)}"
            )

        modality_tokens = []
        for i, feat in enumerate(features):
            mask_i = masks[i] if masks is not None and i < len(masks) else None
            pooled = self._masked_pool(feat, mask_i)
            projected = self.modality_projs[i](pooled)

            if self.use_token_layernorm:
                projected = self.modality_token_norms[i](projected)

            modality_tokens.append(projected.unsqueeze(1))

        modality_tokens = torch.cat(modality_tokens, dim=1)

        fused_token_value = self._pool_fused_repr(fused_repr, fused_mask)
        if self.use_token_layernorm:
            fused_token_value = self.fused_token_norm(fused_token_value)

        fused_token = fused_token_value.unsqueeze(1)

        self.last_behavior_repr = None
        if self.use_behavior_state_token and self.behavior_state_generator is not None:
            behavior_token = self.behavior_state_generator(
                features=features,
                masks=masks,
                modality_tokens=modality_tokens,
            )
            tokens = torch.cat([fused_token, modality_tokens, behavior_token], dim=1)
        else:
            behavior_token = None
            tokens = torch.cat([fused_token, modality_tokens], dim=1)

        if tokens.size(1) != self.token_type_embedding.size(0):
            raise ValueError(
                f"Token count mismatch: tokens={tokens.size(1)}, "
                f"token_type_embedding={self.token_type_embedding.size(0)}"
            )

        tokens = tokens + self.token_type_embedding.unsqueeze(0).to(
            device=tokens.device,
            dtype=tokens.dtype,
        )

        tokens = self._apply_modality_token_dropout(tokens)

        tokens = self.encoder(tokens)
        if self.use_behavior_state_token and behavior_token is not None:
            self.last_behavior_repr = tokens[:, -1, :]
        else:
            self.last_behavior_repr = None
        new_fused_hidden = self.out_norm(tokens[:, 0, :])

        encoded_modality_tokens = tokens[:, 1:1 + self.n_modalities, :]

        if self.use_residual:
            original_fused_hidden = self._pool_fused_repr(fused_repr, fused_mask)

            if original_fused_hidden.shape != new_fused_hidden.shape:
                raise ValueError(
                    f"Residual skip shape mismatch: "
                    f"original={tuple(original_fused_hidden.shape)}, "
                    f"new={tuple(new_fused_hidden.shape)}"
                )

            new_fused_hidden = original_fused_hidden + self.residual_alpha * new_fused_hidden

        if return_tokens:
            return new_fused_hidden, encoded_modality_tokens

        return new_fused_hidden


class ModalityGating(torch.nn.Module):
    def __init__(self, input_dims: list[int], hidden_ratio: float = 0.5):
        super().__init__()
        self.input_dims = input_dims
        self.n_modalities = len(input_dims)

        self.gates = torch.nn.ModuleList()
        for dim in input_dims:
            hidden_dim = max(8, int(dim * hidden_ratio))
            self.gates.append(
                torch.nn.Sequential(
                    torch.nn.Linear(dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
            )

    def forward(self, features: list[torch.Tensor], masks: list[torch.Tensor] | None = None):
        scores = []

        for i, feat in enumerate(features):
            if feat.ndim != 3:
                raise ValueError(f"Expected modality feature [B, T, D], got {feat.shape}")

            if masks is not None and masks[i] is not None:
                mask = masks[i].bool()
                masked_feat = feat.masked_fill(~mask.unsqueeze(-1), 0.0)
                valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = masked_feat.sum(dim=1) / valid_counts
            else:
                pooled = feat.mean(dim=1)

            score = self.gates[i](pooled)
            scores.append(score)

        scores = torch.cat(scores, dim=1)
        weights = torch.softmax(scores, dim=1)
        return weights


class QualityAwareModalityGating(torch.nn.Module):
    def __init__(self, input_dims: list[int], hidden_ratio: float = 0.5):
        super().__init__()
        self.input_dims = input_dims
        self.n_modalities = len(input_dims)

        self.content_nets = torch.nn.ModuleList()
        self.quality_nets = torch.nn.ModuleList()
        self.fusion_nets = torch.nn.ModuleList()

        for dim in input_dims:
            hidden_dim = max(8, int(dim * hidden_ratio))

            self.content_nets.append(
                torch.nn.Sequential(
                    torch.nn.Linear(dim, hidden_dim),
                    torch.nn.ReLU(),
                )
            )

            self.quality_nets.append(
                torch.nn.Sequential(
                    torch.nn.Linear(dim + 3, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, 1),
                    torch.nn.Sigmoid(),
                )
            )

            self.fusion_nets.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim + 1, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
            )

    def _compute_quality_stats(self, feat: torch.Tensor, mask: torch.Tensor | None):
        if mask is not None:
            mask = mask.bool()
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            valid_ratio = valid_counts.float() / mask.shape[1]

            masked_feat = feat.masked_fill(~mask.unsqueeze(-1), 0.0)
            pooled = masked_feat.sum(dim=1) / valid_counts
        else:
            valid_ratio = torch.ones(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
            pooled = feat.mean(dim=1)

        temporal_var = feat.var(dim=1, unbiased=False).mean(dim=1, keepdim=True)
        feature_norm = pooled.norm(dim=1, keepdim=True)

        return valid_ratio, temporal_var, feature_norm

    def forward(self, features: list[torch.Tensor], masks: list[torch.Tensor] | None = None):
        quality_scores = []
        final_scores = []

        for i, feat in enumerate(features):
            if feat.ndim != 3:
                raise ValueError(f"Expected modality feature [B, T, D], got {feat.shape}")

            mask_i = masks[i] if masks is not None else None

            if mask_i is not None:
                mask_i = mask_i.bool()
                masked_feat = feat.masked_fill(~mask_i.unsqueeze(-1), 0.0)
                valid_counts = mask_i.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = masked_feat.sum(dim=1) / valid_counts
            else:
                pooled = feat.mean(dim=1)

            valid_ratio, temporal_var, feature_norm = self._compute_quality_stats(feat, mask_i)
            quality_input = torch.cat([pooled, valid_ratio, temporal_var, feature_norm], dim=1)

            q = self.quality_nets[i](quality_input)
            quality_scores.append(q)

            content_hidden = self.content_nets[i](pooled)
            fusion_input = torch.cat([content_hidden, q], dim=1)
            score = self.fusion_nets[i](fusion_input)
            final_scores.append(score)

        final_scores = torch.cat(final_scores, dim=1)
        weights = torch.softmax(final_scores, dim=1)
        quality_scores = torch.cat(quality_scores, dim=1)

        return weights, quality_scores


class ModelWrapper(L.LightningModule):
    def __init__(self, model, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters("config")

        self.model = model

        loss_fn = config["tasks"]["app"]["loss_fn"]
        if loss_fn == "bell_l2_l1_loss":
            self.criterion = bell_l2_l1_loss
        else:
            self.criterion = torch.nn.L1Loss()

        self.metrics = config["tasks"]["app"]["metrics"]

        self.pretrain_epochs = int(config.get("pretrain_epochs", 10))
        self.lambda_cl = float(config.get("lambda_cl", 0.03))
        self.lambda_recovery = float(config.get("lambda_recovery", 0.01))
        self.use_pseudo_label_cl = bool(config.get("use_pseudo_label_cl", True))
        self.feat_eps = float(config.get("feat_eps", 0.01))
        self.temperature = float(config.get("temperature", 0.5))
        self.enable_recovery = bool(config.get("enable_recovery", True))
        self.recovery_start_epoch = config.get("recovery_start_epoch", 20)

        # Anti-collapse regression terms.
        # These are designed to reduce regression-to-mean and improve R2.
        self.lambda_var = float(config.get("lambda_var", 0.03))
        self.extreme_weight_strength = float(config.get("extreme_weight_strength", 1.5))
        self.lambda_extreme = float(config.get("lambda_extreme", 0.5))

        # Safer collection batch size for pseudo-label bank rebuilding.
        self.collect_batch_size = int(config.get("collect_batch_size", min(config.get("batch_size", 8), 4)))

        self.use_modality_gating = bool(config.get("use_modality_gating", True))
        self.use_quality_aware_gating = bool(config.get("use_quality_aware_gating", True))
        self.gating_entropy_weight = float(config.get("gating_entropy_weight", 0.0))
        self.quality_reg_weight = float(config.get("quality_reg_weight", 0.0))

        # Modality Token Fusion.
        # This replaces ModalityGating when enabled.
        self.use_modality_token_fusion = bool(config.get("use_modality_token_fusion", False))
        self.modality_token_fusion_layers = int(config.get("modality_token_fusion_layers", 1))
        self.modality_token_fusion_heads = int(config.get("modality_token_fusion_heads", 4))
        self.modality_token_fusion_dropout = float(config.get("modality_token_fusion_dropout", 0.1))
        self.modality_token_fusion_ffn_ratio = float(config.get("modality_token_fusion_ffn_ratio", 2.0))
        self.modality_token_fusion_residual = bool(config.get("modality_token_fusion_residual", False))
        self.modality_token_fusion_residual_alpha = float(
            config.get("modality_token_fusion_residual_alpha", 0.5)
        )
        self.use_modality_token_dropout = bool(config.get("use_modality_token_dropout", False))
        self.modality_token_dropout_prob = float(config.get("modality_token_dropout_prob", 0.05))
        self.use_modality_token_layernorm = bool(config.get("use_modality_token_layernorm", False))

        # Trait-conditioned Modality Selection (TCMS).
        # When enabled, MTF returns encoded modality tokens and TCMS creates
        # trait-specific hidden states [B, 5, D] before TraitSpecificRegressionHead.
        self.use_trait_conditioned_modality_selection = bool(
            config.get("use_trait_conditioned_modality_selection", False)
        )
        self.tcms_dropout = float(config.get("tcms_dropout", 0.1))
        self.tcms_use_global_context = bool(config.get("tcms_use_global_context", True))

        # Behavior-Reliability Summary Token (BRST).
        # Generates one extra behavior/reliability-guided token inside MTF.
        self.use_behavior_state_token = bool(config.get("use_behavior_state_token", False))
        self.behavior_state_hidden_ratio = float(config.get("behavior_state_hidden_ratio", 0.5))
        self.behavior_state_dropout = float(config.get("behavior_state_dropout", 0.1))
        self.behavior_state_use_stats = bool(config.get("behavior_state_use_stats", True))
        self.behavior_state_gate = bool(config.get("behavior_state_gate", True))
        self.behavior_state_gate_alpha = float(config.get("behavior_state_gate_alpha", 0.2))

        if self.use_modality_token_fusion and self.use_modality_gating:
            print("[INFO] use_modality_token_fusion=True, disable ModalityGating for clean ablation.")
            self.use_modality_gating = False

        # Method 1: Agreeableness-aware label contrastive learning.
        self.use_agreeableness_cl = bool(config.get("use_agreeableness_cl", True))
        self.lambda_agreeableness_cl = float(config.get("lambda_agreeableness_cl", 0.05))
        self.lambda_agreeableness_cl_recovery = float(
            config.get("lambda_agreeableness_cl_recovery", self.lambda_agreeableness_cl * 0.5)
        )
        self.agreeableness_index = int(config.get("agreeableness_index", 3))
        self.agreeableness_top_k = int(config.get("agreeableness_top_k", 3))
        self.agreeableness_temperature = float(config.get("agreeableness_temperature", 0.2))
        self.agreeableness_label_sigma = float(config.get("agreeableness_label_sigma", 0.08))
        # Optional: use both clean and adversarial representations for Agreeableness CL.
        # Default is False to preserve the original behavior unless explicitly enabled in YAML.
        self.agree_cl_use_clean_adv_pair = bool(config.get("agree_cl_use_clean_adv_pair", False))

        # Method 1.5: regression-aware ACL for continuous OCEAN labels.
        # This is a continuous-label replacement/supplement for pseudo-label CL.
        self.use_regression_aware_cl = bool(config.get("use_regression_aware_cl", True))
        self.lambda_regression_cl = float(config.get("lambda_regression_cl", 0.03))
        self.lambda_regression_cl_recovery = float(
            config.get("lambda_regression_cl_recovery", self.lambda_regression_cl * 0.5)
        )
        self.regression_cl_top_k = int(config.get("regression_cl_top_k", 3))
        self.regression_cl_temperature = float(config.get("regression_cl_temperature", 0.2))
        self.regression_cl_label_sigma = float(config.get("regression_cl_label_sigma", 0.08))
        self.regression_cl_use_clean_adv_pair = bool(config.get("regression_cl_use_clean_adv_pair", True))

        # Trait weights for continuous-label CL. Agreeableness keeps a slightly higher weight.
        trait_cl_weights = config.get("regression_cl_trait_weights", None)
        if trait_cl_weights is None:
            trait_cl_weights = [1.0, 1.0, 1.0, float(config.get("agreeableness_cl_weight", 1.5)), 1.0]
        self.regression_cl_trait_weights = torch.tensor(trait_cl_weights, dtype=torch.float32)

        # Behavior-aware adversarial regression contrastive learning.
        # This couples BRST behavior/reliability representations with clean/adv
        # contrastive learning. It uses continuous OCEAN label proximity and
        # behavior-state similarity to construct soft positive weights.
        self.use_behavior_aware_cl = bool(config.get("use_behavior_aware_cl", False))
        self.lambda_behavior_aware_cl = float(config.get("lambda_behavior_aware_cl", 0.03))
        self.lambda_behavior_aware_cl_recovery = float(
            config.get("lambda_behavior_aware_cl_recovery", self.lambda_behavior_aware_cl * 0.5)
        )
        self.behavior_cl_top_k = int(config.get("behavior_cl_top_k", self.regression_cl_top_k))
        self.behavior_cl_temperature = float(config.get("behavior_cl_temperature", self.regression_cl_temperature))
        self.behavior_cl_label_sigma = float(config.get("behavior_cl_label_sigma", self.regression_cl_label_sigma))
        self.behavior_cl_behavior_sigma = float(config.get("behavior_cl_behavior_sigma", 0.5))
        self.behavior_cl_behavior_weight_alpha = float(config.get("behavior_cl_behavior_weight_alpha", 0.5))
        self.behavior_cl_use_clean_adv_pair = bool(config.get("behavior_cl_use_clean_adv_pair", True))
        self.behavior_cl_replace_regression_cl = bool(config.get("behavior_cl_replace_regression_cl", True))
        self.last_behavior_repr = None

        # Optimization 1: automatic loss balancing.
        # Learnable log-variance terms softly balance task / pseudo-label CL /
        # regression-aware CL / Agreeableness CL, reducing manual lambda sensitivity.
        self.use_loss_auto_balance = bool(config.get("use_loss_auto_balance", True))
        self.loss_log_vars = torch.nn.Parameter(torch.zeros(5, dtype=torch.float32))

        # Optimization 2: Agreeableness-specific regression strengthening.
        # These terms directly target the weakest trait without changing data loading.
        self.lambda_agree_extreme = float(config.get("lambda_agree_extreme", 0.3))
        self.agree_extreme_weight_strength = float(config.get("agree_extreme_weight_strength", 2.0))
        self.lambda_agree_var = float(config.get("lambda_agree_var", 0.02))
        self.lambda_agree_rank = float(config.get("lambda_agree_rank", 0.02))
        self.agree_rank_margin = float(config.get("agree_rank_margin", 0.02))
        self.agree_rank_min_delta = float(config.get("agree_rank_min_delta", 0.03))

        # Optimization 3: training stability controls.
        # CL warmup prevents contrastive losses from dominating immediately after pretraining.
        self.cl_start_epoch = int(config.get("cl_start_epoch", self.pretrain_epochs))
        self.cl_warmup_epochs = int(config.get("cl_warmup_epochs", 5))
        self.adversarial_start_epoch = int(config.get("adversarial_start_epoch", self.pretrain_epochs))

        # Structural enhancement 1: Residual Prediction Head.
        # The model predicts a residual delta around the training-set label mean:
        #     y_pred = label_mean + delta
        # This directly targets the regression-to-the-mean behavior of FI labels.
        self.use_residual_prediction = bool(config.get("use_residual_prediction", False))
        label_mean = config.get("label_mean", [0.5, 0.5, 0.5, 0.5, 0.5])
        if isinstance(label_mean, torch.Tensor):
            label_mean_tensor = label_mean.detach().float().view(1, -1)
        else:
            label_mean_tensor = torch.tensor(label_mean, dtype=torch.float32).view(1, -1)
        if label_mean_tensor.numel() != 5:
            raise ValueError(f"label_mean must contain 5 values for OCEAN, got {label_mean}")
        self.register_buffer("label_mean", label_mean_tensor)

        # Trait-wise Residual Calibration.
        # This keeps the original residual prediction form, but lets each OCEAN
        # dimension learn its own residual magnitude:
        #     y_t = mean_t + scale_t * delta_t
        # It adds only 5 trainable parameters and does not introduce a new
        # fusion/prediction module.
        self.use_trait_residual_scale = bool(config.get("use_trait_residual_scale", False))
        trait_residual_scale_init = float(config.get("trait_residual_scale_init", 1.0))

        if self.use_trait_residual_scale:
            self.trait_residual_scale = torch.nn.Parameter(
                torch.ones(1, 5, dtype=torch.float32) * trait_residual_scale_init
            )
        else:
            self.trait_residual_scale = None

        # Structural enhancement 2: Trait-aware Fusion.
        # Create five trait-specific hidden representations before trait heads.
        self.use_trait_aware_fusion = bool(config.get("use_trait_aware_fusion", False))
        self.trait_fusion_hidden_dim = int(
            config.get("trait_fusion_hidden_dim", config.get("trait_head_input_dim", config.get("d_model", 40))))
        self.trait_fusion_dropout = float(config.get("trait_fusion_dropout", 0.1))

        # Method 2: decoupled OCEAN heads with a deeper Agreeableness head.
        self.use_trait_specific_heads = bool(config.get("use_trait_specific_heads", True))
        self.trait_loss_alpha_a = float(config.get("trait_loss_alpha_a", 1.5))
        self.trait_head_hidden_dim = int(config.get("trait_head_hidden_dim", config.get("d_model", 40)))
        self.trait_head_dropout = float(config.get("trait_head_dropout", 0.1))
        self.trait_head_use_sigmoid = bool(config.get("trait_head_use_sigmoid", False))

        if self.use_trait_specific_heads:
            output_dim = _resolve_output_dim(config, default=5)
            if output_dim != 5 or "target_id" in config:
                print("[WARNING] use_trait_specific_heads=True requires five-dimensional OCEAN output. Disabled.")
                self.use_trait_specific_heads = False
                self.trait_heads = None
            else:
                # IMPORTANT: LinMulT fused hidden dim may be larger than d_model
                # e.g. actual pooled_hidden can be 384 while d_model is 32.
                # train_model() auto-infers trait_head_input_dim before ModelWrapper is created.
                trait_head_input_dim = int(config.get("trait_head_input_dim", config.get("d_model", 40)))
                self.trait_heads = TraitSpecificRegressionHead(
                    input_dim=trait_head_input_dim,
                    hidden_dim=self.trait_head_hidden_dim,
                    dropout=self.trait_head_dropout,
                    use_sigmoid=self.trait_head_use_sigmoid,
                )
                if self.use_trait_aware_fusion:
                    self.trait_aware_fusion = TraitAwareFusion(
                        input_dim=trait_head_input_dim,
                        n_traits=5,
                        hidden_dim=self.trait_fusion_hidden_dim,
                        dropout=self.trait_fusion_dropout,
                    )
                else:
                    self.trait_aware_fusion = None
                print(f"[AutoInfer] trait_head_input_dim = {trait_head_input_dim}")
        else:
            self.trait_heads = None
            self.trait_aware_fusion = None

        input_dims = config.get("input_feature_dim", None)

        if self.use_modality_token_fusion:
            if input_dims is None:
                raise ValueError("input_feature_dim must be available when use_modality_token_fusion=True")

            trait_head_input_dim = int(config.get("trait_head_input_dim", config.get("d_model", 40)))
            self.modality_token_fusion = ModalityTokenFusion(
                input_dims=input_dims,
                fused_dim=trait_head_input_dim,
                num_layers=self.modality_token_fusion_layers,
                num_heads=self.modality_token_fusion_heads,
                dropout=self.modality_token_fusion_dropout,
                ffn_ratio=self.modality_token_fusion_ffn_ratio,
                use_residual=self.modality_token_fusion_residual,
                residual_alpha=self.modality_token_fusion_residual_alpha,
                use_token_dropout=self.use_modality_token_dropout,
                token_dropout_prob=self.modality_token_dropout_prob,
                use_token_layernorm=self.use_modality_token_layernorm,
                use_behavior_state_token=self.use_behavior_state_token,
                behavior_state_hidden_ratio=self.behavior_state_hidden_ratio,
                behavior_state_dropout=self.behavior_state_dropout,
                behavior_state_use_stats=self.behavior_state_use_stats,
                behavior_state_gate=self.behavior_state_gate,
                behavior_state_gate_alpha=self.behavior_state_gate_alpha,
            )
            print(
                f"[ModalityFusion] Using ModalityTokenFusion: "
                f"fused_dim={trait_head_input_dim}, "
                f"layers={self.modality_token_fusion_layers}, "
                f"heads={self.modality_token_fusion_heads}, "
                f"dropout={self.modality_token_fusion_dropout}, "
                f"residual={self.modality_token_fusion_residual}, "
                f"residual_alpha={self.modality_token_fusion_residual_alpha}, "
                f"token_dropout={self.use_modality_token_dropout}, "
                f"token_dropout_prob={self.modality_token_dropout_prob}, "
                f"token_layernorm={self.use_modality_token_layernorm}, "
                f"behavior_state_token={self.use_behavior_state_token}, "
                f"behavior_state_stats={self.behavior_state_use_stats}, "
                f"behavior_state_gate={self.behavior_state_gate}, "
                f"behavior_state_alpha={self.behavior_state_gate_alpha}"
            )
        else:
            self.modality_token_fusion = None

        if self.use_trait_conditioned_modality_selection:
            if not self.use_modality_token_fusion or self.modality_token_fusion is None:
                raise ValueError("use_trait_conditioned_modality_selection=True requires use_modality_token_fusion=True")
            if self.trait_heads is None:
                raise ValueError("use_trait_conditioned_modality_selection=True requires use_trait_specific_heads=True")
            trait_head_input_dim = int(config.get("trait_head_input_dim", config.get("d_model", 40)))
            self.trait_modality_selector = TraitConditionedModalitySelection(
                dim=trait_head_input_dim,
                n_traits=5,
                n_modalities=len(input_dims),
                dropout=self.tcms_dropout,
                use_global_context=self.tcms_use_global_context,
            )
            self.last_trait_modality_weights = None
            print(
                f"[TCMS] Using TraitConditionedModalitySelection: "
                f"dim={trait_head_input_dim}, modalities={len(input_dims)}, "
                f"dropout={self.tcms_dropout}, global_context={self.tcms_use_global_context}"
            )
        else:
            self.trait_modality_selector = None
            self.last_trait_modality_weights = None

        if self.use_modality_gating:
            if input_dims is None:
                raise ValueError("input_feature_dim must be available when use_modality_gating=True")

            if self.use_quality_aware_gating:
                self.modality_gating = QualityAwareModalityGating(input_dims=input_dims)
            else:
                self.modality_gating = ModalityGating(input_dims=input_dims)
        else:
            self.modality_gating = None

        self.pseudo_label_valid_racc_threshold = float(
            config.get("pseudo_label_valid_racc_threshold", 0.895)
        )
        self.pseudo_label_clusters = int(config.get("pseudo_label_clusters", 32))
        self.pseudo_label_ready = False

        self.train_feature_cache = []
        self.train_id_cache = []

        self.bank_features = None
        self.bank_labels = None
        self.bank_ids = None

        self.train_preds = []
        self.train_targets = []
        self.valid_preds = []
        self.valid_targets = []
        self.test_preds = []
        self.test_targets = []

        self.log_dir = Path(config["experiment_dir"])
        self.history = History(self.log_dir)

    def _decayed_lambda(self, base_value: float, start_epoch: int | None = None,
                        warmup_epochs: int | None = None) -> float:
        """带有 Warmup 和 Cosine Decay 的平滑调度器"""
        base_value = float(base_value)
        if base_value <= 0:
            return 0.0

        start_epoch = self.cl_start_epoch if start_epoch is None else int(start_epoch)
        warmup_epochs = self.cl_warmup_epochs if warmup_epochs is None else int(warmup_epochs)

        # 尚未开始
        if self.current_epoch < start_epoch:
            return 0.0

        # 1. Warmup Phase (线性预热)
        if self.current_epoch < start_epoch + warmup_epochs:
            progress = (self.current_epoch - start_epoch + 1) / float(max(1, warmup_epochs))
            return base_value * max(0.0, min(1.0, progress))

        # 2. Decay Phase (余弦衰减)
        max_epochs = getattr(self.trainer, "max_epochs", 40) or 40
        decay_epochs = max_epochs - (start_epoch + warmup_epochs)
        if decay_epochs <= 0:
            return base_value

        decay_progress = (self.current_epoch - (start_epoch + warmup_epochs)) / float(decay_epochs)

        # [修改点 3]: 加入 min_factor 保底，防止特征约束在最后彻底消失引发过拟合
        min_factor = 0.01
        decay_factor = min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * decay_progress))
        return base_value * decay_factor

    def _auto_balance_component(self, loss_value: torch.Tensor, component_idx: int) -> torch.Tensor:
        """Apply uncertainty-style automatic balancing to one loss component."""
        if not self.use_loss_auto_balance:
            return loss_value
        log_var = self.loss_log_vars[component_idx].to(device=loss_value.device, dtype=loss_value.dtype)
        return torch.exp(-log_var) * loss_value + log_var

    def _compose_total_loss(
            self,
            task_loss: torch.Tensor,
            cl_loss: torch.Tensor,
            reg_cl_loss: torch.Tensor,
            a_cl_loss: torch.Tensor,
            behavior_cl_loss: torch.Tensor,
            stage: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compose total loss with warmup and optional automatic loss balancing.

        Component order of learnable log vars:
        0: task regression,
        1: pseudo-label CL,
        2: regression-aware CL,
        3: A-CL,
        4: behavior-aware regression CL.
        """
        total = self._auto_balance_component(task_loss, 0)

        if stage == "acl":
            w_cl = self._decayed_lambda(self.lambda_cl)
            w_reg_cl = self._decayed_lambda(self.lambda_regression_cl)
            w_a_cl = self._decayed_lambda(self.lambda_agreeableness_cl)
            w_behavior_cl = self._decayed_lambda(self.lambda_behavior_aware_cl)
        elif stage == "recovery":
            w_cl = float(self.lambda_recovery)
            w_reg_cl = float(self.lambda_regression_cl_recovery)
            w_a_cl = float(self.lambda_agreeableness_cl_recovery)
            w_behavior_cl = float(self.lambda_behavior_aware_cl_recovery)
        else:
            w_cl = 0.0
            w_reg_cl = 0.0
            w_a_cl = 0.0
            w_behavior_cl = 0.0

        if w_cl > 0:
            total = total + w_cl * self._auto_balance_component(cl_loss, 1)
        if w_reg_cl > 0:
            total = total + w_reg_cl * self._auto_balance_component(reg_cl_loss, 2)
        if w_a_cl > 0:
            total = total + w_a_cl * self._auto_balance_component(a_cl_loss, 3)
        if w_behavior_cl > 0:
            total = total + w_behavior_cl * self._auto_balance_component(behavior_cl_loss, 4)

        weights = {
            "w_cl": float(w_cl),
            "w_regression_cl": float(w_reg_cl),
            "w_agreeableness_cl": float(w_a_cl),
            "w_behavior_aware_cl": float(w_behavior_cl),
        }
        return total, weights

    def _agreeableness_rank_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss for Agreeableness."""
        if pred.ndim != 2 or target.ndim != 2 or pred.size(0) <= 1:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if pred.size(1) <= self.agreeableness_index or target.size(1) <= self.agreeableness_index:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        a_pred = pred[:, self.agreeableness_index]
        a_target = target[:, self.agreeableness_index]
        target_diff = a_target.unsqueeze(1) - a_target.unsqueeze(0)
        pred_diff = a_pred.unsqueeze(1) - a_pred.unsqueeze(0)
        valid = torch.abs(target_diff) > self.agree_rank_min_delta
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        direction = torch.sign(target_diff.detach())
        rank_loss = F.relu(self.agree_rank_margin - direction * pred_diff)
        return rank_loss[valid].mean()

    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Main regression loss.

        When trait-specific heads are enabled, use explicit O/C/E/A/N decoupled loss:
            L = L_O + L_C + L_E + alpha_A * L_A + L_N

        This gives Agreeableness stronger supervision and reduces suppression by the
        shared five-dimensional prediction layer.
        """
        if (
                self.use_trait_specific_heads
                and pred.ndim == 2
                and target.ndim == 2
                and pred.size(1) == 5
                and target.size(1) == 5
        ):
            trait_loss = F.smooth_l1_loss(pred, target, reduction="none")
            weights = torch.ones(5, device=pred.device, dtype=pred.dtype)
            weights[self.agreeableness_index] = self.trait_loss_alpha_a
            base_loss = (trait_loss * weights.view(1, -1)).mean()
        else:
            base_loss = self.criterion(pred, target)

        loss = base_loss

        if self.extreme_weight_strength > 0 and self.lambda_extreme > 0:
            weight = 1.0 + self.extreme_weight_strength * torch.abs(target - 0.5)
            if weight.ndim == 2 and weight.size(1) > self.agreeableness_index:
                weight[:, self.agreeableness_index] = (
                        weight[:, self.agreeableness_index] * self.trait_loss_alpha_a
                )
            element_loss = F.smooth_l1_loss(pred, target, reduction="none")
            weighted_loss = (weight.detach() * element_loss).mean()
            loss = loss + self.lambda_extreme * weighted_loss

        if self.lambda_var > 0 and pred.size(0) > 1:
            pred_std = pred.std(dim=0, unbiased=False)
            target_std = target.std(dim=0, unbiased=False)
            var_loss = F.mse_loss(pred_std, target_std.detach())
            loss = loss + self.lambda_var * var_loss

        # Agreeableness-specific extreme-value learning.
        if (
                self.lambda_agree_extreme > 0
                and pred.ndim == 2
                and target.ndim == 2
                and pred.size(1) > self.agreeableness_index
                and target.size(1) > self.agreeableness_index
        ):
            a_pred = pred[:, self.agreeableness_index]
            a_target = target[:, self.agreeableness_index]
            a_weight = 1.0 + self.agree_extreme_weight_strength * torch.abs(a_target - 0.5)
            a_loss = F.smooth_l1_loss(a_pred, a_target, reduction="none")
            loss = loss + self.lambda_agree_extreme * (a_weight.detach() * a_loss).mean()

        # Agreeableness-specific variance matching to reduce A-dimension collapse.
        if (
                self.lambda_agree_var > 0
                and pred.ndim == 2
                and target.ndim == 2
                and pred.size(0) > 1
                and pred.size(1) > self.agreeableness_index
                and target.size(1) > self.agreeableness_index
        ):
            a_pred_std = pred[:, self.agreeableness_index].std(unbiased=False)
            a_target_std = target[:, self.agreeableness_index].std(unbiased=False).detach()
            loss = loss + self.lambda_agree_var * F.mse_loss(a_pred_std, a_target_std)

        if self.lambda_agree_rank > 0:
            loss = loss + self.lambda_agree_rank * self._agreeableness_rank_loss(pred, target)

        return loss

    def _aggregate_seq_output(self, pred: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if pred.ndim == 2:
            return pred

        if pred.ndim != 3:
            raise ValueError(f"Unexpected prediction ndim={pred.ndim}, shape={pred.shape}")

        if mask is None:
            return pred.mean(dim=1)

        mask = mask.bool()
        pred = pred.masked_fill(~mask.unsqueeze(-1), 0.0)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return pred.sum(dim=1) / valid_counts

    def _apply_residual_prediction(self, pred_delta: torch.Tensor) -> torch.Tensor:
        if not self.use_residual_prediction:
            return pred_delta

        if pred_delta.ndim != 2 or pred_delta.size(1) != 5:
            return pred_delta

        label_mean = self.label_mean.to(device=pred_delta.device, dtype=pred_delta.dtype)

        if self.use_trait_residual_scale and self.trait_residual_scale is not None:
            scale = self.trait_residual_scale.to(device=pred_delta.device, dtype=pred_delta.dtype)
            return label_mean + scale * pred_delta

        return label_mean + pred_delta

    def _predict_from_hidden(self, fused_repr, fused_mask):
        if self.use_trait_specific_heads and self.trait_heads is not None:
            pooled_hidden = self._aggregate_seq_output(fused_repr, fused_mask)
            if self.use_trait_aware_fusion and self.trait_aware_fusion is not None:
                pooled_hidden = self.trait_aware_fusion(pooled_hidden)
            pred_delta = self.trait_heads(pooled_hidden)
            return self._apply_residual_prediction(pred_delta)

        outputs = self.model.predict_from_hidden(fused_repr, fused_mask)
        pred_delta = self._aggregate_seq_output(outputs[0], fused_mask)
        return self._apply_residual_prediction(pred_delta)

    def _get_stage(self):
        if self.current_epoch < self.pretrain_epochs:
            return "pretrain"

        if self.enable_recovery and self.recovery_start_epoch is not None:
            if self.current_epoch >= int(self.recovery_start_epoch):
                return "recovery"

        return "acl"

    def _build_masks(self, batch, batch_idx=0, phase="train"):
        masks = []
        for feature_name in self.config["feature_list"]:
            mask_key = feature_name + "_mask"
            if mask_key in batch:
                masks.append(batch[mask_key])
            else:
                if batch_idx == 0:
                    print(f"[{phase}] {mask_key} not found, generating default mask")
                feat = batch[feature_name]
                default_mask = torch.ones(feat.shape[:2], dtype=torch.bool, device=feat.device)
                masks.append(default_mask)
        return masks

    def _pool_modality_feature(self, feat: torch.Tensor, mask: torch.Tensor | None):
        if mask is not None:
            mask = mask.bool()
            masked_feat = feat.masked_fill(~mask.unsqueeze(-1), 0.0)
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return masked_feat.sum(dim=1) / valid_counts
        return feat.mean(dim=1)

    def _concat_modal_features(self, x, mask):
        pooled = []
        for xi, mi in zip(x, mask):
            pooled.append(self._pool_modality_feature(xi, mi))
        return torch.cat(pooled, dim=1)

    def _cache_pseudo_features(self, x, mask, sample_ids):
        feat = self._concat_modal_features(x, mask).detach().cpu()
        self.train_feature_cache.append(feat)
        self.train_id_cache.extend([str(s) for s in sample_ids])

    def _collect_features_from_trainset(self):
        self.train_feature_cache = []
        self.train_id_cache = []

        dataset = self.trainer.datamodule.dataset_train
        loader = DataLoader(
            dataset,
            batch_size=self.collect_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )

        self.eval()
        with torch.no_grad():
            for batch in loader:
                x = [batch[f].to(self.device) for f in self.config["feature_list"]]
                masks = self._build_masks(batch, phase="collect")
                masks = [m.to(self.device) for m in masks]
                sample_ids = batch["sample_id"]

                feat = self._concat_modal_features(x, masks).detach().cpu()
                self.train_feature_cache.append(feat)
                self.train_id_cache.extend([str(s) for s in sample_ids])

        self.train()

        total_feats = sum(f.shape[0] for f in self.train_feature_cache)
        print(f"[INFO] Collected {total_feats} features for pseudo-labeling")

    def _build_pseudo_label_dictionary(self):
        if len(self.train_feature_cache) == 0:
            print("[INFO] No cached train features, skip pseudo-label dictionary.")
            return

        features = torch.cat(self.train_feature_cache, dim=0)
        features = F.normalize(features, dim=1)
        features_np = features.cpu().numpy()
        ids = list(self.train_id_cache)

        n_clusters = min(self.pseudo_label_clusters, max(2, len(ids) // 50))
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=int(self.config.get("seed", 42)),
            batch_size=256,
            n_init="auto",
        )
        labels = clusterer.fit_predict(features_np)

        self.bank_features = features.cpu().float()
        self.bank_labels = torch.from_numpy(labels).long()
        self.bank_ids = np.asarray(ids)
        self.pseudo_label_ready = True

        print(f"[INFO] Pseudo-label dictionary built: {len(ids)} samples, {len(set(labels))} clusters")

        self.train_feature_cache = []
        self.train_id_cache = []

    def _lookup_pseudo_labels(self, sample_ids, device):
        if self.bank_ids is None or self.bank_labels is None:
            return torch.full((len(sample_ids),), -1, dtype=torch.long, device=device)

        id_to_label = {
            sid: int(lbl)
            for sid, lbl in zip(self.bank_ids.tolist(), self.bank_labels.tolist())
        }
        labels = [id_to_label.get(str(sid), -1) for sid in sample_ids]
        return torch.tensor(labels, dtype=torch.long, device=device)

    def _feature_adv_perturbation_pre_fusion(self, x, mask, target):
        current_eps = self._decayed_lambda(self.feat_eps, start_epoch=self.adversarial_start_epoch, warmup_epochs=0)

        if current_eps <= 0:
            return [xi.detach() for xi in x]

        x_adv_seed = [xi.detach().clone().requires_grad_(True) for xi in x]

        with torch.enable_grad():
            y_pred_seed, _, _, _, gating_reg = self._predict_clean(x_adv_seed, mask)
            seed_loss = self._regression_loss(y_pred_seed, target) + gating_reg

        grads = torch.autograd.grad(
            seed_loss,
            x_adv_seed,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        x_adv = []
        for idx, (xi, gi) in enumerate(zip(x_adv_seed, grads)):
            if gi is not None:
                perturb = current_eps * torch.sign(gi.detach())

                # Only perturb valid timesteps. This avoids injecting adversarial noise into
                # padded positions for variable-length modality sequences.
                if mask is not None and idx < len(mask) and mask[idx] is not None:
                    mi = mask[idx].bool().unsqueeze(-1).to(device=perturb.device)
                    perturb = perturb * mi

                x_adv.append(xi.detach() + perturb)
            else:
                x_adv.append(xi.detach())

        return x_adv

    def _apply_modality_gating(self, x, mask, fused_repr):
        if not self.use_modality_gating:
            zero = torch.tensor(0.0, device=fused_repr.device)
            return fused_repr, None, zero

        if self.use_quality_aware_gating:
            weights, quality_scores = self.modality_gating(x, mask)
        else:
            weights = self.modality_gating(x, mask)
            quality_scores = None

        # Residual-style lightweight gating.
        n_modalities = weights.size(1)
        uniform_weight = 1.0 / float(max(n_modalities, 1))

        # confidence in [0, 1]; 0 means close to uniform, 1 means one modality dominates.
        max_weight = weights.max(dim=1, keepdim=True).values
        confidence = (max_weight - uniform_weight) / max(1.0 - uniform_weight, 1e-8)
        confidence = confidence.clamp(0.0, 1.0)

        scale = float(self.config.get("gating_residual_scale", 0.2))
        gating_strength = 1.0 + scale * confidence

        if fused_repr.ndim == 3:
            gated_fused_repr = fused_repr * gating_strength.unsqueeze(1)
        elif fused_repr.ndim == 2:
            gated_fused_repr = fused_repr * gating_strength
        else:
            raise ValueError(f"Unexpected fused_repr shape: {fused_repr.shape}")

        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()

        gating_reg = self.gating_entropy_weight * (-entropy)

        if quality_scores is not None and self.quality_reg_weight > 0:
            quality_probs = quality_scores / (quality_scores.sum(dim=1, keepdim=True) + 1e-8)
            quality_reg = F.mse_loss(weights, quality_probs.detach())
            gating_reg = gating_reg + self.quality_reg_weight * quality_reg

        return gated_fused_repr, weights, gating_reg

    def _predict_clean(self, x, mask):
        outputs, fused_repr, fused_mask = self.model(x, mask, return_hidden=True)

        if self.use_modality_token_fusion and self.modality_token_fusion is not None:
            if self.use_trait_conditioned_modality_selection and self.trait_modality_selector is not None:
                fused_hidden, encoded_modality_tokens = self.modality_token_fusion(
                    features=x,
                    masks=mask,
                    fused_repr=fused_repr,
                    fused_mask=fused_mask,
                    return_tokens=True,
                )
                self.last_behavior_repr = getattr(self.modality_token_fusion, "last_behavior_repr", None)
                trait_hidden, trait_modality_weights = self.trait_modality_selector(
                    modality_tokens=encoded_modality_tokens,
                    global_hidden=fused_hidden,
                )
                self.last_trait_modality_weights = trait_modality_weights.detach()
                gating_reg = torch.tensor(0.0, device=fused_hidden.device, dtype=fused_hidden.dtype)

                pred_delta = self.trait_heads(trait_hidden)
                y_pred = self._apply_residual_prediction(pred_delta)

                # Return fused_hidden for B-ARCL so the contrastive objective still
                # operates on a compact sample-level representation, while prediction
                # is made from trait-specific modality-selected hidden states.
                return y_pred, fused_hidden, None, trait_modality_weights, gating_reg

            fused_hidden = self.modality_token_fusion(
                features=x,
                masks=mask,
                fused_repr=fused_repr,
                fused_mask=fused_mask,
            )
            self.last_behavior_repr = getattr(self.modality_token_fusion, "last_behavior_repr", None)
            self.last_trait_modality_weights = None
            modality_weights = None
            gating_reg = torch.tensor(0.0, device=fused_hidden.device, dtype=fused_hidden.dtype)

            # fused_hidden is already [B, D], so no fused_mask is needed.
            y_pred = self._predict_from_hidden(fused_hidden, None)
            return y_pred, fused_hidden, None, modality_weights, gating_reg

        self.last_behavior_repr = None
        gated_fused_repr, modality_weights, gating_reg = self._apply_modality_gating(x, mask, fused_repr)
        y_pred = self._predict_from_hidden(gated_fused_repr, fused_mask)
        return y_pred, gated_fused_repr, fused_mask, modality_weights, gating_reg

    def forward(self, x, mask=None):
        y_pred, _, _, _, _ = self._predict_clean(x, mask)
        return y_pred

    def training_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config["feature_list"]]
        mask = self._build_masks(batch, batch_idx=batch_idx, phase="train")
        y_true = batch["app"]
        sample_ids = batch["sample_id"]

        y_pred_clean, clean_repr, _, modality_weights, gating_reg = self._predict_clean(x, mask)
        clean_behavior_repr = self.last_behavior_repr
        stage = self._get_stage()

        if stage == "pretrain":
            if self.use_pseudo_label_cl and self.current_epoch == self.pretrain_epochs - 1:
                self._cache_pseudo_features(x, mask, sample_ids)

            task_loss = self._regression_loss(y_pred_clean, y_true) + gating_reg
            cl_loss = torch.tensor(0.0, device=self.device)
            reg_cl_loss = torch.tensor(0.0, device=self.device)
            a_cl_loss = torch.tensor(0.0, device=self.device)
            behavior_cl_loss = torch.tensor(0.0, device=self.device)
            loss, effective_weights = self._compose_total_loss(
                task_loss, cl_loss, reg_cl_loss, a_cl_loss, behavior_cl_loss, stage="pretrain"
            )
            y_pred = y_pred_clean

        elif stage == "acl":
            if self.current_epoch >= self.adversarial_start_epoch:
                x_adv = self._feature_adv_perturbation_pre_fusion(x, mask, y_true)
                y_pred_adv, adv_repr, _, _, gating_reg_adv = self._predict_clean(x_adv, mask)
                adv_behavior_repr = self.last_behavior_repr
                task_loss = self._regression_loss(y_pred_adv, y_true) + gating_reg_adv
            else:
                x_adv = [xi.detach() for xi in x]
                y_pred_adv = y_pred_clean
                adv_repr = clean_repr
                adv_behavior_repr = clean_behavior_repr
                task_loss = self._regression_loss(y_pred_clean, y_true) + gating_reg

            if self.use_pseudo_label_cl and self.lambda_cl > 0:
                anchor_adv = self._concat_modal_features(x_adv, mask)
                anchor_labels = self._lookup_pseudo_labels(sample_ids, anchor_adv.device)

                cl_loss = pseudo_label_contrastive_loss(
                    anchor_features=anchor_adv,
                    anchor_ids=[str(s) for s in sample_ids],
                    anchor_labels=anchor_labels,
                    bank_features=self.bank_features,
                    bank_ids=self.bank_ids,
                    bank_labels=self.bank_labels,
                    temperature=self.temperature,
                )
            else:
                cl_loss = torch.tensor(0.0, device=self.device)

            if self.use_regression_aware_cl and self.lambda_regression_cl > 0:
                if self.regression_cl_use_clean_adv_pair:
                    reg_features = torch.cat([clean_repr.detach(), adv_repr], dim=0)
                    reg_labels = torch.cat([y_true, y_true], dim=0)
                else:
                    reg_features = adv_repr
                    reg_labels = y_true

                reg_cl_loss = regression_aware_contrastive_loss(
                    features=reg_features,
                    labels=reg_labels,
                    top_k=self.regression_cl_top_k,
                    temperature=self.regression_cl_temperature,
                    label_sigma=self.regression_cl_label_sigma,
                    trait_weights=self.regression_cl_trait_weights,
                )
            else:
                reg_cl_loss = torch.tensor(0.0, device=self.device)

            if self.use_behavior_aware_cl and self.lambda_behavior_aware_cl > 0 and clean_behavior_repr is not None:
                if self.behavior_cl_use_clean_adv_pair:
                    behavior_features = torch.cat([clean_repr.detach(), adv_repr], dim=0)
                    behavior_labels = torch.cat([y_true, y_true], dim=0)
                    if adv_behavior_repr is not None:
                        behavior_states = torch.cat([clean_behavior_repr.detach(), adv_behavior_repr], dim=0)
                    else:
                        behavior_states = torch.cat([clean_behavior_repr.detach(), clean_behavior_repr.detach()], dim=0)
                else:
                    behavior_features = adv_repr
                    behavior_labels = y_true
                    behavior_states = adv_behavior_repr if adv_behavior_repr is not None else clean_behavior_repr

                behavior_cl_loss = behavior_aware_regression_contrastive_loss(
                    features=behavior_features,
                    labels=behavior_labels,
                    behavior_repr=behavior_states,
                    top_k=self.behavior_cl_top_k,
                    temperature=self.behavior_cl_temperature,
                    label_sigma=self.behavior_cl_label_sigma,
                    behavior_sigma=self.behavior_cl_behavior_sigma,
                    behavior_weight_alpha=self.behavior_cl_behavior_weight_alpha,
                    trait_weights=self.regression_cl_trait_weights,
                )

                if self.behavior_cl_replace_regression_cl:
                    reg_cl_loss = torch.tensor(0.0, device=self.device)
            else:
                behavior_cl_loss = torch.tensor(0.0, device=self.device)

            if self.use_agreeableness_cl:
                if self.agree_cl_use_clean_adv_pair:
                    # Match regression-aware CL: align clean and adversarial representations
                    # under the same Agreeableness labels. clean_repr is detached so the extra
                    # A-CL gradient mainly regularizes the adversarial branch.
                    a_features = torch.cat([clean_repr.detach(), adv_repr], dim=0)
                    a_labels = torch.cat([y_true, y_true], dim=0)
                else:
                    a_features = adv_repr
                    a_labels = y_true

                a_cl_loss = agreeableness_label_contrastive_loss(
                    features=a_features,
                    labels=a_labels,
                    agreeableness_index=self.agreeableness_index,
                    top_k=self.agreeableness_top_k,
                    temperature=self.agreeableness_temperature,
                    label_sigma=self.agreeableness_label_sigma,
                )
            else:
                a_cl_loss = torch.tensor(0.0, device=self.device)

            loss, effective_weights = self._compose_total_loss(task_loss, cl_loss, reg_cl_loss, a_cl_loss, behavior_cl_loss, stage="acl")

            if self.current_epoch >= self.adversarial_start_epoch:
                y_pred = y_pred_adv
            else:
                y_pred = y_pred_clean

        else:
            task_loss = self._regression_loss(y_pred_clean, y_true) + gating_reg

            if self.use_pseudo_label_cl and self.lambda_recovery > 0:
                anchor_clean = self._concat_modal_features(x, mask)
                anchor_labels = self._lookup_pseudo_labels(sample_ids, anchor_clean.device)

                cl_loss = pseudo_label_contrastive_loss(
                    anchor_features=anchor_clean,
                    anchor_ids=[str(s) for s in sample_ids],
                    anchor_labels=anchor_labels,
                    bank_features=self.bank_features,
                    bank_ids=self.bank_ids,
                    bank_labels=self.bank_labels,
                    temperature=self.temperature,
                )
            else:
                cl_loss = torch.tensor(0.0, device=self.device)

            if self.use_regression_aware_cl and self.lambda_regression_cl_recovery > 0:
                reg_cl_loss = regression_aware_contrastive_loss(
                    features=clean_repr,
                    labels=y_true,
                    top_k=self.regression_cl_top_k,
                    temperature=self.regression_cl_temperature,
                    label_sigma=self.regression_cl_label_sigma,
                    trait_weights=self.regression_cl_trait_weights,
                )
            else:
                reg_cl_loss = torch.tensor(0.0, device=self.device)

            if self.use_behavior_aware_cl and self.lambda_behavior_aware_cl_recovery > 0 and clean_behavior_repr is not None:
                behavior_cl_loss = behavior_aware_regression_contrastive_loss(
                    features=clean_repr,
                    labels=y_true,
                    behavior_repr=clean_behavior_repr,
                    top_k=self.behavior_cl_top_k,
                    temperature=self.behavior_cl_temperature,
                    label_sigma=self.behavior_cl_label_sigma,
                    behavior_sigma=self.behavior_cl_behavior_sigma,
                    behavior_weight_alpha=self.behavior_cl_behavior_weight_alpha,
                    trait_weights=self.regression_cl_trait_weights,
                )
                if self.behavior_cl_replace_regression_cl:
                    reg_cl_loss = torch.tensor(0.0, device=self.device)
            else:
                behavior_cl_loss = torch.tensor(0.0, device=self.device)

            if self.use_agreeableness_cl:
                a_cl_loss = agreeableness_label_contrastive_loss(
                    features=clean_repr,
                    labels=y_true,
                    agreeableness_index=self.agreeableness_index,
                    top_k=self.agreeableness_top_k,
                    temperature=self.agreeableness_temperature,
                    label_sigma=self.agreeableness_label_sigma,
                )
            else:
                a_cl_loss = torch.tensor(0.0, device=self.device)

            loss, effective_weights = self._compose_total_loss(
                task_loss, cl_loss, reg_cl_loss, a_cl_loss, behavior_cl_loss, stage="recovery"
            )
            y_pred = y_pred_clean

        self.log("train_loss_step", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train_cl_loss_step", cl_loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("train_regression_cl_loss_step", reg_cl_loss, prog_bar=False, logger=True, on_step=True,
                 on_epoch=False)
        if "behavior_cl_loss" in locals():
            self.log(
                "train_behavior_aware_cl_loss_step",
                behavior_cl_loss,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
            )
        if "effective_weights" in locals():
            self.log("w_cl_step", effective_weights.get("w_cl", 0.0), prog_bar=False, logger=True, on_step=True,
                     on_epoch=False)
            self.log("w_regression_cl_step", effective_weights.get("w_regression_cl", 0.0), prog_bar=False, logger=True,
                     on_step=True, on_epoch=False)
            self.log("w_agreeableness_cl_step", effective_weights.get("w_agreeableness_cl", 0.0), prog_bar=False,
                     logger=True, on_step=True, on_epoch=False)
            self.log("w_behavior_aware_cl_step", effective_weights.get("w_behavior_aware_cl", 0.0), prog_bar=False,
                     logger=True, on_step=True, on_epoch=False)
        if self.use_loss_auto_balance:
            for i, name in enumerate(["task", "pseudo_cl", "regression_cl", "agreeableness_cl", "behavior_aware_cl"]):
                self.log(f"loss_log_var_{name}", self.loss_log_vars[i].detach(), prog_bar=False, logger=True,
                         on_step=False, on_epoch=True)
        if "a_cl_loss" in locals():
            self.log(
                "train_agreeableness_cl_loss_step",
                a_cl_loss,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
            )

        if batch_idx == 0 and modality_weights is not None:
            feature_names = self.config["feature_list"]
            weights_cpu = modality_weights.detach().cpu()

            # Normal modality gating returns [B, M], while TCMS returns [B, 5, M].
            # The old logger assumed [B, M] and crashed when TCMS produced a
            # trait-by-modality matrix. Keep this logging robust for both cases.
            if weights_cpu.ndim == 3:
                trait_names = ["O", "C", "E", "A", "N"]
                mean_weights = weights_cpu.mean(dim=0)  # [5, M]
                pretty = {}
                for trait_idx, trait_name in enumerate(trait_names[:mean_weights.size(0)]):
                    pretty[trait_name] = {
                        name: round(float(w), 4)
                        for name, w in zip(feature_names, mean_weights[trait_idx].tolist())
                    }
                print(f"[TCMS] mean trait-modality weights = {pretty}")
            else:
                mean_weights = weights_cpu.mean(dim=0).tolist()
                pretty = {name: round(float(w), 4) for name, w in zip(feature_names, mean_weights)}
                gate_name = "QualityAwareGating" if self.use_quality_aware_gating else "ModalityGating"
                print(f"[{gate_name}] mean weights = {pretty}")

        self.train_preds.append(y_pred.detach())
        self.train_targets.append(y_true.detach())
        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_preds, dim=0)
        targets = torch.cat(self.train_targets, dim=0)

        preds = torch.clamp(preds, min=0, max=1)

        preds_np = preds.cpu().detach().numpy()
        targets_np = targets.cpu().detach().numpy()

        metrics = calculate_app_metrics(preds_np, targets_np, self.config)
        for metric_name, metric_value in metrics.items():
            self.history.update(
                phase="train",
                task="app",
                metric=metric_name,
                value=metric_value,
                epoch=self.current_epoch,
            )

            if metric_name in self.metrics:
                self.log(f"train_{metric_name}", metric_value, prog_bar=True, logger=True, on_epoch=True)
                self.history.plot("app", metric_name, "racc")

        avg_loss = self.trainer.logged_metrics["train_loss_step"]
        self.history.update(
            phase="train",
            task="all",
            metric="avg_loss",
            value=avg_loss.item(),
            epoch=self.current_epoch,
        )
        self.log("train_loss", avg_loss.item(), prog_bar=True, logger=True, on_epoch=True)
        self.history.plot("all", "avg_loss")
        self.history.save()

        self.train_preds = []
        self.train_targets = []

    def validation_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config["feature_list"]]
        mask = self._build_masks(batch, batch_idx=batch_idx, phase="valid")
        y_true = batch["app"]

        y_pred, _, _, _, _ = self._predict_clean(x, mask)

        loss = self.criterion(y_pred, y_true)

        self.log("valid_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.valid_preds.append(y_pred.detach())
        self.valid_targets.append(y_true.detach())
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.valid_preds, dim=0)
        targets = torch.cat(self.valid_targets, dim=0)

        preds = torch.clamp(preds, min=0, max=1)

        preds_np = preds.cpu().detach().numpy()
        targets_np = targets.cpu().detach().numpy()

        metrics = calculate_app_metrics(preds_np, targets_np, self.config)
        for metric_name, metric_value in metrics.items():
            self.history.update(
                phase="valid",
                task="app",
                metric=metric_name,
                value=metric_value,
                epoch=self.current_epoch,
            )

            if metric_name in self.metrics:
                self.log(f"valid_{metric_name}", metric_value, prog_bar=True, logger=True, on_epoch=True)

        avg_loss = self.trainer.logged_metrics["valid_loss"]
        self.history.update(
            phase="valid",
            task="all",
            metric="avg_loss",
            value=avg_loss.item(),
            epoch=self.current_epoch,
        )
        self.log("valid_loss", avg_loss.item(), prog_bar=True, logger=True, on_epoch=True)

        should_build_bank = (
                self.use_pseudo_label_cl
                and (not self.pseudo_label_ready)
                and (self.current_epoch == self.pretrain_epochs - 1)
        )

        if should_build_bank:
            print("[INFO] Building pseudo-label dictionary ...")

            if len(self.train_feature_cache) == 0:
                print("[WARNING] Cache empty, rebuilding from one pass...")
                self._collect_features_from_trainset()

            self._build_pseudo_label_dictionary()

        self.valid_preds = []
        self.valid_targets = []

    def test_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config["feature_list"]]
        mask = self._build_masks(batch, batch_idx=batch_idx, phase="test")
        y_true = batch["app"]

        y_pred, _, _, _, _ = self._predict_clean(x, mask)

        self.test_preds.append(y_pred.detach())
        self.test_targets.append(y_true.detach())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds, dim=0)
        targets = torch.cat(self.test_targets, dim=0)

        preds = torch.clamp(preds, min=0, max=1)

        preds_np = preds.cpu().detach().numpy()
        targets_np = targets.cpu().detach().numpy()

        metrics = calculate_app_metrics(
            preds_np,
            targets_np,
            self.config,
            output_path=self.log_dir / "metrics_test.csv",
        )
        for metric_name, metric_value in metrics.items():
            self.history.update(
                phase="test",
                task="app",
                metric=metric_name,
                value=metric_value,
                epoch=self.current_epoch,
            )

            if metric_name in self.metrics:
                self.log(f"test_{metric_name}", metric_value, prog_bar=True, logger=True, on_epoch=True)

        self.history.save_test()

        if "target_id" in self.config:
            assert preds_np.shape[1] == 1 and targets_np.shape[1] == 1
            df = pd.DataFrame({"Prediction": preds_np[:, 0], "GroundTruth": targets_np[:, 0]})
            pred_file = self.log_dir / f'test_predictions_{self.config["target_id"]}.csv'
            df.to_csv(pred_file, index=False)
            print(f"Test predictions saved to {str(pred_file)}")
        else:
            assert preds_np.shape[1] == 5 and targets_np.shape[1] == 5
            for trait_id in range(5):
                df = pd.DataFrame(
                    {"Prediction": preds_np[:, trait_id], "GroundTruth": targets_np[:, trait_id]}
                )
                pred_file = self.log_dir / f"test_predictions_{trait_id}.csv"
                df.to_csv(pred_file, index=False)
                print(f"Test predictions saved to {str(pred_file)}")

        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer_config = self.config.get("optimizer", {})
        optimizer_name = optimizer_config.get("name", "adam")
        base_lr = float(optimizer_config.get("base_lr", 1e-3))
        weight_decay = float(optimizer_config.get("weight_decay", 0))

        if optimizer_name == "radam":
            optimizer = torch.optim.RAdam(
                self.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
                decoupled_weight_decay=optimizer_config.get("decoupled_weight_decay", False),
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
            )

        lr_scheduler_config = self.config.get("lr_scheduler", {})
        lr_scheduler_name = lr_scheduler_config.get("name", "ReduceLROnPlateau")
        warmup_steps = int(self.trainer.estimated_stepping_batches * self.config.get("warmup_ratio", 0.0))

        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(lr_scheduler_config.get("start_factor", 0.1)),
                total_iters=warmup_steps,
            )
        else:
            warmup_scheduler = None

        if lr_scheduler_name == "ReduceLROnPlateau":
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(lr_scheduler_config.get("factor", 0.1)),
                patience=int(lr_scheduler_config.get("patience", 5)),
                min_lr=1e-8,
            )

            if warmup_steps > 0:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": torch.optim.lr_scheduler.SequentialLR(
                            optimizer,
                            schedulers=[warmup_scheduler, main_scheduler],
                            milestones=[warmup_steps],
                        ),
                        "monitor": lr_scheduler_config.get("monitor", "valid_loss"),
                    },
                }
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": main_scheduler,
                    "monitor": lr_scheduler_config.get("monitor", "valid_loss"),
                },
            }

        if lr_scheduler_name == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=base_lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }

        if lr_scheduler_name == "CosineAnnealingLR":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config["n_epochs"],
            )
        else:
            raise ValueError(f"Given lr scheduler is not supported: {lr_scheduler_name}")

        if warmup_steps > 0:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
        else:
            lr_scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def train_model(config: dict):
    print("CONFIG KEYS =", config.keys())
    print("FULL CONFIG =", config)

    seed_everything(config.get("seed", 42))
    torch.set_float32_matmul_precision(config.get("float32_matmul_precision", "medium"))

    experiment_name = config.get("experiment_name", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if "target_id" in config:
        experiment_dir = Path(config.get("output_dir", "results")) / experiment_name / "OCEAN"[config["target_id"]]
    else:
        experiment_dir = Path(config.get("output_dir", "results")) / experiment_name

    if experiment_dir.exists() and "test_only" not in config:
        raise ValueError(f"Experiment with {experiment_name} already exists. Skip.")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    config["experiment_dir"] = str(experiment_dir)
    config["num_workers"] = int(config.get("num_workers", 0))

    # ================= [绝对防御版代码备份机制] =================
    backup_dir = experiment_dir / "code_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    import inspect
    import os

    try:
        # 1. 备份当前主脚本 (train_app.py)
        shutil.copy(os.path.abspath(__file__), backup_dir / "train_app.py")

        # 2. 顺藤摸瓜：直接获取目前真正在内存里运行的模型文件物理路径
        linmult_path = inspect.getfile(LinMulT)
        shutil.copy(linmult_path, backup_dir / "LinMulT.py")

        # 3. 顺藤摸瓜：获取数据加载文件的路径
        fi_path = inspect.getfile(FiDataModule)
        shutil.copy(fi_path, backup_dir / "fi.py")

        # 4. 备份本次运行的 YAML 配置文件
        for path_key in ["db_config_path", "model_config_path", "train_config_path"]:
            if path_key in config:
                cfg_path = Path(config[path_key])
                if cfg_path.exists():
                    shutil.copy(cfg_path, backup_dir / cfg_path.name)

        print(f"[INFO] 绝对防御备份成功！真实模型代码来源: {linmult_path}")
    except Exception as e:
        print(f"[WARNING] 备份过程中出现小问题，但不影响训练: {e}")
    # =================================================================

    data_module = FiDataModule(config=config)
    data_module.setup(stage="fit")

    train_loader = data_module.train_dataloader()
    sample_batch = next(iter(train_loader))

    if bool(config.get("use_residual_prediction", False)) and bool(config.get("auto_label_mean", True)):
        if "label_mean" not in config or config.get("label_mean") in (None, "auto"):
            label_mean = _infer_label_mean_from_loader(train_loader)
            config["label_mean"] = [float(v) for v in label_mean]
            print(f"[AutoInfer] label_mean = {[round(v, 6) for v in config['label_mean']]}")

    feature_list = config["feature_list"]
    input_feature_dim = [int(sample_batch[feature_name].shape[-1]) for feature_name in feature_list]
    config["input_feature_dim"] = input_feature_dim
    print(f"[AutoInfer] input_feature_dim = {input_feature_dim}")

    model = LinMulT(config=config)

    if bool(config.get("use_trait_specific_heads", False)):
        model.eval()
        with torch.no_grad():
            sample_x = [sample_batch[feature_name] for feature_name in feature_list]
            sample_masks = []
            for feature_name in feature_list:
                mask_key = feature_name + "_mask"
                if mask_key in sample_batch:
                    sample_masks.append(sample_batch[mask_key])
                else:
                    feat = sample_batch[feature_name]
                    sample_masks.append(torch.ones(feat.shape[:2], dtype=torch.bool, device=feat.device))

            _, sample_fused_repr, _ = model(sample_x, sample_masks, return_hidden=True)
            if sample_fused_repr.ndim in (2, 3):
                config["trait_head_input_dim"] = int(sample_fused_repr.shape[-1])
            else:
                raise ValueError(f"Unexpected fused representation shape: {sample_fused_repr.shape}")
            print(f"[AutoInfer] trait_head_input_dim = {config['trait_head_input_dim']}")
        model.train()

    lightning_model = ModelWrapper(model, config=config)

    callbacks = []
    for checkpoint_config in config["checkpoints"]:
        callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=experiment_dir / "checkpoint",
            filename=f"{checkpoint_config['name']}",
            monitor=checkpoint_config["monitor"],
            mode=checkpoint_config["mode"],
            save_top_k=1,
            verbose=True,
            save_weights_only=True,
        )
        callbacks.append(callback)

    config_es = config.get("early_stopping", False)
    if config_es:
        early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor=config_es["monitor"],
            patience=config_es["patience"],
            mode=config_es["mode"],
            verbose=True,
        )
        callbacks.append(early_stopping)

    time_tracker = TimeTrackingCallback(experiment_dir)
    callbacks.append(time_tracker)

    csv_logger = L.pytorch.loggers.CSVLogger(save_dir=str(experiment_dir), name="csv_logs")

    trainer = L.Trainer(
        accelerator=config.get("accelerator", "gpu"),
        devices=config.get("devices", [0]),
        max_epochs=config.get("n_epochs", 20),
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=csv_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    if "test_only" not in config:
        if "cp_path" in config:
            trainer.fit(lightning_model, datamodule=data_module, ckpt_path=config["cp_path"])
        else:
            trainer.fit(lightning_model, datamodule=data_module)

    print("Evaluating on the test set...")
    if "cp_path" in config:
        checkpoint_path = config["cp_path"]
        print(f"Loading model from: {checkpoint_path}")
    else:
        preferred_monitor = str(config.get("test_checkpoint_monitor", "valid_r2"))
        fallback_path = ""
        preferred_path = ""
        for cb in callbacks:
            if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint) and cb.best_model_path:
                if not fallback_path:
                    fallback_path = cb.best_model_path
                if preferred_monitor in cb.best_model_path:
                    preferred_path = cb.best_model_path
                    break
        checkpoint_path = preferred_path or fallback_path
        print(f"Loading best model from: {checkpoint_path}")

    if not checkpoint_path:
        print("No checkpoint found, skip test stage.")
        return

    lightning_model = ModelWrapper.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        config=config,
        map_location=torch.device(f'cuda:{config.get("gpu_id", 0)}'),
    )
    trainer.test(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    config: dict = argparser()
    train_model(config)