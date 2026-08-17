"""Minimal public implementation of the formal high-capacity Full branch.

This file keeps only the path enabled in the locked seed-42 Full experiment:
LinMulT -> MTF(+BRST) -> TCMS -> trait-specific residual prediction,
plus the training-only B-ARCL and Agreeableness-aware contrastive objectives.

The optional/disabled historical ablation branches from the original train_app.py
are intentionally omitted. Module/parameter names on the active prediction path
are kept compatible with the archived Lightning checkpoint where practical.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    from linmult import LinMulT as _LinMulTBase
except ImportError as exc:  # dependency error is surfaced on construction
    _LinMulTBase = None
    _LINMULT_IMPORT_ERROR = exc
else:
    _LINMULT_IMPORT_ERROR = None


class LinMulTBackbone(_LinMulTBase if _LinMulTBase is not None else nn.Module):
    """LinMulT 1.5.2 plus the hidden-state interface used by the formal Full run."""

    def __init__(self, config: dict[str, Any]):
        if _LinMulTBase is None:
            raise ImportError(
                "The Full model requires linmult>=1.5,<1.6. Install requirements.txt first."
            ) from _LINMULT_IMPORT_ERROR
        super().__init__(config)

    def forward_features(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.BoolTensor] | None = None,
        names: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if masks is None:
            masks = [None] * self.n_sequences
        projected_inputs = self._apply_projections(inputs, names)
        projected_inputs, masks = self._apply_multimodal_signal(projected_inputs, masks)
        branch_representations = self._apply_branch(projected_inputs, masks)
        return self._apply_fusion(branch_representations, masks)

    def forward(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.BoolTensor] | None = None,
        names: list[str] | None = None,
        return_hidden: bool = False,
    ):
        fused_representation, mask = self.forward_features(inputs, masks, names)
        outputs = self._apply_output_heads(fused_representation, mask)
        if return_hidden:
            return outputs, fused_representation, mask
        return outputs


TRAITS = ("O", "C", "E", "A", "N")


def agreeableness_label_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    agreeableness_index: int = 3,
    top_k: int = 3,
    temperature: float = 0.2,
    label_sigma: float = 0.08,
) -> torch.Tensor:
    """Continuous-label contrastive loss used for Agreeableness."""
    if features is None or labels is None or features.size(0) <= 1:
        device = features.device if features is not None else labels.device
        return torch.tensor(0.0, device=device)
    if labels.ndim != 2 or labels.size(1) <= agreeableness_index:
        return torch.tensor(0.0, device=features.device)
    if features.ndim == 3:
        features = features.mean(dim=1)

    features = F.normalize(features, dim=1)
    a_labels = labels[:, agreeableness_index].view(-1, 1)
    sim = features @ features.T / max(float(temperature), 1e-8)
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
    pos_weight = torch.exp(-dist / max(float(label_sigma), 1e-8)) * pos_mask.float()

    exp_sim = torch.exp(sim) * (~eye).float()
    numerator = (exp_sim * pos_weight).sum(dim=1)
    denominator = exp_sim.sum(dim=1).clamp_min(1e-8)
    valid = numerator > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    return -torch.log((numerator[valid] / denominator[valid]).clamp_min(1e-8)).mean()


def behavior_aware_regression_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    behavior_repr: torch.Tensor,
    top_k: int = 3,
    temperature: float = 0.2,
    label_sigma: float = 0.08,
    behavior_sigma: float = 0.5,
    behavior_weight_alpha: float = 0.5,
    trait_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """B-ARCL: soft positives from OCEAN proximity modulated by BRST similarity."""
    if features is None or labels is None or behavior_repr is None or features.size(0) <= 1:
        device = features.device if features is not None else labels.device
        return torch.tensor(0.0, device=device)
    if features.ndim == 3:
        features = features.mean(dim=1)
    if behavior_repr.ndim == 3:
        behavior_repr = behavior_repr.mean(dim=1)
    if labels.ndim != 2 or behavior_repr.ndim != 2:
        return torch.tensor(0.0, device=features.device)
    if features.size(0) != labels.size(0) or features.size(0) != behavior_repr.size(0):
        return torch.tensor(0.0, device=features.device)

    batch_size = features.size(0)
    features = F.normalize(features, dim=1)
    behavior_repr = F.normalize(behavior_repr.detach(), dim=1)
    labels = labels.detach()

    sim = features @ features.T / max(float(temperature), 1e-8)
    eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    sim = sim.masked_fill(eye, -1e9)

    behavior_sim = (behavior_repr @ behavior_repr.T).clamp(-1.0, 1.0)
    behavior_dist = (1.0 - behavior_sim).clamp_min(0.0)
    behavior_weight = torch.exp(-behavior_dist / max(float(behavior_sigma), 1e-8))
    behavior_weight = behavior_weight.masked_fill(eye, 0.0)
    alpha = max(0.0, min(1.0, float(behavior_weight_alpha)))

    n_traits = labels.size(1)
    if trait_weights is None or trait_weights.numel() != n_traits:
        trait_weights = torch.ones(n_traits, device=features.device, dtype=features.dtype)
    else:
        trait_weights = trait_weights.to(device=features.device, dtype=features.dtype)

    k = min(int(top_k), batch_size - 1)
    if k <= 0:
        return torch.tensor(0.0, device=features.device)

    pos_weight = torch.zeros(batch_size, batch_size, device=features.device, dtype=features.dtype)
    for trait_idx in range(n_traits):
        y = labels[:, trait_idx].view(-1, 1)
        dist = torch.abs(y - y.T).masked_fill(eye, 1e9)
        pos_indices = torch.topk(dist, k=k, largest=False, dim=1).indices
        pos_mask = torch.zeros_like(dist, dtype=torch.bool)
        pos_mask.scatter_(1, pos_indices, True)
        label_weight = torch.exp(-dist / max(float(label_sigma), 1e-8)) * pos_mask.float()
        behavior_modulator = (1.0 - alpha) + alpha * behavior_weight
        pos_weight += trait_weights[trait_idx] * label_weight * behavior_modulator

    pos_weight = pos_weight / pos_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
    exp_sim = torch.exp(sim) * (~eye).float()
    numerator = (exp_sim * pos_weight).sum(dim=1)
    denominator = exp_sim.sum(dim=1).clamp_min(1e-8)
    valid = numerator > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    return -torch.log((numerator[valid] / denominator[valid]).clamp_min(1e-8)).mean()


class TraitSpecificRegressionHead(nn.Module):
    """O/C/E/N: 384->32->1; Agreeableness: 384->32->32->1."""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList()
        for trait_idx in range(5):
            if trait_idx == 3:
                self.heads.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                ))
            else:
                self.heads.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                ))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            if hidden.size(1) != 5:
                raise ValueError(f"Expected [B,5,D], got {tuple(hidden.shape)}")
            outputs = [head(hidden[:, i, :]) for i, head in enumerate(self.heads)]
        elif hidden.ndim == 2:
            outputs = [head(hidden) for head in self.heads]
        else:
            raise ValueError(f"Expected [B,D] or [B,5,D], got {tuple(hidden.shape)}")
        return torch.cat(outputs, dim=1)


class TraitConditionedModalitySelection(nn.Module):
    """Full TCMS: q_(b,t)=q_t + W_g h_b, followed by attention over four MTF tokens."""
    def __init__(self, dim: int = 384, n_traits: int = 5, n_modalities: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.n_traits = int(n_traits)
        self.n_modalities = int(n_modalities)
        self.trait_queries = nn.Parameter(torch.randn(self.n_traits, self.dim) * 0.02)
        self.query_proj = nn.Linear(self.dim, self.dim)
        self.key_proj = nn.Linear(self.dim, self.dim)
        self.value_proj = nn.Linear(self.dim, self.dim)
        self.global_proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(float(dropout))
        self.out_norm = nn.LayerNorm(self.dim)
        self.last_trait_modality_weights = None

    def forward(self, modality_tokens: torch.Tensor, global_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if modality_tokens.ndim != 3:
            raise ValueError(f"Expected modality_tokens [B,M,D], got {tuple(modality_tokens.shape)}")
        batch_size, n_modalities, dim = modality_tokens.shape
        if n_modalities != self.n_modalities or dim != self.dim:
            raise ValueError(f"Expected [B,{self.n_modalities},{self.dim}], got {tuple(modality_tokens.shape)}")
        if global_hidden.ndim != 2 or global_hidden.size(-1) != self.dim:
            raise ValueError(f"Expected global_hidden [B,{self.dim}], got {tuple(global_hidden.shape)}")

        trait_queries = self.trait_queries.unsqueeze(0).expand(batch_size, -1, -1)
        trait_queries = trait_queries + self.global_proj(global_hidden).unsqueeze(1)
        q = self.query_proj(trait_queries)
        k = self.key_proj(modality_tokens)
        v = self.value_proj(modality_tokens)
        attn_logits = q @ k.transpose(1, 2) / math.sqrt(max(dim, 1))
        attn_weights = self.dropout(torch.softmax(attn_logits, dim=-1))
        trait_hidden = self.out_norm(attn_weights @ v)
        self.last_trait_modality_weights = attn_weights.detach()
        return trait_hidden, attn_weights


class BehaviorReliabilitySummaryToken(nn.Module):
    """BRST used by the formal Full model (stats enabled, gated residual token)."""
    def __init__(
        self,
        n_modalities: int = 4,
        fused_dim: int = 384,
        hidden_ratio: float = 1.0,
        dropout: float = 0.1,
        gate_alpha: float = 0.2,
    ):
        super().__init__()
        self.n_modalities = int(n_modalities)
        self.fused_dim = int(fused_dim)
        self.gate_alpha = float(gate_alpha)
        hidden_dim = max(32, int(self.fused_dim * float(hidden_ratio)))
        n_pairs = self.n_modalities * (self.n_modalities - 1) // 2
        self.stat_dim = self.n_modalities * 5 + n_pairs
        self.stat_encoder = nn.Sequential(
            nn.LayerNorm(self.stat_dim), nn.Linear(self.stat_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, self.fused_dim),
        )
        self.summary_encoder = nn.Sequential(
            nn.LayerNorm(self.fused_dim), nn.Linear(self.fused_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, self.fused_dim), nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.LayerNorm(self.fused_dim), nn.Linear(self.fused_dim, self.fused_dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(self.fused_dim)

    @staticmethod
    def _masked_raw_pool(feat: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            return feat.mean(dim=1), torch.ones(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
        mask = mask.bool().to(device=feat.device)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        valid_ratio = valid_counts.to(dtype=feat.dtype) / max(int(mask.size(1)), 1)
        pooled = feat.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / valid_counts.to(dtype=feat.dtype)
        return pooled, valid_ratio

    def _build_stats(self, features: list[torch.Tensor], masks: list[torch.Tensor] | None, modality_tokens: torch.Tensor) -> torch.Tensor:
        stats: list[torch.Tensor] = []
        for i, feat in enumerate(features):
            mask_i = masks[i] if masks is not None and i < len(masks) else None
            pooled_raw, valid_ratio = self._masked_raw_pool(feat, mask_i)
            if mask_i is not None:
                mask_bool = mask_i.bool().to(device=feat.device)
                valid_counts = mask_bool.sum(dim=1).clamp(min=1).to(dtype=feat.dtype)
                centered = (feat - pooled_raw.unsqueeze(1)).masked_fill(~mask_bool.unsqueeze(-1), 0.0)
                temporal_var = (
                    centered.pow(2).sum(dim=(1, 2))
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

        norm_tokens = F.normalize(modality_tokens, dim=-1, eps=1e-6)
        for i in range(self.n_modalities):
            for j in range(i + 1, self.n_modalities):
                cos_ij = (norm_tokens[:, i, :] * norm_tokens[:, j, :]).sum(dim=-1, keepdim=True)
                stats.append(1.0 - cos_ij)
        return torch.cat(stats, dim=1)

    def forward(self, features: list[torch.Tensor], masks: list[torch.Tensor] | None, modality_tokens: torch.Tensor) -> torch.Tensor:
        mean_token = modality_tokens.mean(dim=1)
        stats = self._build_stats(features, masks, modality_tokens)
        stat_emb = self.stat_encoder(stats.to(dtype=modality_tokens.dtype))
        candidate = self.summary_encoder(mean_token + stat_emb)
        gate = self.gate(stat_emb)
        token = mean_token + self.gate_alpha * gate * candidate
        return self.out_norm(token).unsqueeze(1)


class ModalityTokenFusion(nn.Module):
    """Formal MTF path: 1 global + 4 modality + 1 BRST token, one Transformer layer."""
    def __init__(
        self,
        input_dims: list[int],
        fused_dim: int = 384,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_ratio: float = 2.0,
        behavior_hidden_ratio: float = 1.0,
        behavior_dropout: float = 0.1,
        behavior_gate_alpha: float = 0.2,
    ):
        super().__init__()
        self.input_dims = list(input_dims)
        self.n_modalities = len(self.input_dims)
        self.fused_dim = int(fused_dim)
        self.last_behavior_repr = None
        if self.fused_dim % int(num_heads) != 0:
            raise ValueError("fused_dim must be divisible by num_heads for the formal Full configuration")

        self.modality_projs = nn.ModuleList([nn.Linear(int(d), self.fused_dim) for d in self.input_dims])
        self.modality_token_norms = nn.ModuleList([nn.LayerNorm(self.fused_dim) for _ in self.input_dims])
        self.fused_token_norm = nn.LayerNorm(self.fused_dim)
        self.token_type_embedding = nn.Parameter(torch.randn(self.n_modalities + 2, self.fused_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.fused_dim,
            nhead=int(num_heads),
            dim_feedforward=max(self.fused_dim, int(self.fused_dim * float(ffn_ratio))),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(self.fused_dim)
        self.behavior_state_generator = BehaviorReliabilitySummaryToken(
            n_modalities=self.n_modalities,
            fused_dim=self.fused_dim,
            hidden_ratio=behavior_hidden_ratio,
            dropout=behavior_dropout,
            gate_alpha=behavior_gate_alpha,
        )

    @staticmethod
    def _masked_pool(feat: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return feat.mean(dim=1)
        mask = mask.bool().to(device=feat.device)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return feat.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / valid_counts

    @staticmethod
    def _pool_fused_repr(fused_repr: torch.Tensor, fused_mask: torch.Tensor | None) -> torch.Tensor:
        if fused_repr.ndim == 2:
            return fused_repr
        if fused_mask is None:
            return fused_repr.mean(dim=1)
        fused_mask = fused_mask.bool().to(device=fused_repr.device)
        valid_counts = fused_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return fused_repr.masked_fill(~fused_mask.unsqueeze(-1), 0.0).sum(dim=1) / valid_counts

    def forward(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor] | None,
        fused_repr: torch.Tensor,
        fused_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modality_tokens = []
        for i, feat in enumerate(features):
            mask_i = masks[i] if masks is not None and i < len(masks) else None
            token = self.modality_projs[i](self._masked_pool(feat, mask_i))
            modality_tokens.append(self.modality_token_norms[i](token).unsqueeze(1))
        modality_tokens = torch.cat(modality_tokens, dim=1)

        fused_token_value = self.fused_token_norm(self._pool_fused_repr(fused_repr, fused_mask))
        fused_token = fused_token_value.unsqueeze(1)
        behavior_token = self.behavior_state_generator(features, masks, modality_tokens)
        tokens = torch.cat([fused_token, modality_tokens, behavior_token], dim=1)
        tokens = tokens + self.token_type_embedding.unsqueeze(0).to(device=tokens.device, dtype=tokens.dtype)
        tokens = self.encoder(tokens)

        self.last_behavior_repr = tokens[:, -1, :]
        fused_hidden = self.out_norm(tokens[:, 0, :])
        encoded_modality_tokens = tokens[:, 1:1 + self.n_modalities, :]
        return fused_hidden, encoded_modality_tokens


class FullModel(nn.Module):
    """Locked high-capacity Full architecture used as the paper's Full endpoint/teacher."""
    def __init__(self, config: dict[str, Any], backbone: nn.Module | None = None):
        super().__init__()
        self.config = dict(config)
        if backbone is None:
            backbone = LinMulTBackbone(self.config)
        self.model = backbone

        input_dims = [int(v) for v in self.config["input_feature_dim"]]
        hidden_dim = int(self.config.get("trait_head_input_dim", 384))
        if hidden_dim != 384:
            raise ValueError("The formal Full configuration uses trait_head_input_dim=384.")

        label_mean = torch.tensor(self.config["label_mean"], dtype=torch.float32).view(1, 5)
        self.register_buffer("label_mean", label_mean)
        self.trait_residual_scale = nn.Parameter(
            torch.ones(1, 5, dtype=torch.float32) * float(self.config.get("trait_residual_scale_init", 1.0))
        )

        self.trait_heads = TraitSpecificRegressionHead(
            input_dim=hidden_dim,
            hidden_dim=int(self.config.get("trait_head_hidden_dim", 32)),
            dropout=float(self.config.get("trait_head_dropout", 0.1)),
        )
        self.modality_token_fusion = ModalityTokenFusion(
            input_dims=input_dims,
            fused_dim=hidden_dim,
            num_layers=int(self.config.get("modality_token_fusion_layers", 1)),
            num_heads=int(self.config.get("modality_token_fusion_heads", 4)),
            dropout=float(self.config.get("modality_token_fusion_dropout", 0.1)),
            ffn_ratio=float(self.config.get("modality_token_fusion_ffn_ratio", 2.0)),
            behavior_hidden_ratio=float(self.config.get("behavior_state_hidden_ratio", 1.0)),
            behavior_dropout=float(self.config.get("behavior_state_dropout", 0.1)),
            behavior_gate_alpha=float(self.config.get("behavior_state_gate_alpha", 0.2)),
        )
        self.trait_modality_selector = TraitConditionedModalitySelection(
            dim=hidden_dim,
            n_traits=5,
            n_modalities=len(input_dims),
            dropout=float(self.config.get("tcms_dropout", 0.1)),
        )

        # The archived ModelWrapper created this trainable tensor even though automatic
        # loss balancing was disabled. Keeping it preserves formal checkpoint/state-dict
        # compatibility and the exact archived parameter count; it is never used below.
        self.loss_log_vars = nn.Parameter(torch.zeros(5, dtype=torch.float32))

        self.last_behavior_repr: torch.Tensor | None = None
        self.last_trait_modality_weights: torch.Tensor | None = None

        # Active training constants from the locked Full run.
        self.trait_loss_alpha_a = float(self.config.get("trait_loss_alpha_a", 1.5))
        self.agreeableness_index = int(self.config.get("agreeableness_index", 3))
        self.lambda_extreme = float(self.config.get("lambda_extreme", 0.05))
        self.extreme_weight_strength = float(self.config.get("extreme_weight_strength", 1.0))
        self.lambda_var = float(self.config.get("lambda_var", 0.005))
        self.pretrain_epochs = int(self.config.get("pretrain_epochs", 5))
        self.cl_start_epoch = int(self.config.get("cl_start_epoch", 5))
        self.cl_warmup_epochs = int(self.config.get("cl_warmup_epochs", 5))
        self.adversarial_start_epoch = int(self.config.get("adversarial_start_epoch", 10))
        self.feat_eps = float(self.config.get("feat_eps", 0.005))
        self.lambda_agreeableness_cl = float(self.config.get("lambda_agreeableness_cl", 0.05))
        self.lambda_behavior_aware_cl = float(self.config.get("lambda_behavior_aware_cl", 0.03))
        self.regression_cl_trait_weights = torch.tensor(
            self.config.get("regression_cl_trait_weights", [1.0, 1.1, 1.0, 1.5, 1.2]),
            dtype=torch.float32,
        )

    def _predict_clean(
        self, features: list[torch.Tensor], masks: list[torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, fused_repr, fused_mask = self.model(features, masks, return_hidden=True)
        fused_hidden, modality_tokens = self.modality_token_fusion(
            features=features, masks=masks, fused_repr=fused_repr, fused_mask=fused_mask
        )
        behavior_repr = self.modality_token_fusion.last_behavior_repr
        trait_hidden, weights = self.trait_modality_selector(modality_tokens, fused_hidden)
        pred_delta = self.trait_heads(trait_hidden)
        pred = self.label_mean.to(pred_delta) + self.trait_residual_scale.to(pred_delta) * pred_delta
        self.last_behavior_repr = behavior_repr
        self.last_trait_modality_weights = weights.detach()
        return pred, fused_hidden, weights, behavior_repr

    def forward(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor] | None = None,
        return_aux: bool = False,
    ):
        pred, hidden, weights, behavior = self._predict_clean(features, masks)
        if return_aux:
            return {
                "prediction": pred,
                "global_hidden": hidden,
                "trait_modality_weights": weights,
                "behavior_repr": behavior,
            }
        return pred

    def regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        trait_loss = F.smooth_l1_loss(pred, target, reduction="none")
        weights = torch.ones(5, device=pred.device, dtype=pred.dtype)
        weights[self.agreeableness_index] = self.trait_loss_alpha_a
        loss = (trait_loss * weights.view(1, -1)).mean()

        if self.extreme_weight_strength > 0 and self.lambda_extreme > 0:
            extreme_weight = 1.0 + self.extreme_weight_strength * torch.abs(target - 0.5)
            extreme_weight[:, self.agreeableness_index] *= self.trait_loss_alpha_a
            weighted = (extreme_weight.detach() * trait_loss).mean()
            loss = loss + self.lambda_extreme * weighted

        if self.lambda_var > 0 and pred.size(0) > 1:
            pred_std = pred.std(dim=0, unbiased=False)
            target_std = target.std(dim=0, unbiased=False)
            loss = loss + self.lambda_var * F.mse_loss(pred_std, target_std.detach())
        return loss

    @staticmethod
    def _scheduled_weight(base: float, epoch: int, max_epochs: int, start: int, warmup: int) -> float:
        base = float(base)
        if base <= 0 or epoch < start:
            return 0.0
        if epoch < start + warmup:
            progress = (epoch - start + 1) / float(max(1, warmup))
            return base * max(0.0, min(1.0, progress))
        decay_epochs = max_epochs - (start + warmup)
        if decay_epochs <= 0:
            return base
        progress = (epoch - (start + warmup)) / float(decay_epochs)
        min_factor = 0.01
        factor = min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
        return base * factor

    def _adversarial_features(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor] | None,
        target: torch.Tensor,
        epoch: int,
        max_epochs: int,
    ) -> list[torch.Tensor]:
        eps = self._scheduled_weight(self.feat_eps, epoch, max_epochs, self.adversarial_start_epoch, 0)
        if eps <= 0:
            return [x.detach() for x in features]

        seeds = [x.detach().clone().requires_grad_(True) for x in features]
        with torch.enable_grad():
            pred, _, _, _ = self._predict_clean(seeds, masks)
            seed_loss = self.regression_loss(pred, target)
        grads = torch.autograd.grad(seed_loss, seeds, retain_graph=False, create_graph=False, allow_unused=True)

        adv = []
        for i, (x, grad) in enumerate(zip(seeds, grads)):
            if grad is None:
                adv.append(x.detach())
                continue
            perturb = eps * torch.sign(grad.detach())
            if masks is not None and i < len(masks) and masks[i] is not None:
                perturb = perturb * masks[i].bool().unsqueeze(-1).to(perturb.device)
            adv.append(x.detach() + perturb)
        return adv

    def compute_training_objective(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor] | None,
        target: torch.Tensor,
        epoch: int,
        max_epochs: int | None = None,
    ) -> dict[str, torch.Tensor | float]:
        """Numerically matches the active Full training path (no disabled ablation branches)."""
        max_epochs = int(max_epochs or self.config.get("n_epochs", 100))
        clean_pred, clean_repr, _, clean_behavior = self._predict_clean(features, masks)

        if epoch < self.pretrain_epochs:
            task = self.regression_loss(clean_pred, target)
            zero = torch.tensor(0.0, device=target.device)
            return {
                "loss": task,
                "prediction": clean_pred,
                "task_loss": task,
                "agreeableness_cl_loss": zero,
                "behavior_aware_cl_loss": zero,
                "w_agreeableness_cl": 0.0,
                "w_behavior_aware_cl": 0.0,
            }

        if epoch >= self.adversarial_start_epoch:
            adv_features = self._adversarial_features(features, masks, target, epoch, max_epochs)
            adv_pred, adv_repr, _, adv_behavior = self._predict_clean(adv_features, masks)
            task = self.regression_loss(adv_pred, target)
            prediction = adv_pred
        else:
            adv_repr = clean_repr
            adv_behavior = clean_behavior
            task = self.regression_loss(clean_pred, target)
            prediction = clean_pred

        behavior_features = torch.cat([clean_repr.detach(), adv_repr], dim=0)
        behavior_labels = torch.cat([target, target], dim=0)
        if adv_behavior is not None:
            behavior_states = torch.cat([clean_behavior.detach(), adv_behavior], dim=0)
        else:
            behavior_states = torch.cat([clean_behavior.detach(), clean_behavior.detach()], dim=0)

        b_arcl = behavior_aware_regression_contrastive_loss(
            features=behavior_features,
            labels=behavior_labels,
            behavior_repr=behavior_states,
            top_k=int(self.config.get("behavior_cl_top_k", 3)),
            temperature=float(self.config.get("behavior_cl_temperature", 0.2)),
            label_sigma=float(self.config.get("behavior_cl_label_sigma", 0.08)),
            behavior_sigma=float(self.config.get("behavior_cl_behavior_sigma", 0.5)),
            behavior_weight_alpha=float(self.config.get("behavior_cl_behavior_weight_alpha", 0.5)),
            trait_weights=self.regression_cl_trait_weights,
        )
        a_cl = agreeableness_label_contrastive_loss(
            features=adv_repr,
            labels=target,
            agreeableness_index=self.agreeableness_index,
            top_k=int(self.config.get("agreeableness_top_k", 3)),
            temperature=float(self.config.get("agreeableness_temperature", 0.2)),
            label_sigma=float(self.config.get("agreeableness_label_sigma", 0.08)),
        )

        w_a = self._scheduled_weight(
            self.lambda_agreeableness_cl, epoch, max_epochs, self.cl_start_epoch, self.cl_warmup_epochs
        )
        w_b = self._scheduled_weight(
            self.lambda_behavior_aware_cl, epoch, max_epochs, self.cl_start_epoch, self.cl_warmup_epochs
        )
        loss = task + w_a * a_cl + w_b * b_arcl
        return {
            "loss": loss,
            "prediction": prediction,
            "task_loss": task,
            "agreeableness_cl_loss": a_cl,
            "behavior_aware_cl_loss": b_arcl,
            "w_agreeableness_cl": w_a,
            "w_behavior_aware_cl": w_b,
        }


EXPECTED_FULL_TRAINABLE_PARAMETERS = 5_476_680


def assert_full_parameter_count(model: nn.Module) -> int:
    """Verify the formal seed-42 Full neural parameter count."""
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != EXPECTED_FULL_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"Full parameter mismatch: expected {EXPECTED_FULL_TRAINABLE_PARAMETERS:,}, got {count:,}. "
            "Check linmult>=1.5,<1.6 and full_config.yaml."
        )
    return count


def load_lightning_checkpoint(model: FullModel, checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> FullModel:
    """Load the archived Lightning checkpoint into the minimal public model."""
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # older PyTorch
        payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    return model
