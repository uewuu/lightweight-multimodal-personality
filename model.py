"""Lightweight multimodal personality prediction model.

Architecture:

    projected concat + MLP
        -> Trait-Interactive TCMS-Lite
        -> trait-specific regression heads
        -> residual prediction

The model contains 500,107 trainable parameters. Five ``loss_log_vars``
parameters are retained for checkpoint compatibility; automatic loss balancing
is disabled by the configuration used in the experiments.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


TRAITS = ("O", "C", "E", "A", "N")
DEFAULT_FEATURES = ("dinov2_face", "wavlm", "roberta", "egemaps_lld")
DEFAULT_INPUT_DIMS = (384, 768, 1024, 25)
EXPECTED_TRAINABLE_PARAMETERS = 500_107


class ProjectedConcatMLPFusion(nn.Module):
    """Mask-aware projected concatenation followed by a lightweight MLP.

    Each modality sequence is mean pooled over valid timesteps, LayerNorm'ed,
    projected to ``projection_dim``, and concatenated. The projected vectors are
    also returned as modality tokens for Trait-Interactive TCMS-Lite.
    """

    def __init__(
        self,
        input_dims: list[int] | tuple[int, ...],
        projection_dim: int = 96,
        hidden_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("ProjectedConcatMLPFusion requires at least one modality.")

        self.input_dims = [int(dim) for dim in input_dims]
        self.n_modalities = len(self.input_dims)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)

        self.modality_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for dim in self.input_dims]
        )
        self.modality_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, self.projection_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                )
                for dim in self.input_dims
            ]
        )

        concat_dim = self.n_modalities * self.projection_dim
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.hidden_dim),
        )

    @staticmethod
    def _masked_mean_pool(
        feat: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if feat.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(feat.shape)}")
        if mask is None:
            return feat.mean(dim=1)

        mask = mask.bool().to(device=feat.device)
        if mask.ndim != 2 or mask.shape[:2] != feat.shape[:2]:
            raise ValueError(
                f"Mask {tuple(mask.shape)} does not match feature {tuple(feat.shape)}"
            )
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_feat = feat.masked_fill(~mask.unsqueeze(-1), 0.0)
        return masked_feat.sum(dim=1) / valid_counts.to(dtype=feat.dtype)

    def forward(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor] | None = None,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(features) != self.n_modalities:
            raise ValueError(
                f"Expected {self.n_modalities} modalities, got {len(features)}"
            )
        if masks is not None and len(masks) != self.n_modalities:
            raise ValueError(
                f"Expected {self.n_modalities} masks, got {len(masks)}"
            )

        projected: list[torch.Tensor] = []
        for idx, feat in enumerate(features):
            if feat.size(-1) != self.input_dims[idx]:
                raise ValueError(
                    f"Modality {idx} expects dim={self.input_dims[idx]}, "
                    f"got {feat.size(-1)}"
                )
            mask_i = masks[idx] if masks is not None else None
            pooled = self._masked_mean_pool(feat, mask_i)
            pooled = self.modality_norms[idx](pooled)
            projected.append(self.modality_projs[idx](pooled))

        modality_tokens = torch.stack(projected, dim=1)
        fused_hidden = self.fusion_mlp(torch.cat(projected, dim=1))

        if return_tokens:
            return fused_hidden, modality_tokens
        return fused_hidden


class TraitInteractiveTCMSLite(nn.Module):
    """Trait-Interactive TCMS-Lite used by the final lightweight student.

    A low-rank sample x trait interaction corrects each learned trait query before
    attending to the projected modality tokens. The returned attention weights
    have shape [B, 5, M].
    """

    def __init__(
        self,
        token_dim: int,
        output_dim: int,
        n_traits: int = 5,
        n_modalities: int = 4,
        dropout: float = 0.1,
        residual_alpha: float = 0.2,
        learnable_alpha: bool = True,
        interaction_rank: int = 8,
        interaction_scale_init: float = 0.1,
        interaction_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.n_traits = int(n_traits)
        self.n_modalities = int(n_modalities)
        self.learnable_alpha = bool(learnable_alpha)
        self.interaction_rank = int(interaction_rank)

        if self.token_dim <= 0 or self.output_dim <= 0:
            raise ValueError("token_dim and output_dim must be positive")
        if self.n_traits <= 0 or self.n_modalities <= 0:
            raise ValueError("n_traits and n_modalities must be positive")
        if self.interaction_rank <= 0:
            raise ValueError("interaction_rank must be positive")
        if not 0.0 < float(residual_alpha) < 1.0:
            raise ValueError("residual_alpha must be in (0,1)")
        if not 0.0 < float(interaction_scale_init) < 1.0:
            raise ValueError("interaction_scale_init must be in (0,1)")

        self.trait_embeddings = nn.Parameter(
            torch.randn(self.n_traits, self.token_dim) * 0.02
        )
        self.query_proj = nn.Linear(self.token_dim, self.token_dim)
        self.key_proj = nn.Linear(self.token_dim, self.token_dim)
        self.value_proj = nn.Linear(self.token_dim, self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.output_dim)

        self.global_norm = nn.LayerNorm(self.output_dim)
        self.sample_factor = nn.Linear(self.output_dim, self.interaction_rank)
        self.trait_factor = nn.Linear(
            self.token_dim, self.interaction_rank, bias=False
        )
        self.interaction_to_query = nn.Linear(
            self.interaction_rank, self.token_dim, bias=False
        )
        self.interaction_dropout = nn.Dropout(float(interaction_dropout))

        self.attn_dropout = nn.Dropout(float(dropout))
        self.out_dropout = nn.Dropout(float(dropout))
        self.out_norm = nn.LayerNorm(self.output_dim)

        alpha_logit = math.log(float(residual_alpha) / (1.0 - float(residual_alpha)))
        alpha_tensor = torch.tensor(alpha_logit, dtype=torch.float32)
        if self.learnable_alpha:
            self.residual_alpha_logit = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("residual_alpha_logit", alpha_tensor)

        scale_logit = math.log(
            float(interaction_scale_init) / (1.0 - float(interaction_scale_init))
        )
        self.interaction_scale_logit = nn.Parameter(
            torch.tensor(scale_logit, dtype=torch.float32)
        )
        self.last_trait_modality_weights: torch.Tensor | None = None

    def _residual_alpha(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.sigmoid(
            self.residual_alpha_logit.to(device=device, dtype=dtype)
        )

    def _interaction_scale(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.interaction_scale_logit.to(device=device, dtype=dtype)
        )

    def forward(
        self,
        modality_tokens: torch.Tensor,
        global_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modality_tokens.ndim != 3:
            raise ValueError(
                f"Expected modality_tokens [B,M,D], got {tuple(modality_tokens.shape)}"
            )
        if global_hidden.ndim != 2:
            raise ValueError(
                f"Expected global_hidden [B,H], got {tuple(global_hidden.shape)}"
            )

        batch_size, n_modalities, token_dim = modality_tokens.shape
        if n_modalities != self.n_modalities or token_dim != self.token_dim:
            raise ValueError(
                f"Expected [B,{self.n_modalities},{self.token_dim}], "
                f"got {tuple(modality_tokens.shape)}"
            )
        if global_hidden.shape != (batch_size, self.output_dim):
            raise ValueError(
                f"Expected global_hidden [{batch_size},{self.output_dim}], "
                f"got {tuple(global_hidden.shape)}"
            )

        base_trait = self.trait_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        sample_factor = torch.tanh(
            self.sample_factor(self.global_norm(global_hidden))
        )
        trait_factor = torch.tanh(self.trait_factor(self.trait_embeddings))
        interaction = sample_factor.unsqueeze(1) * trait_factor.unsqueeze(0)
        interaction = self.interaction_dropout(interaction)
        query_delta = self.interaction_to_query(interaction)

        beta = self._interaction_scale(global_hidden.dtype, global_hidden.device)
        queries = self.query_proj(base_trait + beta * query_delta)
        keys = self.key_proj(modality_tokens)
        values = self.value_proj(modality_tokens)

        logits = torch.matmul(queries, keys.transpose(1, 2))
        logits = logits / math.sqrt(max(self.token_dim, 1))
        attn_weights = torch.softmax(logits, dim=-1)

        trait_context = torch.matmul(self.attn_dropout(attn_weights), values)
        trait_delta = self.out_dropout(self.out_proj(trait_context))
        alpha = self._residual_alpha(global_hidden.dtype, global_hidden.device)
        trait_hidden = self.out_norm(
            global_hidden.unsqueeze(1) + alpha * trait_delta
        )

        self.last_trait_modality_weights = attn_weights.detach()
        return trait_hidden, attn_weights


# Compatibility alias for the class name used by the training code.
TraitConditionedModalitySelectionLite = TraitInteractiveTCMSLite


class TraitSpecificRegressionHead(nn.Module):
    """Five decoupled OCEAN regression heads; A uses a deeper head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or input_dim)
        self.use_sigmoid = bool(use_sigmoid)
        self.heads = nn.ModuleList()

        for trait_idx in range(5):
            if trait_idx == 3:
                self.heads.append(
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                )
            else:
                self.heads.append(
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        if hidden.ndim == 3:
            if hidden.size(1) != len(self.heads):
                raise ValueError(f"Expected [B,5,D], got {tuple(hidden.shape)}")
            for trait_idx, head in enumerate(self.heads):
                outputs.append(head(hidden[:, trait_idx, :]))
        else:
            for head in self.heads:
                outputs.append(head(hidden))

        pred = torch.cat(outputs, dim=1)
        if self.use_sigmoid:
            pred = torch.sigmoid(pred)
        return pred


class FinalStudent(nn.Module):
    """Inference-time architecture of the 500,107-parameter student."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        cfg = dict(config)
        input_dims = [int(x) for x in cfg.get("input_feature_dim", DEFAULT_INPUT_DIMS)]
        if len(input_dims) != 4:
            raise ValueError("The student expects exactly four modalities")

        projection_dim = int(cfg.get("concat_projection_dim", 96))
        hidden_dim = int(cfg.get("concat_hidden_dim", 384))
        tcms_dim = int(cfg.get("concat_tcms_lite_dim", projection_dim))
        if tcms_dim != projection_dim:
            raise ValueError("concat_tcms_lite_dim must equal concat_projection_dim")

        self.concat_mlp_fusion = ProjectedConcatMLPFusion(
            input_dims=input_dims,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout=float(cfg.get("concat_dropout", 0.1)),
        )
        self.concat_tcms_lite = TraitInteractiveTCMSLite(
            token_dim=tcms_dim,
            output_dim=hidden_dim,
            n_traits=5,
            n_modalities=len(input_dims),
            dropout=float(cfg.get("concat_tcms_lite_dropout", 0.1)),
            residual_alpha=float(cfg.get("concat_tcms_lite_residual_alpha", 0.2)),
            learnable_alpha=bool(cfg.get("concat_tcms_lite_learnable_alpha", True)),
            interaction_rank=int(cfg.get("concat_tcms_interaction_rank", 8)),
            interaction_scale_init=float(
                cfg.get("concat_tcms_interaction_scale_init", 0.1)
            ),
            interaction_dropout=float(
                cfg.get(
                    "concat_tcms_interaction_dropout",
                    cfg.get("concat_tcms_lite_dropout", 0.1),
                )
            ),
        )
        self.trait_heads = TraitSpecificRegressionHead(
            input_dim=hidden_dim,
            hidden_dim=int(cfg.get("trait_head_hidden_dim", 32)),
            dropout=float(cfg.get("trait_head_dropout", 0.1)),
            use_sigmoid=bool(cfg.get("trait_head_use_sigmoid", False)),
        )

        label_mean = torch.tensor(
            cfg.get("label_mean", [0.5] * 5), dtype=torch.float32
        ).view(1, -1)
        if label_mean.numel() != 5:
            raise ValueError("label_mean must contain five OCEAN values")
        self.register_buffer("label_mean", label_mean)

        if not bool(cfg.get("use_residual_prediction", True)):
            raise ValueError("Residual prediction must be enabled for this model")
        if not bool(cfg.get("use_trait_residual_scale", True)):
            raise ValueError("Trait residual scaling must be enabled for this model")

        self.trait_residual_scale = nn.Parameter(
            torch.ones(1, 5, dtype=torch.float32)
            * float(cfg.get("trait_residual_scale_init", 1.0))
        )

        # Retained for compatibility with checkpoints from the experiment code.
        self.loss_log_vars = nn.Parameter(torch.zeros(5, dtype=torch.float32))

    def forward(
        self,
        features: list[torch.Tensor],
        masks: list[torch.Tensor] | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused_hidden, modality_tokens = self.concat_mlp_fusion(
            features, masks, return_tokens=True
        )
        trait_hidden, trait_modality_weights = self.concat_tcms_lite(
            modality_tokens=modality_tokens,
            global_hidden=fused_hidden,
        )
        pred_delta = self.trait_heads(trait_hidden)
        scale = self.trait_residual_scale.to(
            device=pred_delta.device, dtype=pred_delta.dtype
        )
        label_mean = self.label_mean.to(
            device=pred_delta.device, dtype=pred_delta.dtype
        )
        prediction = label_mean + scale * pred_delta

        if return_aux:
            return prediction, fused_hidden, trait_modality_weights
        return prediction

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a state dict and normalize an optional ``model.`` prefix."""
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized[key[6:] if key.startswith("model.") else key] = value
        self.load_state_dict(normalized, strict=True)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_parameter_count(model: nn.Module) -> int:
    count = count_trainable_parameters(model)
    if count != EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(
            f"Expected {EXPECTED_TRAINABLE_PARAMETERS:,} trainable parameters, "
            f"got {count:,}"
        )
    return count
