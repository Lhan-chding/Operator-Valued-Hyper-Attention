from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList

from ..operators.generic_seed import (
    GenericDenseSeedPredictor, matched_generic_hidden_dim)
from ..operators.base import SeedResult, memory_valid_mask
from ..operators.rqgo import RQGO
from ..role_encoder import LatentRoleEncoder
from .deterministic_grounding_dino import DeterministicGroundingDINO


@MODELS.register_module()
class OVHAGroundingDINO(DeterministicGroundingDINO):
    """MM-Grounding-DINO with a Phase-1 dense seed operator.

    The parent token-maximum selection score remains the sole base score.  A
    zero-initialized RQGO or generic control may add a bounded bias before
    Top-K.  Dense proposal tensors are retained only for training losses and
    diagnostics; inference never reads annotations from ``batch_data_samples``.
    """

    _SEED_OPERATORS = frozenset({"none", "rqgo", "generic"})

    def __init__(
        self,
        *args,
        seed_operator: Optional[str] = None,
        seed_operator_cfg: Optional[Dict] = None,
        role_encoder_cfg: Optional[Dict] = None,
        ovha_cfg: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        phase_cfg = dict(ovha_cfg or {})
        operator_name = str(
            seed_operator or phase_cfg.get("variant", "rqgo")
        ).lower()
        if operator_name not in self._SEED_OPERATORS:
            choices = ", ".join(sorted(self._SEED_OPERATORS))
            raise ValueError(
                f"seed_operator must be one of {{{choices}}}, got {seed_operator!r}"
            )
        self.seed_operator_name = operator_name
        branch_cfg = dict(
            phase_cfg.get(
                "generic_seed" if operator_name == "generic" else "rqgo", {}
            )
        )
        auxiliary_head_cfg = {
            "loss_ref_weight": phase_cfg.get("loss_ref_weight"),
            "loss_role_div_weight": phase_cfg.get("loss_role_div_weight"),
            "loss_seed_weight": (
                0.0
                if operator_name == "none"
                else branch_cfg.pop("loss_seed_weight", None)
            ),
            "seed_target_gamma": branch_cfg.pop("seed_target_gamma", None),
        }
        # These keys belong to the superseded pixel-kernel draft.  The Phase-1
        # operator uses normalized relation_scales and fixed audited types.
        branch_cfg.pop("kernel_scales", None)
        branch_cfg.pop("relation_types", None)
        self.seed_operator_cfg = {
            **branch_cfg,
            **dict(seed_operator_cfg or {}),
        }
        self.role_encoder_cfg = dict(role_encoder_cfg or {})
        if "role_count" in phase_cfg:
            self.role_encoder_cfg.setdefault(
                "role_count", int(phase_cfg["role_count"])
            )
        super().__init__(*args, **kwargs)

        for name, value in auxiliary_head_cfg.items():
            if value is not None and hasattr(self.bbox_head, name):
                setattr(self.bbox_head, name, float(value))
                if name == "loss_seed_weight":
                    self.bbox_head.loss_seed_target_weight = float(value)

        self.role_encoder: Optional[nn.Module]
        self.seed_operator: Optional[nn.Module]
        if operator_name == "rqgo":
            role_cfg = {"d_model": self.embed_dims, **self.role_encoder_cfg}
            seed_cfg = {"d_model": self.embed_dims, **self.seed_operator_cfg}
            self.role_encoder = LatentRoleEncoder(**role_cfg)
            self.seed_operator = RQGO(**seed_cfg)
        elif operator_name == "generic":
            generic_cfg = dict(self.seed_operator_cfg)
            relation_scales = generic_cfg.pop(
                "relation_scales", (0.05, 0.15, 0.30))
            generic_cfg.setdefault(
                "hidden_dim",
                matched_generic_hidden_dim(
                    self.embed_dims,
                    self.num_feature_levels,
                    relation_count=8,
                    scale_count=len(relation_scales),
                ),
            )
            seed_cfg = {
                "d_model": self.embed_dims,
                "num_levels": self.num_feature_levels,
                **generic_cfg,
            }
            self.role_encoder = None
            self.seed_operator = GenericDenseSeedPredictor(**seed_cfg)
        else:
            self.role_encoder = None
            self.seed_operator = None

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare decoder inputs without changing the parent base ranking."""
        batch_size = memory.shape[0]
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        encoder_branch = self.decoder.num_layers
        enc_outputs_class = self.bbox_head.cls_branches[encoder_branch](
            output_memory, memory_text, text_token_mask
        )
        cls_out_features = self.bbox_head.cls_branches[
            encoder_branch
        ].max_text_len
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[encoder_branch](output_memory)
            + output_proposals
        )
        dense_enc_outputs_coord = enc_outputs_coord_unact.sigmoid()

        # Keep this expression byte-for-byte recognizable in static audits:
        # zero-init must reproduce the pinned parent's token-max Top-K.
        dense_base_score = enc_outputs_class.max(-1)[0]
        seed_result, role_attention = self._forward_seed_operator(
            output_memory=output_memory,
            dense_boxes=dense_enc_outputs_coord,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
        )
        finite_boxes = torch.isfinite(enc_outputs_coord_unact).all(dim=-1)
        seed_valid = seed_result.valid.to(dtype=torch.bool) & finite_boxes
        seed_bias = torch.where(
            seed_valid, seed_result.seed_bias, torch.zeros_like(dense_base_score)
        )
        # Do not add a new validity mask to selection: the pinned parent ranks
        # its raw token-max tensor directly.  ``seed_valid`` only masks the
        # auxiliary dense loss, while invalid seed residuals are exactly zero.
        select_score = dense_base_score + seed_bias

        topk_indices = torch.topk(
            select_score, k=self.num_queries, dim=1
        ).indices
        parent_topk_indices = torch.topk(
            dense_base_score, k=self.num_queries, dim=1
        ).indices
        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4),
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        selected_score = torch.gather(select_score, 1, topk_indices)
        selected_base_score = torch.gather(
            dense_base_score, 1, topk_indices
        )
        selected_seed_bias = torch.gather(seed_bias, 1, topk_indices)
        self.last_seed_diagnostics = {
            **_detached_scalar_diagnostics(
                getattr(seed_result, "diagnostics", {})
            ),
            "seed_select_score_mean": _finite_mean(selected_score).detach(),
            "seed_selected_base_score_mean": _finite_mean(
                selected_base_score
            ).detach(),
            "seed_selected_bias_abs_mean": _finite_mean(
                selected_seed_bias.abs()
            ).detach(),
            "seed_bias_abs_max": seed_bias.detach().abs().amax(),
            "seed_invalid_bias_abs_max": seed_bias.detach().masked_fill(
                seed_valid, 0.0).abs().amax(),
            "seed_rank_position_change_rate": (
                topk_indices != parent_topk_indices
            ).to(dtype=output_memory.dtype).mean().detach(),
            "seed_valid_fraction": seed_valid.to(
                dtype=output_memory.dtype
            ).mean().detach(),
        }

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, batch_size, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = (
                self.dn_query_generator(batch_data_samples)
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1
            )
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        head_inputs_dict = (
            dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta,
            )
            if self.training
            else {}
        )
        head_inputs_dict.update(
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            dense_enc_outputs_coord=dense_enc_outputs_coord,
            dense_base_score=dense_base_score,
            seed_bias=seed_bias,
            seed_valid=seed_valid,
            selected_encoder_coords=topk_coords,
            role_attention=role_attention,
        )
        return decoder_inputs_dict, head_inputs_dict

    def _forward_seed_operator(
        self,
        *,
        output_memory: Tensor,
        dense_boxes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
    ):
        valid = memory_valid_mask(output_memory, memory_mask)
        if self.seed_operator_name == "none":
            zeros = output_memory.new_zeros(valid.shape)
            result = SeedResult(zeros, zeros, valid, {})
            return result, None
        if self.seed_operator_name == "rqgo":
            role_state = self.role_encoder(memory_text, text_token_mask)
            result = self.seed_operator(
                output_memory,
                dense_boxes,
                role_state,
                spatial_shapes,
                memory_mask,
            )
            return result, role_state.attention

        pooled_text = _masked_text_mean(memory_text, text_token_mask)
        level_ids = _flattened_level_ids(
            spatial_shapes, output_memory.shape[1], output_memory.device
        )
        result = self.seed_operator(
            output_memory,
            dense_boxes,
            pooled_text,
            level_ids,
            memory_mask,
        )
        return result, None


def _masked_text_mean(memory_text: Tensor, text_token_mask: Tensor) -> Tensor:
    mask = text_token_mask.to(dtype=memory_text.dtype).unsqueeze(-1)
    return (memory_text * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _flattened_level_ids(
    spatial_shapes: Tensor, expected_tokens: int, device: torch.device
) -> Tensor:
    counts = spatial_shapes.to(device=device, dtype=torch.long).prod(dim=-1)
    level_ids = torch.arange(
        counts.numel(), device=device, dtype=torch.long
    ).repeat_interleave(counts)
    if level_ids.numel() != expected_tokens:
        raise ValueError(
            "spatial_shapes do not cover encoder memory: "
            f"{level_ids.numel()} != {expected_tokens}"
        )
    return level_ids


def _finite_mean(value: Tensor) -> Tensor:
    finite = torch.isfinite(value)
    clean = torch.where(finite, value, torch.zeros_like(value))
    return clean.sum() / finite.sum().clamp_min(1).to(dtype=value.dtype)


def _detached_scalar_diagnostics(values: Dict) -> Dict[str, Tensor]:
    scalars = {}
    for name, value in values.items():
        if isinstance(value, Tensor):
            scalars[str(name)] = _finite_mean(value).detach()
        elif isinstance(value, (float, int)):
            scalars[str(name)] = torch.as_tensor(float(value))
    return scalars
