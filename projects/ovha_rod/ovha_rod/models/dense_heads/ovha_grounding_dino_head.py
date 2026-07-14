from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.models.layers import inverse_sigmoid
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from ..losses import build_seed_quality_targets, quality_focal_seed_loss
from ...prediction_metadata import (
    require_prediction_batch_alignment,
    with_encoder_query_boxes,
)
from ..role_encoder import role_diversity_loss
from ..scoring import referent_focal_loss


@MODELS.register_module()
class OVHAGroundingDINOHead(GroundingDINOHead):
    """Grounding-DINO head with additive Phase-1 auxiliary losses.

    Parent token classification, box, encoder, decoder and DN losses are
    delegated unchanged to ``GroundingDINOHead``.  The referent branch is
    additive and zero-initialized, while the seed loss consumes dense encoder
    proposals passed by :class:`OVHAGroundingDINO`.
    """

    def __init__(
        self,
        *args,
        loss_seed_weight: float = 0.5,
        loss_ref_weight: float = 0.5,
        loss_role_div_weight: float = 0.005,
        seed_target_gamma: float = 1.0,
        **kwargs,
    ) -> None:
        if (
            loss_seed_weight < 0
            or loss_ref_weight < 0
            or loss_role_div_weight < 0
        ):
            raise ValueError("auxiliary loss weights must be non-negative")
        if seed_target_gamma <= 0:
            raise ValueError("seed_target_gamma must be positive")
        self.loss_seed_weight = float(loss_seed_weight)
        self.loss_seed_target_weight = float(loss_seed_weight)
        self.loss_ref_weight = float(loss_ref_weight)
        self.loss_role_div_weight = float(loss_role_div_weight)
        self.seed_target_gamma = float(seed_target_gamma)
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        super()._init_layers()
        self.referent_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, 1),
        )
        nn.init.zeros_(self.referent_head[-1].weight)
        nn.init.zeros_(self.referent_head[-1].bias)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.zeros_(self.referent_head[-1].weight)
        nn.init.zeros_(self.referent_head[-1].bias)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        operator_box_deltas: Optional[Tensor] = None,
        operator_referent_scores: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        if operator_box_deltas is None and operator_referent_scores is None:
            return super().forward(
                hidden_states,
                references,
                memory_text,
                text_token_mask,
            )
        all_layers_cls_scores, all_layers_bbox_preds = super().forward(
            hidden_states,
            references,
            memory_text,
            text_token_mask,
        )
        matching_count = _validate_operator_stacks(
            hidden_states,
            operator_box_deltas,
            operator_referent_scores,
        )
        if operator_box_deltas is not None:
            prefix_count = all_layers_bbox_preds.shape[2] - matching_count
            matching_boxes = all_layers_bbox_preds[:, :, prefix_count:, :]
            adjusted_matching_boxes = (
                inverse_sigmoid(matching_boxes, eps=1e-3)
                + operator_box_deltas
            ).sigmoid()
            all_layers_bbox_preds = torch.cat(
                (
                    all_layers_bbox_preds[:, :, :prefix_count, :],
                    adjusted_matching_boxes,
                ),
                dim=2,
            )
        return all_layers_cls_scores, all_layers_bbox_preds

    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Dict[str, int],
        dense_enc_outputs_coord: Optional[Tensor] = None,
        dense_base_score: Optional[Tensor] = None,
        seed_bias: Optional[Tensor] = None,
        seed_valid: Optional[Tensor] = None,
        selected_encoder_coords: Optional[Tensor] = None,
        role_attention: Optional[Tensor] = None,
        operator_box_deltas: Optional[Tensor] = None,
        operator_referent_scores: Optional[Tensor] = None,
    ) -> dict:
        if operator_box_deltas is None and operator_referent_scores is None:
            losses = super().loss(
                hidden_states=hidden_states,
                references=references,
                memory_text=memory_text,
                text_token_mask=text_token_mask,
                enc_outputs_class=enc_outputs_class,
                enc_outputs_coord=enc_outputs_coord,
                batch_data_samples=batch_data_samples,
                dn_meta=dn_meta,
            )
            all_layers_cls_scores, all_layers_bbox_preds = self(
                hidden_states, references, memory_text, text_token_mask)
        else:
            batch_gt_instances = [
                data_sample.gt_instances
                for data_sample in batch_data_samples
            ]
            batch_img_metas = [
                data_sample.metainfo for data_sample in batch_data_samples
            ]
            all_layers_cls_scores, all_layers_bbox_preds = self(
                hidden_states,
                references,
                memory_text,
                text_token_mask,
                operator_box_deltas,
                operator_referent_scores,
            )
            self.text_masks = text_token_mask
            losses = self.loss_by_feat(
                all_layers_cls_scores,
                all_layers_bbox_preds,
                enc_outputs_class,
                enc_outputs_coord,
                batch_gt_instances,
                batch_img_metas,
                dn_meta,
            )
        losses["loss_ref"] = self.loss_ref_weight * self._loss_referent(
            hidden_states=hidden_states,
            final_cls_scores=all_layers_cls_scores[-1],
            final_bbox_preds=all_layers_bbox_preds[-1],
            batch_data_samples=batch_data_samples,
            dn_meta=dn_meta,
            operator_referent_scores=operator_referent_scores,
        )
        losses["loss_seed"] = self.loss_seed_weight * self._loss_seed(
            dense_enc_outputs_coord=dense_enc_outputs_coord,
            dense_base_score=dense_base_score,
            seed_bias=seed_bias,
            seed_valid=seed_valid,
            selected_encoder_coords=selected_encoder_coords,
            batch_data_samples=batch_data_samples,
        )
        role_loss = (
            role_diversity_loss(role_attention)
            if role_attention is not None
            else _zero_from_parameters(self)
        )
        losses["loss_role_div"] = self.loss_role_div_weight * role_loss
        return losses

    def predict(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
        dense_enc_outputs_coord: Optional[Tensor] = None,
        dense_base_score: Optional[Tensor] = None,
        seed_bias: Optional[Tensor] = None,
        seed_valid: Optional[Tensor] = None,
        selected_encoder_coords: Optional[Tensor] = None,
        role_attention: Optional[Tensor] = None,
        operator_box_deltas: Optional[Tensor] = None,
        operator_referent_scores: Optional[Tensor] = None,
    ) -> InstanceList:
        # Dense tensors are deliberately ignored here: inference must depend
        # only on image/text forward state, never ground-truth annotations.
        del (
            dense_enc_outputs_coord,
            dense_base_score,
            seed_bias,
            seed_valid,
            role_attention,
        )
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_sample.token_positive_map for data_sample in batch_data_samples
        ]
        all_layers_cls_scores, all_layers_bbox_preds = self(
            hidden_states,
            references,
            memory_text,
            text_token_mask,
            operator_box_deltas,
            operator_referent_scores,
        )
        ref_delta = self.referent_head(hidden_states[-1]).squeeze(-1)
        if operator_referent_scores is not None:
            matching_count = operator_referent_scores.shape[2]
            prefix_count = ref_delta.shape[1] - matching_count
            ref_delta = torch.cat(
                (
                    ref_delta[:, :prefix_count],
                    ref_delta[:, prefix_count:]
                    + operator_referent_scores[-1],
                ),
                dim=1,
            )
        final_cls_scores = all_layers_cls_scores[-1]
        adjusted_final_scores = torch.where(
            torch.isfinite(final_cls_scores),
            final_cls_scores + ref_delta.unsqueeze(-1),
            final_cls_scores,
        )
        adjusted_cls_scores = torch.cat(
            [all_layers_cls_scores[:-1], adjusted_final_scores.unsqueeze(0)],
            dim=0,
        )
        predictions = self.predict_by_feat(
            adjusted_cls_scores,
            all_layers_bbox_preds,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale,
        )
        if selected_encoder_coords is not None:
            predictions = _attach_encoder_query_boxes(
                predictions, selected_encoder_coords, batch_img_metas, rescale)
        return predictions

    def _loss_seed(
        self,
        *,
        dense_enc_outputs_coord: Optional[Tensor],
        dense_base_score: Optional[Tensor],
        seed_bias: Optional[Tensor],
        seed_valid: Optional[Tensor],
        selected_encoder_coords: Optional[Tensor],
        batch_data_samples: SampleList,
    ) -> Tensor:
        if dense_base_score is None:
            return _zero_from_parameters(self)
        if (
            dense_enc_outputs_coord is None
            or seed_bias is None
            or seed_valid is None
        ):
            raise ValueError(
                "dense seed loss requires boxes, base score, bias and valid mask"
            )
        gt_boxes, gt_mask = _normalized_ground_truth(
            batch_data_samples, dense_enc_outputs_coord
        )
        targets = build_seed_quality_targets(
            dense_enc_outputs_coord,
            gt_boxes,
            gt_mask,
            gamma=self.seed_target_gamma,
        )
        if selected_encoder_coords is not None:
            selected_targets = build_seed_quality_targets(
                selected_encoder_coords,
                gt_boxes,
                gt_mask,
                gamma=1.0,
            )
            best = selected_targets.amax(dim=1)
            self.last_seed_metrics = {
                "selected_encoder_oracle_0.5": (best >= 0.5).float().mean().detach(),
                "selected_encoder_oracle_0.75": (best >= 0.75).float().mean().detach(),
                "selected_encoder_max_iou": best.mean().detach(),
            }
        seed_logits = dense_base_score.detach() + seed_bias
        return quality_focal_seed_loss(seed_logits, targets, seed_valid)

    def _loss_referent(
        self,
        *,
        hidden_states: Tensor,
        final_cls_scores: Tensor,
        final_bbox_preds: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Dict[str, int],
        operator_referent_scores: Optional[Tensor] = None,
    ) -> Tensor:
        dn_count = int((dn_meta or {}).get("num_denoising_queries", 0))
        matching_hidden = hidden_states[-1, :, dn_count:, :]
        matching_cls = final_cls_scores[:, dn_count:, :]
        matching_boxes = final_bbox_preds[:, dn_count:, :]
        parent_score = matching_cls.max(dim=-1).values
        ref_score = parent_score + self.referent_head(
            matching_hidden
        ).squeeze(-1)
        if operator_referent_scores is not None:
            final_operator_score = operator_referent_scores[-1]
            if final_operator_score.shape != ref_score.shape:
                raise ValueError(
                    "operator referent scores must match matching queries")
            ref_score = ref_score + final_operator_score
        positive_rows = []
        with torch.no_grad():
            for batch_index, data_sample in enumerate(batch_data_samples):
                targets = self._get_targets_single(
                    matching_cls[batch_index],
                    matching_boxes[batch_index],
                    data_sample.gt_instances,
                    data_sample.metainfo,
                )
                pos_inds = targets[-2]
                row = torch.zeros_like(ref_score[batch_index], dtype=torch.bool)
                positive_rows.append(
                    row.scatter(
                        0,
                        pos_inds,
                        torch.ones_like(pos_inds, dtype=torch.bool),
                    )
                )
        positives = torch.stack(positive_rows, dim=0)
        return referent_focal_loss(ref_score, positives)


def _validate_operator_stacks(
    hidden_states: Tensor,
    operator_box_deltas: Optional[Tensor],
    operator_referent_scores: Optional[Tensor],
) -> int:
    layers, batch_size, total_queries = hidden_states.shape[:3]
    matching_count = None
    if operator_box_deltas is not None:
        if (
            operator_box_deltas.ndim != 4
            or operator_box_deltas.shape[:2] != (layers, batch_size)
            or operator_box_deltas.shape[-1] != 4
        ):
            raise ValueError(
                "operator_box_deltas must have shape [L,B,Q,4]")
        matching_count = operator_box_deltas.shape[2]
        if (
            operator_box_deltas.device != hidden_states.device
            or operator_box_deltas.dtype != hidden_states.dtype
        ):
            raise ValueError(
                "operator box deltas must share hidden-state dtype/device")
    if operator_referent_scores is not None:
        if (
            operator_referent_scores.ndim != 3
            or operator_referent_scores.shape[:2] != (layers, batch_size)
        ):
            raise ValueError(
                "operator_referent_scores must have shape [L,B,Q]")
        score_count = operator_referent_scores.shape[2]
        if matching_count is not None and score_count != matching_count:
            raise ValueError("operator box and score query counts must agree")
        matching_count = score_count
        if (
            operator_referent_scores.device != hidden_states.device
            or operator_referent_scores.dtype != hidden_states.dtype
        ):
            raise ValueError(
                "operator scores must share hidden-state dtype/device")
    if matching_count is None or matching_count <= 0:
        raise ValueError("at least one non-empty operator stack is required")
    if matching_count > total_queries:
        raise ValueError("operator matching queries cannot exceed total queries")
    return matching_count


def _normalized_ground_truth(
    batch_data_samples: SampleList, reference: Tensor
) -> tuple[Tensor, Tensor]:
    max_count = max(
        (len(data_sample.gt_instances.bboxes) for data_sample in batch_data_samples),
        default=0,
    )
    gt_rows = []
    mask_rows = []
    for data_sample in batch_data_samples:
        boxes = data_sample.gt_instances.bboxes
        count = len(boxes)
        if count == 0:
            normalized = reference.new_zeros((0, 4))
        else:
            image_height, image_width = data_sample.metainfo["img_shape"][:2]
            factor = reference.new_tensor(
                [image_width, image_height, image_width, image_height]
            )
            normalized_xyxy = boxes.to(reference) / factor
            normalized = bbox_xyxy_to_cxcywh(normalized_xyxy)
        gt_rows.append(F.pad(normalized, (0, 0, 0, max_count - count)))
        mask_rows.append(
            F.pad(
                torch.ones(count, dtype=torch.bool, device=reference.device),
                (0, max_count - count),
            )
        )
    return torch.stack(gt_rows, dim=0), torch.stack(mask_rows, dim=0)


def _zero_from_parameters(module: nn.Module) -> Tensor:
    parameter = next(module.parameters())
    return parameter.sum() * 0.0


def _attach_encoder_query_boxes(predictions: InstanceList,
                                normalized_boxes: Tensor,
                                batch_img_metas: List[dict],
                                rescale: bool) -> InstanceList:
    require_prediction_batch_alignment(
        predictions=predictions,
        encoder_boxes=normalized_boxes,
        image_metas=batch_img_metas,
    )
    updated_predictions = []
    for prediction, boxes, meta in zip(
            predictions, normalized_boxes, batch_img_metas):
        absolute = bbox_cxcywh_to_xyxy(boxes.clone())
        height, width = meta["img_shape"][:2]
        absolute[:, 0::2] *= width
        absolute[:, 1::2] *= height
        absolute[:, 0::2].clamp_(0, width)
        absolute[:, 1::2].clamp_(0, height)
        if rescale:
            scale_factor = absolute.new_tensor(meta["scale_factor"]).repeat((1, 2))
            absolute = absolute / scale_factor
        updated_predictions.append(
            with_encoder_query_boxes(prediction, absolute))
    return updated_predictions
