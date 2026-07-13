from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import Tensor, nn

from .decoder_contracts import DecoderOperatorResidual


@dataclass(frozen=True)
class TQCATOResult:
    """Structurally frozen token-to-query transport and residual proposal."""

    residual: DecoderOperatorResidual
    transport: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.residual, DecoderOperatorResidual):
            raise ValueError("residual must be a DecoderOperatorResidual")
        if self.transport.ndim != 3:
            raise ValueError("transport must have shape [B,T,Q]")
        batch_size, _, query_count = self.transport.shape
        if self.residual.query_delta.shape[:2] != (batch_size, query_count):
            raise ValueError("transport and residual shapes must agree")
        if not self.transport.is_floating_point():
            raise ValueError("transport must be floating point")
        if self.transport.device != self.residual.query_delta.device:
            raise ValueError("transport and residual must share a device")
        if self.transport.dtype != self.residual.query_delta.dtype:
            raise ValueError("transport and residual must share a dtype")
        if not torch.isfinite(self.transport).all():
            raise ValueError("transport must contain only finite values")


class TQCATO(nn.Module):
    """Competitive row-normalized transport from text tokens to queries."""

    def __init__(self, d_model: int, *, temperature: float = 1.0) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not isfinite(float(temperature))
        ):
            raise ValueError("temperature must be a positive finite number")
        if temperature <= 0:
            raise ValueError("temperature must be a positive finite number")

        self.d_model = d_model
        self.temperature = float(temperature)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.text_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.gate_head = nn.Linear(d_model, 3)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        query: Tensor,
        text: Tensor,
        query_valid: Tensor,
        text_valid: Tensor,
    ) -> TQCATOResult:
        self._validate_inputs(query, text, query_valid, text_valid)

        query_keys = self.query_projection(query)
        text_keys = self.text_projection(text)
        scale = self.temperature * sqrt(self.d_model)
        logits = torch.matmul(text_keys, query_keys.transpose(-1, -2)) / scale
        transport = self._masked_row_softmax(logits, query_valid, text_valid)

        text_values = self.value_projection(text)
        query_context = torch.einsum("btq,btd->bqd", transport, text_values)
        incoming_mass = transport.sum(dim=1)
        normalizer = torch.where(
            incoming_mass > 0, incoming_mass, torch.ones_like(incoming_mass)
        )
        query_context = query_context / normalizer[..., None]
        query_mask = query_valid.to(dtype=query.dtype)[..., None]
        query_context = query_context * query_mask

        query_delta = self.output_projection(query_context) * query_mask
        gate_logits = self.gate_head(query + query_context) * query_mask
        row_error = (transport.sum(dim=-1) - text_valid.to(query.dtype)).abs()
        entropy = -(
            transport
            * transport.clamp_min(torch.finfo(query.dtype).tiny).log()
        ).sum(dim=-1)
        residual = DecoderOperatorResidual(
            query_delta=query_delta,
            box_delta=query.new_zeros((*query.shape[:2], 4)),
            score_delta=query.new_zeros(query.shape[:2]),
            gate_logits=gate_logits,
            valid=query_valid,
            diagnostics={
                "transport_row_error": row_error.amax(),
                "transport_entropy": entropy[text_valid].mean(),
            },
        )
        return TQCATOResult(residual=residual, transport=transport)

    @staticmethod
    def _masked_row_softmax(
        logits: Tensor,
        query_valid: Tensor,
        text_valid: Tensor,
    ) -> Tensor:
        query_mask = query_valid[:, None, :]
        valid_pairs = query_mask & text_valid[..., None]
        masked_logits = logits.masked_fill(~query_mask, -torch.inf)
        row_maximum = masked_logits.amax(dim=-1, keepdim=True)
        weights = torch.exp(masked_logits - row_maximum)
        weights = weights * valid_pairs.to(dtype=logits.dtype)
        denominator = weights.sum(dim=-1, keepdim=True)
        denominator = torch.where(
            text_valid[..., None], denominator, torch.ones_like(denominator)
        )
        return weights / denominator

    def _validate_inputs(
        self,
        query: Tensor,
        text: Tensor,
        query_valid: Tensor,
        text_valid: Tensor,
    ) -> None:
        tensors = {
            "query": query,
            "text": text,
            "query_valid": query_valid,
            "text_valid": text_valid,
        }
        for name, value in tensors.items():
            if not isinstance(value, Tensor):
                raise ValueError(f"{name} must be a tensor")
        if query.ndim != 3 or text.ndim != 3:
            raise ValueError("query and text must have shape [B,Q,D] and [B,T,D]")
        if query.shape[0] != text.shape[0]:
            raise ValueError("query and text batch dimensions must agree")
        if query.shape[-1] != self.d_model or text.shape[-1] != self.d_model:
            raise ValueError(f"query and text feature dimension must be {self.d_model}")
        if not query.is_floating_point() or not text.is_floating_point():
            raise ValueError("query and text must be floating point")
        if query.device != text.device:
            raise ValueError("query and text must share a device")
        if query.dtype != text.dtype:
            raise ValueError("query and text must share a dtype")
        if query_valid.shape != query.shape[:2]:
            raise ValueError("query_valid must have shape [B,Q]")
        if text_valid.shape != text.shape[:2]:
            raise ValueError("text_valid must have shape [B,T]")
        if query_valid.dtype != torch.bool or text_valid.dtype != torch.bool:
            raise ValueError("query_valid and text_valid must be boolean")
        if query_valid.device != query.device or text_valid.device != text.device:
            raise ValueError("features and masks must share a device")
        if not torch.isfinite(query).all() or not torch.isfinite(text).all():
            raise ValueError("query and text must contain only finite values")

        invalid_query_samples = (~query_valid.any(dim=-1)).nonzero(as_tuple=False)
        if invalid_query_samples.numel():
            sample = int(invalid_query_samples[0, 0])
            raise ValueError(f"sample {sample} must contain at least one valid query")
        invalid_text_samples = (~text_valid.any(dim=-1)).nonzero(as_tuple=False)
        if invalid_text_samples.numel():
            sample = int(invalid_text_samples[0, 0])
            raise ValueError(f"sample {sample} must contain at least one valid text token")
