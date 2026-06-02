from __future__ import annotations

from torch import nn

from moat_ovha_torch.models.multimodal.hyper_adapter import MultimodalHyperAdapter
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES
from moat_ovha_torch.models.multimodal.router import MultimodalRelationRouter


class MultimodalJointRouterAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES,
        use_evidence_router: bool = True,
    ):
        super().__init__()
        self.router = MultimodalRelationRouter(
            d_model=d_model,
            candidate_names=candidate_names,
            use_evidence_router=use_evidence_router,
        )
        self.hyper_adapter = MultimodalHyperAdapter(d_model=d_model, candidate_names=candidate_names)

    def forward(self, memory_bank, evidence, reliability):
        router_output = self.router(memory_bank, evidence, reliability)
        params = self.hyper_adapter(memory_bank, evidence, reliability)
        return router_output, params
