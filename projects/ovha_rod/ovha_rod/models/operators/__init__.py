from typing import TYPE_CHECKING

from .decoder_contracts import (
    DecoderOperatorResidual,
    DecoderResidualState,
    StructuredResidualFusion,
)
from .decoder_integration import (
    DecoderIntegrationResult,
    apply_matching_query_residual,
    stack_decoder_operator_outputs,
)
from .generic_seed import GenericDenseSeedPredictor, matched_generic_hidden_dim
from .hyper_adapter import HyperAdapterResult, LowRankHyperAdapter
from .operator_memory import OperatorMemory, OperatorMemoryState
from .operator_router import OperatorRouter, OperatorRouterResult
from .relation_fields import RELATION_TYPES, RelationFieldBank
from .rceo import RCEO, RCEOResult
from .rqgo import RQGO, RQGOResult
from .tq_cato import TQCATO, TQCATOResult

if TYPE_CHECKING:  # pragma: no cover - static imports only.
    from .qsro import QSRO, QuerySpatialRelationOperator

_LAZY_QSRO_EXPORTS = frozenset({"QSRO", "QuerySpatialRelationOperator"})


def __getattr__(name: str):
    if name in _LAZY_QSRO_EXPORTS:
        from .qsro import QSRO, QuerySpatialRelationOperator
        exports = {
            "QSRO": QSRO,
            "QuerySpatialRelationOperator": QuerySpatialRelationOperator,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DecoderOperatorResidual",
    "DecoderIntegrationResult",
    "DecoderResidualState",
    "GenericDenseSeedPredictor",
    "HyperAdapterResult",
    "LowRankHyperAdapter",
    "matched_generic_hidden_dim",
    "OperatorMemory",
    "OperatorMemoryState",
    "OperatorRouter",
    "OperatorRouterResult",
    "RCEO",
    "RCEOResult",
    "QSRO",
    "QuerySpatialRelationOperator",
    "RELATION_TYPES",
    "RelationFieldBank",
    "RQGO",
    "RQGOResult",
    "StructuredResidualFusion",
    "apply_matching_query_residual",
    "stack_decoder_operator_outputs",
    "TQCATO",
    "TQCATOResult",
]
