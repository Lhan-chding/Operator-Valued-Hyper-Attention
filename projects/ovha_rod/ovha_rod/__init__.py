"""OVHA-ROD MMDetection project extension.

The pure PyTorch Phase-1 modules remain importable without MMDetection.  The
registry integration is loaded opportunistically when the OpenMMLab runtime is
available on the server.
"""

from .models import (GenericDenseSeedPredictor, LatentRoleEncoder, RQGO,
                     RQGOResult)

try:  # pragma: no cover - exercised in the server MMDetection environment.
    from . import optim as _optim
    from .models import dense_heads as _dense_heads
    from .models import detectors as _detectors
    from . import hooks as _hooks
    from .evaluation import OVHARefExpMetric as _OVHARefExpMetric
except ModuleNotFoundError as error:
    if error.name not in {"mmdet", "mmcv", "mmengine"}:
        raise

__all__ = [
    "GenericDenseSeedPredictor",
    "LatentRoleEncoder",
    "RQGO",
    "RQGOResult",
]
