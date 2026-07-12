from .operators.generic_seed import GenericDenseSeedPredictor
from .operators.rqgo import RQGO, RQGOResult
from .role_encoder import LatentRoleEncoder, RoleState, role_diversity_loss

try:  # Optional until the pinned MMDetection runtime is installed.
    from .dense_heads import OVHAGroundingDINOHead
    from .detectors import OVHAGroundingDINO
except ModuleNotFoundError as error:
    if error.name not in {"mmcv", "mmdet", "mmengine"}:
        raise
    OVHAGroundingDINO = None
    OVHAGroundingDINOHead = None

__all__ = [
    "GenericDenseSeedPredictor",
    "LatentRoleEncoder",
    "OVHAGroundingDINO",
    "OVHAGroundingDINOHead",
    "RoleState",
    "RQGO",
    "RQGOResult",
    "role_diversity_loss",
]
