from .operators.generic_seed import GenericDenseSeedPredictor
from .operators.rqgo import RQGO, RQGOResult
from .positional_encoding import DeterministicSinePositionalEncoding
from .role_encoder import LatentRoleEncoder, RoleState, role_diversity_loss

try:  # Optional until the pinned MMDetection runtime is installed.
    from .dense_heads import OVHAGroundingDINOHead
    from .detectors import DeterministicGroundingDINO, OVHAGroundingDINO
except ModuleNotFoundError as error:
    if error.name not in {"mmcv", "mmdet", "mmengine"}:
        raise
    OVHAGroundingDINO = None
    OVHAGroundingDINOHead = None
    DeterministicGroundingDINO = None

__all__ = [
    "GenericDenseSeedPredictor",
    "DeterministicGroundingDINO",
    "DeterministicSinePositionalEncoding",
    "LatentRoleEncoder",
    "OVHAGroundingDINO",
    "OVHAGroundingDINOHead",
    "RoleState",
    "RQGO",
    "RQGOResult",
    "role_diversity_loss",
]
