from ktabforge.proposals.config import ProposalConfig, load_proposal_config
from ktabforge.proposals.runner import (
    ProposalResult,
    register_proposal_from_config,
    validate_proposal_config,
)

__all__ = [
    "ProposalConfig",
    "ProposalResult",
    "load_proposal_config",
    "register_proposal_from_config",
    "validate_proposal_config",
]
