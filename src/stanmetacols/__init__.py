"""stanmetacols — identify which .obs columns fill standard metadata roles."""

from .schema import Candidate, MetaColsResult, ObsDigest
from .llm_client import LLMUnavailable, OpenAICompatClient, call_structured, extract_json
from .profile import profile_obs
from .roles import ROLES, ROLE_KEYS
from .rank import rank_meta_columns

__version__ = "0.2.0"

__all__ = [
    "rank_meta_columns", "profile_obs", "Candidate", "MetaColsResult",
    "ObsDigest", "ROLES", "ROLE_KEYS", "LLMUnavailable",
    "OpenAICompatClient", "call_structured", "extract_json", "__version__",
]
