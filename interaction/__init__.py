"""ARYA-X Interaction & Negotiation Layer — conflict detection, resolution, and reward computation."""

__version__ = "1.0.0"

try:
    from .negotiation import NegotiationLayer
except ImportError as e:
    raise ImportError(
        f"interaction/negotiation.py not found. Build it first. Error: {e}"
    ) from e

try:
    from .conflict import ConflictDetector
except ImportError as e:
    raise ImportError(
        f"interaction/conflict.py not found. Build it first. Error: {e}"
    ) from e

try:
    from .resolver import ConflictResolver
except ImportError as e:
    raise ImportError(
        f"interaction/resolver.py not found. Build it first. Error: {e}"
    ) from e

try:
    from .reward import RewardEngine
except ImportError as e:
    raise ImportError(
        f"interaction/reward.py not found. Build it first. Error: {e}"
    ) from e

__all__ = [
    "NegotiationLayer",
    "ConflictDetector",
    "ConflictResolver",
    "RewardEngine",
]
