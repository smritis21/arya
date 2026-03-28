"""
Baseline Policy: Assigns available sensors to the highest-priority active targets.
Compatible with Vishal's SentinelEnv Observation model (string IDs, active flag).
"""
from env.models import Observation, Action


def select_action(obs: Observation) -> Action | None:
    """
    Greedy priority-based policy.
    Assigns the first available sensor to the highest-priority active target.

    Args:
        obs: Observation object from SentinelEnv

    Returns:
        Action(sensor_id, target_id) or None if no valid assignment possible
    """
    available = [s for s in obs.sensors if s.available]
    active = sorted([t for t in obs.targets if t.active], key=lambda t: -t.priority)

    if not available or not active:
        return None

    return Action(sensor_id=available[0].id, target_id=active[0].id)


def random_action(obs: Observation) -> Action | None:
    """
    Random policy for exploration baseline.
    """
    import random

    available = [s for s in obs.sensors if s.available]
    active = [t for t in obs.targets if t.active]

    if not available or not active:
        return None

    return Action(
        sensor_id=random.choice(available).id,
        target_id=random.choice(active).id
    )
