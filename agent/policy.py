"""
Baseline Policy: Assigns available sensors to the highest-priority active targets.
Compatible with SentinelEnv Observation model (string IDs, active flag).
"""
from env.models import Observation, Action


def select_action(obs: Observation) -> Action | None:
    available = [s for s in obs.sensors if s.available]
    active = sorted([t for t in obs.targets if t.active], key=lambda t: -t.priority)
    if not available or not active:
        return None
    return Action(sensor_id=available[0].id, target_id=active[0].id)


def random_action(obs: Observation) -> Action | None:
    import random
    available = [s for s in obs.sensors if s.available]
    active = [t for t in obs.targets if t.active]
    if not available or not active:
        return None
    return Action(sensor_id=random.choice(available).id, target_id=random.choice(active).id)
