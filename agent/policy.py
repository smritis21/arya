"""
<<<<<<< HEAD
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
=======
Baseline Policy: Assigns available sensors to the highest-priority untracked targets.
Used for initial testing and as a performance baseline.
"""


def select_action(observation: dict) -> list[tuple[int, int]]:
    """
    Greedy priority-based policy.

    Args:
        observation: dict with keys:
            - "sensors": list of {"id": int, "available": bool}
            - "targets": list of {"id": int, "priority": int, "tracked": bool}

    Returns:
        List of (sensor_id, target_id) assignment pairs.
    """
    sensors = [s for s in observation.get("sensors", []) if s["available"]]
    targets = sorted(
        [t for t in observation.get("targets", []) if not t["tracked"]],
        key=lambda t: t["priority"],
        reverse=True,
    )

    assignments = []
    for sensor, target in zip(sensors, targets):
        assignments.append((sensor["id"], target["id"]))

    return assignments


def random_action(observation: dict) -> list[tuple[int, int]]:
    """
    Random policy for exploration baseline.
    Randomly assigns available sensors to targets.
    """
    import random

    sensors = [s for s in observation.get("sensors", []) if s["available"]]
    targets = observation.get("targets", [])

    if not sensors or not targets:
        return []

    random.shuffle(targets)
    assignments = []
    for sensor, target in zip(sensors, targets):
        assignments.append((sensor["id"], target["id"]))

    return assignments
>>>>>>> round1-submission
