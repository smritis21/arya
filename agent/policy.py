"""
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
