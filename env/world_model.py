import random
from typing import List, Dict
from env.models import Sensor, Target


def add_observation_noise(targets: List[Target], sensor_type: str, rng: random.Random) -> List[Target]:
    """Return new Target list with potentially wrong priority values based on sensor type."""
    noisy = []
    for t in targets:
        p = t.priority
        if sensor_type == "satellite":
            if rng.random() < 0.10:
                p = max(1, min(3, p + rng.choice([-1, 1])))
        elif sensor_type == "drone":
            if rng.random() < 0.05:
                p = max(1, min(3, p + rng.choice([-1, 1])))
        elif sensor_type == "radar":
            if rng.random() < 0.15:
                p = max(1, min(3, p + rng.choice([-1, 1])))
        # command sees aggregated reports — same noise as satellite
        elif sensor_type == "command":
            if rng.random() < 0.10:
                p = max(1, min(3, p + rng.choice([-1, 1])))
        noisy.append(Target(id=t.id, priority=p, active=t.active))
    return noisy


def apply_mask(global_targets: List[Target], agent_type: str, sensors: List[Sensor]) -> List[Target]:
    """Filter targets based on agent type's observability constraints."""
    if agent_type in ("satellite", "command"):
        return list(global_targets)
    elif agent_type == "drone":
        # Drone sees kinetic + strategic targets; misses pure airspace
        return [t for t in global_targets if t.type != "airspace"]
    elif agent_type == "radar":
        # Radar sees airspace + kinetic targets; misses pure strategic
        return [t for t in global_targets if t.type != "strategic"]
    return list(global_targets)


def get_priority_mapping(episode_number: int) -> Dict[int, int]:
    """Schema drift: every 20 episodes, shift one priority mapping."""
    base = {1: 1, 2: 2, 3: 3}
    drift_cycle = (episode_number // 20) % 3
    if drift_cycle == 1:
        base[1] = 2  # priority-1 treated as priority-2
    elif drift_cycle == 2:
        base[2] = 3  # priority-2 treated as priority-3
    return base


def _drone_in_range(target: Target) -> bool:
    """Heuristic: drone sees targets whose step index is even (simulates range constraint)."""
    try:
        step = int(target.id.split("_")[0][1:])
        return step % 2 == 0
    except (IndexError, ValueError):
        return True


def _is_airspace_target(target: Target) -> bool:
    """Heuristic: radar sees airspace targets — those with odd sequential index."""
    try:
        idx = int(target.id.split("_")[1])
        return idx % 2 == 1
    except (IndexError, ValueError):
        return True
