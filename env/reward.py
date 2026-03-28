from typing import List, Optional
from env.models import Action, Target, Sensor

PRIORITY_REWARD = {3: 10.0, 2: 5.0, 1: 2.0}
MISSED_HIGH_PENALTY = -10.0
IDLE_PENALTY = -2.0


def compute_reward(
    action: Optional[Action],
    targets: List[Target],
    sensors: List[Sensor]
) -> float:
    # No action taken — idle penalty
    if action is None:
        return IDLE_PENALTY

    target = next((t for t in targets if t.id == action.target_id and t.active), None)
    sensor = next((s for s in sensors if s.id == action.sensor_id and s.available), None)

    # Invalid sensor or target
    if sensor is None or target is None:
        return IDLE_PENALTY

    reward = PRIORITY_REWARD.get(target.priority, 0.0)

    # Penalize for each unassigned high-priority target
    assigned_ids = {action.target_id}
    for t in targets:
        if t.active and t.priority == 3 and t.id not in assigned_ids:
            reward += MISSED_HIGH_PENALTY

    return reward
