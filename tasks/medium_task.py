"""
Medium Task: Dynamic targets, moderate risk, 3 sensors, 5 targets.
Requires better decision-making as targets move each step.
"""

import random

MEDIUM_TASK_CONFIG = {
    "num_sensors": 4,
    "num_targets": 5,
    "max_steps": 40,
    "sensor_failure_prob": 0.05,
    "targets": [
        {"id": 0, "priority": 3, "position": (1, 1), "dynamic": True, "risk": 0.1},
        {"id": 1, "priority": 3, "position": (3, 7), "dynamic": True, "risk": 0.1},
        {"id": 2, "priority": 3, "position": (6, 4), "dynamic": True, "risk": 0.2},
        {"id": 3, "priority": 2, "position": (9, 9), "dynamic": False, "risk": 0.1},
        {"id": 4, "priority": 1, "position": (0, 5), "dynamic": False, "risk": 0.0},
    ],
    "sensors": [
        {"id": 0, "range": 8, "available": True},
        {"id": 1, "range": 6, "available": True},
        {"id": 2, "range": 7, "available": True},
        {"id": 3, "range": 9, "available": True},
    ],
}


def update_target_positions(targets):
    """Move dynamic targets by a random step each timestep."""
    for t in targets:
        if t["dynamic"]:
            x, y = t["position"]
            t["position"] = (
                max(0, min(9, x + random.choice([-1, 0, 1]))),
                max(0, min(9, y + random.choice([-1, 0, 1]))),
            )
    return targets


def get_medium_task():
    return MEDIUM_TASK_CONFIG


def get_medium_env():
    from env import SentinelEnv
    return SentinelEnv(max_steps=MEDIUM_TASK_CONFIG["max_steps"], seed=7, config=MEDIUM_TASK_CONFIG)


def get_medium_multi_env():
    from env.multiagent import AryaXEnv
    return AryaXEnv(max_steps=MEDIUM_TASK_CONFIG["max_steps"], seed=7, density_factor=2.5, conflict_injection=True)
