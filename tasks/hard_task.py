"""
Hard Task: High-risk zones, sensor failures, conflicting priorities.
4 sensors, 8 targets. Tests advanced allocation behavior.
"""

import random

HARD_TASK_CONFIG = {
    "num_sensors": 5,
    "num_targets": 8,
    "max_steps": 60,
    "sensor_failure_prob": 0.15,
    "high_risk_zones": [(3, 3), (7, 7), (5, 5)],
    "targets": [
        {"id": 0, "priority": 3, "position": (3, 3), "dynamic": True,  "risk": 0.4},
        {"id": 1, "priority": 3, "position": (7, 7), "dynamic": True,  "risk": 0.4},
        {"id": 2, "priority": 3, "position": (1, 8), "dynamic": True,  "risk": 0.2},
        {"id": 3, "priority": 3, "position": (8, 1), "dynamic": True,  "risk": 0.2},
        {"id": 4, "priority": 3, "position": (5, 5), "dynamic": True,  "risk": 0.5},
        {"id": 5, "priority": 3, "position": (2, 6), "dynamic": False, "risk": 0.1},
        {"id": 6, "priority": 2, "position": (6, 2), "dynamic": False, "risk": 0.1},
        {"id": 7, "priority": 1, "position": (0, 0), "dynamic": False, "risk": 0.0},
    ],
    "sensors": [
        {"id": 0, "range": 5, "available": True},
        {"id": 1, "range": 6, "available": True},
        {"id": 2, "range": 4, "available": True},
        {"id": 3, "range": 7, "available": True},
        {"id": 4, "range": 6, "available": True},
    ],
}


def apply_sensor_failures(sensors, failure_prob):
    """Randomly disable sensors based on failure probability."""
    for s in sensors:
        if s["available"] and random.random() < failure_prob:
            s["available"] = False
        elif not s["available"] and random.random() < 0.3:
            s["available"] = True  # sensor recovers
    return sensors


def update_target_positions(targets, high_risk_zones):
    """Move dynamic targets; increase risk if entering a high-risk zone."""
    for t in targets:
        if t["dynamic"]:
            x, y = t["position"]
            t["position"] = (
                max(0, min(9, x + random.choice([-1, 0, 1]))),
                max(0, min(9, y + random.choice([-1, 0, 1]))),
            )
            t["risk"] = 0.5 if t["position"] in high_risk_zones else t["risk"]
    return targets


def get_hard_task():
    return HARD_TASK_CONFIG


def get_hard_env():
    from env import SentinelEnv
    return SentinelEnv(max_steps=HARD_TASK_CONFIG["max_steps"], seed=13, config=HARD_TASK_CONFIG)


def get_hard_multi_env():
    from env.multiagent import AryaXEnv
    return AryaXEnv(max_steps=HARD_TASK_CONFIG["max_steps"], seed=13, density_factor=4.0, failure_prob=0.15, conflict_injection=True)
