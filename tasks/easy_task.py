"""
Easy Task: Static targets, low risk, 2 sensors, 3 targets.
Used to test basic sensor allocation functionality.
"""

EASY_TASK_CONFIG = {
    "num_sensors": 3,
    "num_targets": 3,
    "max_steps": 20,
    "sensor_failure_prob": 0.0,
    "targets": [
        {"id": 0, "priority": 3, "position": (2, 2), "dynamic": False, "risk": 0.0},
        {"id": 1, "priority": 2, "position": (5, 5), "dynamic": False, "risk": 0.0},
        {"id": 2, "priority": 1, "position": (8, 3), "dynamic": False, "risk": 0.0},
    ],
    "sensors": [
        {"id": 0, "range": 10, "available": True},
        {"id": 1, "range": 10, "available": True},
        {"id": 2, "range": 10, "available": True},
    ],
}


def get_easy_task():
    return EASY_TASK_CONFIG


def get_easy_env():
    from env import SentinelEnv
    return SentinelEnv(max_steps=EASY_TASK_CONFIG["max_steps"], seed=42, config=EASY_TASK_CONFIG)


def get_easy_multi_env():
    from env.multiagent import AryaXEnv
    return AryaXEnv(max_steps=EASY_TASK_CONFIG["max_steps"], seed=42, density_factor=1.5)
