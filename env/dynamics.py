import random
from typing import List
from env.models import Sensor, Target

SENSOR_TYPES = ["satellite", "drone", "radar"]


def initialize_sensors(seed: int = 42) -> List[Sensor]:
    rng = random.Random(seed)
    return [
        Sensor(id=f"S{i+1}", type=rng.choice(SENSOR_TYPES), range=round(rng.uniform(100.0, 500.0), 2), available=True)
        for i in range(rng.randint(3, 5))
    ]


def spawn_targets(step: int, seed: int = 42) -> List[Target]:
    rng = random.Random((seed * 6364136223846793005 + step) & 0xFFFFFFFFFFFFFFFF)
    return [
        Target(id=f"T{step}_{i+1}", priority=rng.randint(1, 3), active=True)
        for i in range(rng.randint(2, 4))
    ]


def update_targets(targets: List[Target]) -> List[Target]:
    return []
