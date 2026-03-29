import random
from typing import List
from env.models import Sensor, Target

SENSOR_TYPES = ["satellite", "drone", "radar"]


def initialize_sensors(seed: int = 42) -> List[Sensor]:
    rng = random.Random(seed)
    sensors = []
    for i in range(rng.randint(3, 5)):
        sensors.append(Sensor(
            id=f"S{i+1}",
            type=rng.choice(SENSOR_TYPES),
            range=round(rng.uniform(100.0, 500.0), 2),
            available=True
        ))
    return sensors


def spawn_targets(step: int, seed: int = 42) -> List[Target]:
<<<<<<< HEAD
    rng = random.Random(seed + step)
=======
    # XOR with a large prime so step 0 and step 1 don't collide when seed differs by 1
    rng = random.Random((seed * 6364136223846793005 + step) & 0xFFFFFFFFFFFFFFFF)
>>>>>>> round1-submission
    targets = []
    for i in range(rng.randint(2, 4)):
        targets.append(Target(
            id=f"T{step}_{i+1}",
            priority=rng.randint(1, 3),
            active=True
        ))
    return targets


def update_targets(targets: List[Target]) -> List[Target]:
<<<<<<< HEAD
    # Keep only active targets
    return [t for t in targets if t.active]
=======
    # Targets that weren't handled this step expire — no accumulation
    return []
>>>>>>> round1-submission
