import random
import math
from typing import List
from env.models import Sensor, Target

SENSOR_TYPES = ["satellite", "drone", "radar"]


def initialize_sensors(seed: int = 42) -> List[Sensor]:
    rng = random.Random(seed)
    return [
        Sensor(id=f"S{i+1}", type=rng.choice(SENSOR_TYPES), range=round(rng.uniform(100.0, 500.0), 2), available=True)
        for i in range(rng.randint(3, 5))
    ]


def spawn_targets(
    step: int,
    seed: int = 42,
    conflict_injection: bool = False,
) -> List[Target]:
    """
    Spawn targets for the given step.

    Args:
        step: Current environment step (used in the RNG seed mix).
        seed: Episode seed.
        conflict_injection: If True (curriculum Phase 2+), guarantees at least
            one priority-3 target that sits in the overlapping coverage zone of
            >= 2 sensor types (index 1 = drone/radar boundary; index 2 = sat/all
            zone).  This forces the ConflictDetector to fire, exercising the
            resolver and NegotiationLayer during training.
    """
    rng = random.Random((seed * 6364136223846793005 + step) & 0xFFFFFFFFFFFFFFFF)
    count = rng.randint(2, 4)
    targets = [
        Target(id=f"T{step}_{i+1}", priority=rng.randint(1, 3), active=True)
        for i in range(count)
    ]

    if conflict_injection:
        # Guarantee a P3 target at the boundary overlap zone (index 1).
        # Index 1 is claimed by both DroneAgent (proximity bias) and
        # RadarAgent (airspace = odd index), triggering REDUNDANT_COVERAGE.
        overlap_id = f"T{step}_1"
        injected = next((t for t in targets if t.id == overlap_id), None)
        if injected:
            # Elevate existing target to P3
            targets = [
                Target(id=t.id, priority=3 if t.id == overlap_id else t.priority, active=t.active)
                for t in targets
            ]
        else:
            # Prepend new P3 target in the overlap zone
            targets = [Target(id=overlap_id, priority=3, active=True)] + targets

    return targets



def spawn_targets_stochastic(step: int, seed: int = 42, density_factor: float = 2.5) -> List[Target]:
    """Poisson-based target spawning. density_factor: easy=1.5, medium=2.5, hard=4.0"""
    rng = random.Random((seed * 6364136223846793005 + step) & 0xFFFFFFFFFFFFFFFF)
    # Poisson sample via Knuth algorithm
    L = math.exp(-density_factor)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    count = max(1, k - 1)
    return [
        Target(id=f"T{step}_{i+1}", priority=rng.randint(1, 3), active=True)
        for i in range(count)
    ]


def apply_correlated_failures(sensors: List[Sensor], weather_seed: int, failure_prob: float) -> List[Sensor]:
    """Correlated failure: weather event degrades drones + reduces radar range together."""
    rng = random.Random(weather_seed)
    weather_event = rng.random() < 0.25  # 25% chance of weather event per step
    for s in sensors:
        if weather_event:
            if s.type == "drone" and rng.random() < failure_prob:
                s.available = False
            elif s.type == "radar":
                s.range = round(s.range * 0.7, 2)  # 30% range reduction
        else:
            if rng.random() < failure_prob * 0.3:
                s.available = False
    return sensors


def update_targets(targets: List[Target]) -> List[Target]:
    return []
