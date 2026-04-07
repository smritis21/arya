"""
Grader: Evaluates agent performance across a task episode.
Returns a normalized score strictly between (0.0, 1.0).
"""


def grade_episode(total_reward: float, steps: int, num_sensors: int = 5) -> float:
    """Normalize a single episode's total reward into a score strictly in (0, 1)."""
    if steps <= 0:
        return 1e-6
    max_reward_per_step = num_sensors * 2.0
    best = steps * max_reward_per_step
    worst = steps * -2.0
    normalized = (total_reward - worst) / (best - worst)
    epsilon = 1e-6
    return round(max(epsilon, min(normalized, 1 - epsilon)), 6)
