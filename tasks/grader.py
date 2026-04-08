"""
Grader: Evaluates agent performance across a task episode.
Returns a normalized score strictly within (0.0, 1.0).
"""


def grade_episode(total_reward: float, steps: int, num_sensors: int = 5) -> float:
    """Normalize total_reward to a float strictly in (0, 1)."""
    if steps <= 0 or num_sensors <= 0:
        return 0.5

    max_reward = steps * num_sensors * 2.0
<<<<<<< HEAD
    min_reward = steps * -2.0
=======
    min_reward = steps * num_sensors * -2.0

>>>>>>> 2fc3fc6 (feat: implement episode grader and update inference scoring logic to enforce normalized bounds)
    if max_reward == min_reward:
        raw_score = 0.5
    else:
        raw_score = (total_reward - min_reward) / (max_reward - min_reward)

    # Strictly clamp within (0.0, 1.0) — never exactly 0.0 or 1.0
    return max(0.01, min(0.99, float(raw_score)))