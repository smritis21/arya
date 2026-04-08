"""
Grader: Evaluates agent performance across a task episode.
Returns a normalized score strictly within (0.0, 1.0).
"""


def grade_episode(total_reward: float, steps: int, num_sensors: int = 5) -> float:
    """Normalize total_reward to a float strictly in (0, 1)."""
    try:
        steps = int(steps)
        num_sensors = int(num_sensors)
        total_reward = float(total_reward)
    except (TypeError, ValueError):
        return 0.5
    if steps <= 0 or num_sensors <= 0:
        return 0.5
    max_reward = float(steps * num_sensors * 2)
    min_reward = float(steps * -2)
    if max_reward == min_reward:
        return 0.5
    raw_score = (total_reward - min_reward) / (max_reward - min_reward)
    score = float(max(0.01, min(0.99, raw_score)))
    print(f"GRADE: {score} type={type(score)}")
    return score
