"""
Grader: Evaluates agent performance across a task episode.
Returns a normalized score between 0.0 and 1.0.
Compatible with Vishal's SentinelEnv step info output.
"""


def build_episode_log(env_steps: list[tuple]) -> list[dict]:
    """
    Converts SentinelEnv step outputs into grader-compatible episode log.

    Args:
        env_steps: list of (obs, reward, done, info) tuples from env.step()

    Returns:
        episode_log compatible with grade()
    """
    log = []
    for obs, reward, done, info in env_steps:
        log.append({
            "targets": [
                {"priority": t.priority, "tracked": not t.active}
                for t in obs.targets
            ],
            "sensor_failures": 0  # SentinelEnv resets sensors each step
        })
    return log


def grade(episode_log: list[dict]) -> float:
    """
    Evaluate agent performance from an episode log.

    Each entry in episode_log should contain:
        - "assignments": list of (sensor_id, target_id) pairs
        - "targets": list of target dicts with "priority" and "tracked" (bool)
        - "sensor_failures": int count of failures this step

    Returns:
        float: score in [0.0, 1.0]
    """
    if not episode_log:
        return 0.0

    total_possible = 0.0
    total_earned = 0.0
    penalty = 0.0

    for step in episode_log:
        targets = step.get("targets", [])
        sensor_failures = step.get("sensor_failures", 0)

        for t in targets:
            priority = t.get("priority", 1)
            total_possible += priority
            if t.get("tracked", False):
                total_earned += priority

        penalty += sensor_failures * 0.5

    if total_possible == 0:
        return 0.0

    raw_score = (total_earned - penalty) / total_possible
    return round(max(0.0, min(1.0, raw_score)), 4)


def grade_episode(total_reward: float, steps: int) -> float:
    """Normalize a single episode's total reward into a [0.0, 1.0] score."""
    if steps <= 0:
        return 0.0
    return round(max(0.0, min(1.0, total_reward / (steps * 10))), 4)


def grade_summary(episode_log: list[dict]) -> dict:
    """Returns a detailed breakdown alongside the final score."""
    score = grade(episode_log)
    steps = len(episode_log)
    total_tracked = sum(
        sum(1 for t in step.get("targets", []) if t.get("tracked", False))
        for step in episode_log
    )
    total_failures = sum(step.get("sensor_failures", 0) for step in episode_log)

    return {
        "score": score,
        "steps": steps,
        "total_tracked": total_tracked,
        "total_sensor_failures": total_failures,
    }
