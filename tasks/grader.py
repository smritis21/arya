"""
Grader: Evaluates agent performance across a task episode.
Returns a normalized score strictly within (0.0, 1.0).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interaction import NegotiationLayer
    from agent.policy import select_action


def grade_episode(total_reward: float, steps: int, num_sensors: int = 5) -> float:
    """Normalize total_reward to a float strictly in (0, 1)."""
    if steps <= 0 or num_sensors <= 0:
        return 0.5

    max_reward = steps * num_sensors * 2.0
    min_reward = steps * -2.0
    if max_reward == min_reward:
        raw_score = 0.5
    else:
        raw_score = (total_reward - min_reward) / (max_reward - min_reward)

    # Strictly clamp within (0.0, 1.0) — never exactly 0.0 or 1.0
    return max(0.01, min(0.99, float(raw_score)))


# ── Multi-Agent Extensions ─────────────────────────────────────────────────


def grade_multi_agent_episode(
    per_agent_rewards: dict,
    step_count: int,
    num_sensors: int,
    negotiation_layer,
    num_agents: int = 4,
) -> dict:
    """
    Grade a full multi-agent episode.

    Args:
        per_agent_rewards:  {agent_id: float} total reward per agent.
        step_count:         Number of environment steps in the episode.
        num_sensors:        Total sensor count.
        negotiation_layer:  NegotiationLayer instance (provides conflict_rate).
        num_agents:         Number of agents (default 4).

    Returns dict with keys:
        overall_score, efficiency_score, coordination_score,
        conflict_rate, per_agent_scores, improvement_over_baseline.
    """
    # Runtime import guard — grader.py remains usable standalone
    try:
        from interaction import NegotiationLayer as _NL  # noqa: F401
    except ImportError:
        raise ImportError("interaction/ package not found. Build it first.")

    if step_count <= 0 or num_sensors <= 0:
        empty: dict[str, object] = {
            "overall_score":             0.01,
            "efficiency_score":          0.0,
            "coordination_score":        0.0,
            "conflict_rate":             0.0,
            "per_agent_scores":          {aid: 0.0 for aid in per_agent_rewards},
            "improvement_over_baseline": 0.0,
        }
        return empty

    total_reward = sum(per_agent_rewards.values())

    # ── Efficiency score ──────────────────────────────────────────────
    # Max possible: every sensor covers a priority-3 target optimally (+3.0)
    # per step across all agents.
    max_possible = num_sensors * step_count * 3.0
    min_possible = step_count * -2.0
    raw_eff = (total_reward - min_possible) / max(max_possible - min_possible, 1.0)
    efficiency_score = max(0.0, min(1.0, raw_eff))

    # ── Coordination score ─────────────────────────────────────────────
    conflict_rate = negotiation_layer.get_conflict_rate()
    coordination_score = max(0.0, min(1.0, 1.0 - conflict_rate))

    # ── Overall score ──────────────────────────────────────────────────
    raw_overall = 0.6 * efficiency_score + 0.4 * coordination_score
    overall_score = max(0.01, min(0.99, raw_overall))

    # ── Per-agent scores ───────────────────────────────────────────────
    per_agent_scores: dict[str, float] = {}
    for aid, reward in per_agent_rewards.items():
        agent_max = num_sensors * step_count * 3.0 / num_agents
        agent_min = step_count * -2.0 / num_agents
        raw = (reward - agent_min) / max(agent_max - agent_min, 1.0)
        per_agent_scores[aid] = max(0.01, min(0.99, raw))

    # ── Baseline comparison ────────────────────────────────────────────
    baseline_score = grade_episode(total_reward, step_count, num_sensors)
    improvement_over_baseline = overall_score - baseline_score

    return {
        "overall_score":             overall_score,
        "efficiency_score":          efficiency_score,
        "coordination_score":        coordination_score,
        "conflict_rate":             conflict_rate,
        "per_agent_scores":          per_agent_scores,
        "improvement_over_baseline": improvement_over_baseline,
    }


def run_deterministic_eval(
    agents=None,
    verbose: bool = True,
    mode: str = "multi",
) -> dict:
    """
    Run a full deterministic evaluation across easy, medium, and hard tasks.

    Args:
        agents: list of agent instances (multi mode) OR None (uses greedy baseline).
        verbose: If True, prints a clean aligned results table to stdout.
        mode:    "multi" | "single"

    Returns dict:
        easy, medium, hard (grade_multi_agent_episode results or grade_episode float),
        average_score, average_conflict_rate.
    """
    try:
        from env.environment import SentinelEnv
        from env.multiagent import AryaXEnv
        from tasks.easy_task import get_easy_env, get_easy_multi_env
        from tasks.medium_task import get_medium_env, get_medium_multi_env
        from tasks.hard_task import get_hard_env, get_hard_multi_env
    except ImportError as exc:
        raise ImportError(f"env/ or tasks/ package not found: {exc}") from exc

    if mode == "multi":
        TASKS = [
            ("easy",   get_easy_multi_env,   42),
            ("medium", get_medium_multi_env, 7),
            ("hard",   get_hard_multi_env,   13),
        ]
    else:
        TASKS = [
            ("easy",   get_easy_env,   42),
            ("medium", get_medium_env, 7),
            ("hard",   get_hard_env,   13),
        ]

    results: dict = {}
    all_scores: list[float] = []
    all_conflict_rates: list[float] = []

    for task_name, get_env_fn, seed in TASKS:
        if mode == "multi":
            # Multi-agent mode — uses AryaXEnv + NegotiationLayer
            try:
                from interaction import NegotiationLayer
                from interaction.reward import RewardEngine
            except ImportError as exc:
                raise ImportError(f"interaction/ package not found: {exc}") from exc

            env: AryaXEnv = get_env_fn()
            obs_map = env.reset()
            nl           = NegotiationLayer()
            reward_eng   = RewardEngine()
            agent_ids    = ["satellite", "drone", "radar", "command"]
            total_rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}
            done         = False

            rng = __import__("random").Random(seed)

            while not done:
                # Greedy proposals per agent
                from env.multiagent import Proposal
                proposals = []
                used_sensors: set[str] = set()
                used_targets: set[str] = set()
                for agent_id in agent_ids:
                    agent_obs = obs_map[agent_id]
                    my_sensors = [
                        s for s in agent_obs.sensors
                        if s["available"] and s["id"] not in used_sensors
                        and (agent_id == "command" or s["type"] == agent_id)
                    ]
                    targets = sorted(
                        [t for t in agent_obs.targets if t["active"]],
                        key=lambda t: -t["priority"]
                    )
                    for sensor in my_sensors:
                        for target in targets:
                            if target["id"] not in used_targets:
                                proposals.append(Proposal(
                                    agent_id=agent_id,
                                    sensor_id=sensor["id"],
                                    target_id=target["id"]
                                ))
                                used_sensors.add(sensor["id"])
                                used_targets.add(target["id"])
                                break

                obs_map, step_r, done, info = env.step_multiagent(proposals)
                for aid, r in step_r.items():
                    total_rewards[aid] += r

            task_result = grade_multi_agent_episode(
                per_agent_rewards=total_rewards,
                step_count=env.max_steps,
                num_sensors=env.initial_sensor_count,
                negotiation_layer=env.negotiation,
                num_agents=4,
            )
            all_scores.append(task_result["overall_score"])
            all_conflict_rates.append(task_result["conflict_rate"])

        else:
            # Single-agent mode — uses greedy baseline policy
            env: SentinelEnv = get_env_fn()
            obs = env.reset()
            try:
                from agent.policy import select_action
            except ImportError:
                def select_action(obs):
                    available = [s for s in obs.sensors if s.available]
                    active = sorted(
                        [t for t in obs.targets if t.active], key=lambda t: -t.priority
                    )
                    if not available or not active:
                        return None
                    from env.models import Action
                    return Action(sensor_id=available[0].id, target_id=active[0].id)

            total_ep_reward = 0.0
            done = False
            while not done:
                action = select_action(obs)
                obs, reward, done, _ = env.step(action)
                total_ep_reward += reward

            score = grade_episode(total_ep_reward, env.max_steps, env.initial_sensor_count)
            task_result = {
                "overall_score":             score,
                "efficiency_score":          score,
                "coordination_score":        1.0,
                "conflict_rate":             0.0,
                "per_agent_scores":          {},
                "improvement_over_baseline": 0.0,
            }
            all_scores.append(score)
            all_conflict_rates.append(0.0)

        results[task_name] = task_result

    average_score         = sum(all_scores) / max(len(all_scores), 1)
    average_conflict_rate = sum(all_conflict_rates) / max(len(all_conflict_rates), 1)

    if verbose:
        _print_eval_table(results, average_score, average_conflict_rate)

    results["average_score"]         = average_score
    results["average_conflict_rate"] = average_conflict_rate
    return results


def _print_eval_table(
    results: dict,
    avg_score: float,
    avg_conflict_rate: float,
) -> None:
    """Print a clean aligned evaluation results table."""
    divider = "-" * 62
    header  = f"{'Task':<10} {'Overall':>8} {'Efficiency':>12} {'Coord':>8} {'Conflict':>10}"
    print(f"\n{divider}")
    print(f"  ARYA-X Deterministic Evaluation Results")
    print(divider)
    print(f"  {header}")
    print(divider)
    for task_name in ("easy", "medium", "hard"):
        r = results.get(task_name, {})
        print(
            f"  {task_name.capitalize():<10}"
            f" {r.get('overall_score', 0):>8.3f}"
            f" {r.get('efficiency_score', 0):>12.3f}"
            f" {r.get('coordination_score', 0):>8.3f}"
            f" {r.get('conflict_rate', 0):>10.3f}"
        )
    print(divider)
    print(
        f"  {'Average':<10} {avg_score:>8.3f}"
        f"                    {avg_conflict_rate:>10.3f}"
    )
    print(f"{divider}\n")