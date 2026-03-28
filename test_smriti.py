"""
Integration test: Smriti's files + Vishal's SentinelEnv
Run from project root: python test_smriti.py
"""

import sys
sys.path.insert(0, ".")

from env import SentinelEnv
from env.models import Action
from tasks import get_easy_task, get_medium_task, get_hard_task, grade, grade_summary
from tasks.grader import build_episode_log
from agent.policy import select_action, random_action


# ── 1. Task configs ──────────────────────────────────────────────────────────
def test_tasks():
    assert get_easy_task()["num_sensors"]   == 2
    assert get_medium_task()["num_sensors"] == 3
    assert get_hard_task()["num_sensors"]   == 4
    print("✓ Task configs OK")


# ── 2. Policy works with real env observation ────────────────────────────────
def test_policy():
    env = SentinelEnv(max_steps=5, seed=42)
    obs = env.reset()

    action = select_action(obs)
    assert action is None or isinstance(action, Action)

    if action:
        sensor_ids = {s.id for s in obs.sensors if s.available}
        target_ids = {t.id for t in obs.targets if t.active}
        assert action.sensor_id in sensor_ids
        assert action.target_id in target_ids

    rnd = random_action(obs)
    assert rnd is None or isinstance(rnd, Action)

    print(f"✓ Policy OK  |  greedy={action}  random={rnd}")


# ── 3. Full episode with greedy policy ───────────────────────────────────────
def test_full_episode():
    env = SentinelEnv(max_steps=10, seed=42)
    obs = env.reset()

    episode_steps = []
    done = False

    while not done:
        action = select_action(obs)
        obs, reward, done, info = env.step(action)
        episode_steps.append((obs, reward, done, info))

    assert len(episode_steps) > 0
    print(f"✓ Episode OK  |  steps={len(episode_steps)}  total_reward={sum(r for _, r, _, _ in episode_steps):.1f}")


# ── 4. Grader scores a real episode ─────────────────────────────────────────
def test_grader():
    env = SentinelEnv(max_steps=10, seed=42)
    obs = env.reset()

    episode_steps = []
    done = False

    while not done:
        action = select_action(obs)
        obs, reward, done, info = env.step(action)
        episode_steps.append((obs, reward, done, info))

    log = build_episode_log(episode_steps)
    score = grade(log)
    summary = grade_summary(log)

    assert 0.0 <= score <= 1.0
    assert "score" in summary and "steps" in summary

    print(f"✓ Grader OK  |  score={score}  summary={summary}")


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running integration tests...\n")
    test_tasks()
    test_policy()
    test_full_episode()
    test_grader()
    print("\nAll tests passed ✓")
