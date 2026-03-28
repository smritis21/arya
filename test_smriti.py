"""
Test script for Smriti's files:
  tasks/easy_task.py, tasks/medium_task.py, tasks/hard_task.py
  tasks/grader.py, agent/policy.py
Run from project root: python test_smriti.py
"""

import sys
sys.path.insert(0, ".")

from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task, update_target_positions as medium_update
from tasks.hard_task import get_hard_task, apply_sensor_failures, update_target_positions as hard_update
from tasks.grader import grade, grade_summary
from agent.policy import select_action, random_action


# ── 1. Task configs load correctly ──────────────────────────────────────────
def test_tasks():
    easy   = get_easy_task()
    medium = get_medium_task()
    hard   = get_hard_task()

    assert easy["num_sensors"]   == 2
    assert medium["num_sensors"] == 3
    assert hard["num_sensors"]   == 4

    assert len(easy["targets"])   == 3
    assert len(medium["targets"]) == 5
    assert len(hard["targets"])   == 8

    print("✓ Task configs OK")


# ── 2. Dynamic target movement ───────────────────────────────────────────────
def test_dynamics():
    import copy
    medium = get_medium_task()
    targets_before = [t["position"] for t in medium["targets"]]
    updated = medium_update(copy.deepcopy(medium["targets"]))
    # positions are still within grid
    for t in updated:
        x, y = t["position"]
        assert 0 <= x <= 9 and 0 <= y <= 9

    hard = get_hard_task()
    sensors = apply_sensor_failures(copy.deepcopy(hard["sensors"]), failure_prob=1.0)
    assert all(not s["available"] for s in sensors), "All sensors should have failed"

    print("✓ Dynamics OK")


# ── 3. Grader returns value in [0.0, 1.0] ───────────────────────────────────
def test_grader():
    # Perfect episode: every target tracked, no failures
    perfect_log = [
        {"targets": [{"priority": 5, "tracked": True},
                     {"priority": 3, "tracked": True}],
         "sensor_failures": 0}
    ] * 5

    # Zero episode: nothing tracked
    zero_log = [
        {"targets": [{"priority": 5, "tracked": False},
                     {"priority": 3, "tracked": False}],
         "sensor_failures": 2}
    ] * 5

    perfect_score = grade(perfect_log)
    zero_score    = grade(zero_log)

    assert perfect_score == 1.0,          f"Expected 1.0, got {perfect_score}"
    assert 0.0 <= zero_score <= 1.0,      f"Score out of range: {zero_score}"
    assert zero_score < perfect_score,    "Zero log should score lower than perfect"

    summary = grade_summary(perfect_log)
    assert "score" in summary and "steps" in summary
    assert grade([]) == 0.0, "Empty log should return 0.0"

    print(f"✓ Grader OK  |  perfect={perfect_score}  zero={zero_score}")
    print(f"  Summary: {summary}")


# ── 4. Policy returns valid assignments ──────────────────────────────────────
def test_policy():
    obs = {
        "sensors": [
            {"id": 0, "available": True},
            {"id": 1, "available": False},
            {"id": 2, "available": True},
        ],
        "targets": [
            {"id": 0, "priority": 5, "tracked": False},
            {"id": 1, "priority": 2, "tracked": True},   # already tracked
            {"id": 2, "priority": 4, "tracked": False},
        ],
    }

    greedy = select_action(obs)
    # Only 2 sensors available, 2 untracked targets → 2 assignments
    assert len(greedy) == 2, f"Expected 2 assignments, got {len(greedy)}"
    # Highest priority target (id=0, priority=5) should be assigned first
    assert greedy[0][1] == 0, "Greedy should assign sensor to priority-5 target first"

    rnd = random_action(obs)
    assert len(rnd) == 2
    for sensor_id, target_id in rnd:
        assert isinstance(sensor_id, int) and isinstance(target_id, int)

    # Edge case: no available sensors
    obs_no_sensors = {"sensors": [{"id": 0, "available": False}], "targets": obs["targets"]}
    assert select_action(obs_no_sensors) == []

    print(f"✓ Policy OK  |  greedy={greedy}  random={rnd}")


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running tests...\n")
    test_tasks()
    test_dynamics()
    test_grader()
    test_policy()
    print("\nAll tests passed ✓")
