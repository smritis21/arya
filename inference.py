"""
SentinelEnv — Inference Script
================================
Runs the LLM agent against all 3 tasks (Easy / Medium / Hard) and prints
a reproducible baseline score for each.

Environment variables required:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face API token
"""
import os
import json
from openai import OpenAI
from env import SentinelEnv
from env.models import Action
from env.multiagent import AryaXEnv, Proposal, AGENT_TYPES
from interaction import NegotiationLayer
from interaction.reward import RewardEngine
from tasks.grader import grade_episode
from tasks.easy_task import get_easy_env
from tasks.medium_task import get_medium_env
from tasks.hard_task import get_hard_env

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")

client = None

TASKS = [
    {"name": "Easy",   "env_fn": get_easy_env,   "seed": 42},
    {"name": "Medium", "env_fn": get_medium_env, "seed": 7},
    {"name": "Hard",   "env_fn": get_hard_env,   "seed": 13},
]


def build_prompt(obs) -> str:
    sensors = "\n".join(
        f"  - id={s.id} type={s.type} range={s.range}km available={s.available}"
        for s in obs.sensors if s.available
    )
    targets = "\n".join(
        f"  - id={t.id} priority={t.priority}"
        for t in obs.targets if t.active
    )
    n = sum(1 for s in obs.sensors if s.available)
    return f"""You are a military sensor allocation AI. Assign ALL {n} available sensors to threats.
Priority 3=HIGH (critical), 2=MED, 1=LOW. Cover HIGH threats first. Each sensor to a DIFFERENT target.

Timestep: {obs.timestep}

Available Sensors:
{sensors}

Active Threats:
{targets}

Respond ONLY with a JSON array, one object per available sensor:
[{{"sensor_id": "S1", "target_id": "T0_1"}}, {{"sensor_id": "S2", "target_id": "T0_2"}}]
"""


def parse_llm_actions(text: str, obs) -> list[Action]:
    try:
        start = text.find("[")
        end   = text.rfind("]") + 1
        data  = json.loads(text[start:end])
        valid_sensors = {s.id for s in obs.sensors if s.available}
        valid_targets = {t.id for t in obs.targets if t.active}
        actions, used_s, used_t = [], set(), set()
        for item in data:
            sid, tid = item.get("sensor_id"), item.get("target_id")
            if sid in valid_sensors and tid in valid_targets \
                    and sid not in used_s and tid not in used_t:
                actions.append(Action(sensor_id=sid, target_id=tid))
                used_s.add(sid)
                used_t.add(tid)
        return actions
    except Exception:
        return []


def greedy_actions(obs) -> list[Action]:
    available = [s for s in obs.sensors if s.available]
    targets   = sorted([t for t in obs.targets if t.active], key=lambda t: -t.priority)
    actions, used = [], set()
    for sensor in available:
        for target in targets:
            if target.id not in used:
                actions.append(Action(sensor_id=sensor.id, target_id=target.id))
                used.add(target.id)
                break
    return actions


def parse_action(output: str, obs) -> Action | None:
    """Wrapper around parse_llm_actions — returns first valid Action or None."""
    actions = parse_llm_actions(output, obs)
    return actions[0] if actions else None


def fallback_action(obs) -> Action | None:
    """Wrapper around greedy_actions — returns highest-priority Action or None."""
    actions = greedy_actions(obs)
    return actions[0] if actions else None


def get_actions(obs) -> tuple[list[Action], str]:
    if HF_TOKEN and client is not None:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": build_prompt(obs)}],
                max_tokens=256,
                temperature=0.0
            )
            raw     = response.choices[0].message.content.strip()
            actions = parse_llm_actions(raw, obs)
            if actions:
                return actions, "llm"
            print(f"  [WARN] Bad LLM response, using greedy. Raw: {raw!r}")
        except Exception as e:
            print(f"  [WARN] LLM error: {e}. Using greedy.")
    return greedy_actions(obs), "greedy"


def log_end(task: str, score: float, steps: int) -> None:
    # Clamp INSIDE log_end — score is always strictly within (0.01, 0.99) when printed
    safe_score = min(max(float(score), 0.01), 0.99)
    print(f"[END] task={task} score={safe_score:.2f} steps={steps}", flush=True)


def run_task(name: str, env: SentinelEnv) -> float:
    print(f"\n{'='*50}")
    print(f"  TASK: {name.upper()}  |  max_steps={env.max_steps}  |  seed={env.seed}")
    print(f"{'='*50}")

    obs          = env.reset()
    total_reward = 0.0
    done         = False
    llm_steps    = 0
    greedy_steps = 0
    steps_taken  = 0

    print(f"[START] task={name} env=sentinel model={MODEL_NAME}", flush=True)
    print(f"Sensors={len(obs.sensors)} | Targets={len(obs.targets)}")

    while not done:
        actions, source = get_actions(obs)
        obs, reward, done, info = env.step_batch(actions)
        total_reward += reward
        steps_taken  += 1
        if source == "llm":
            llm_steps += 1
        else:
            greedy_steps += 1
        assignments_str = ", ".join(f"{a.sensor_id}->{a.target_id}" for a in actions) if actions else "none"

        # ✅ FIXED: matches required format exactly
        # [STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
        print(
            f"[STEP] step={steps_taken} action={assignments_str} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True
        )

    raw_score = grade_episode(total_reward, info["step_count"], num_sensors=env.initial_sensor_count)
    # Clamp strictly within (0.01, 0.99) — validator rejects exactly 0.0 or 1.0
    score = min(max(float(raw_score), 0.01), 0.99)

    print(f"\n  Total Reward : {total_reward:.1f}")
    print(f"  Steps        : {info['step_count']}")
    print(f"  LLM steps    : {llm_steps}  |  Greedy fallback: {greedy_steps}")
    print(f"  Missed HIGH  : {len(info['missed_targets'])}")
    print(f"  SCORE        : {score:.4f}  (strictly in 0.01 – 0.99)")

    # [END] line with clamped score — validator reads this
    log_end(task=name, score=score, steps=steps_taken)

    return score


# ── Multi-agent greedy proposals ─────────────────────────────────────────────
def _greedy_multi_proposals(obs_map: dict) -> list[Proposal]:
    """Coordinated greedy proposals — agents share used_sensors and used_targets globally."""
    proposals: list[Proposal] = []
    used_sensors: set = set()
    used_targets: set = set()
    affinity = {"satellite": "satellite", "drone": "drone", "radar": "radar", "command": None}
    # command agent goes last so it fills gaps
    ordered = [a for a in AGENT_TYPES if a != "command"] + ["command"]
    for agent_id in ordered:
        agent_obs = obs_map[agent_id]
        sensors_raw = agent_obs.sensors if hasattr(agent_obs, "sensors") else []
        targets_raw = agent_obs.targets if hasattr(agent_obs, "targets") else []
        aff = affinity[agent_id]
        my_sensors = [
            s for s in sensors_raw
            if s["available"] and s["id"] not in used_sensors
            and (aff is None or s["type"] == aff)
        ]
        active_targets = sorted(
            [t for t in targets_raw if t["active"] and t["id"] not in used_targets],
            key=lambda t: -t["priority"]
        )
        for sensor in my_sensors:
            for target in active_targets:
                if target["id"] not in used_targets:
                    proposals.append(Proposal(
                        agent_id=agent_id,
                        sensor_id=sensor["id"],
                        target_id=target["id"],
                    ))
                    used_sensors.add(sensor["id"])
                    used_targets.add(target["id"])
                    break
    return proposals


def run_multi_task(name: str, env: AryaXEnv) -> float:
    print(f"\n{'='*50}")
    print(f"  TASK (MULTI): {name.upper()}  |  max_steps={env.max_steps}  |  seed={env.seed}")
    print(f"{'='*50}")

    obs_map      = env.reset()
    negotiation  = NegotiationLayer()
    reward_eng   = RewardEngine()
    total_rewards = {a: 0.0 for a in AGENT_TYPES}
    done          = False
    steps_taken   = 0

    while not done:
        proposals = _greedy_multi_proposals(obs_map)
        obs_map, step_rewards, done, info = env.step_multiagent(proposals)
        steps_taken += 1

        for aid, r in step_rewards.items():
            total_rewards[aid] += r

        if info["conflicts"]:
            types = [c["type"] for c in info["conflicts"]]
            print(
                f"  [CONFLICT] step={steps_taken} types={types} "
                f"conflict_rate={info['conflict_rate']:.3f}",
                flush=True,
            )

    conflict_rate      = info["conflict_rate"]
    coordination_score = 1.0 - conflict_rate
    total_reward       = sum(total_rewards.values())
    scores             = reward_eng.compute_scores(total_rewards, env.max_steps, env.initial_sensor_count)

    print(f"\n  Per-agent rewards : {total_rewards}")
    print(f"  conflict_rate     : {conflict_rate:.3f}")
    print(f"  coordination_score: {coordination_score:.3f}")
    print(f"  efficiency        : {scores['efficiency']:.3f}")
    print(f"  final_score       : {scores['final_score']:.3f}")
    return scores["final_score"]


if __name__ == "__main__":
    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        print(f"[LLM] Using model: {MODEL_NAME}\n")
    else:
        client = None
        print("[WARN] HF_TOKEN not set — running in greedy fallback mode.\n")

    # ── Single-agent tasks ────────────────────────────────────────────
    results = {}
    for task in TASKS:
        env = task["env_fn"]()
        env.seed = task["seed"]
        score = run_task(task["name"], env)
        results[task["name"]] = score

    print(f"\n{'='*50}")
    print("  FINAL SCORES (single-agent)")
    print(f"{'='*50}")
    for name, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {name:<8} {score:.4f}  {bar}")
    print(f"\n  Average: {sum(results.values()) / len(results):.4f}")
    print(f"{'='*50}")

    # ── Multi-agent tasks ─────────────────────────────────────────────
    MULTI_TASKS = [
        {"name": "Easy",   "seed": 42, "max_steps": 20},
        {"name": "Medium", "seed": 7,  "max_steps": 40},
        {"name": "Hard",   "seed": 13, "max_steps": 60},
    ]
    multi_results = {}
    for task in MULTI_TASKS:
        mx_env = AryaXEnv(max_steps=task["max_steps"], seed=task["seed"])
        score  = run_multi_task(task["name"], mx_env)
        multi_results[task["name"]] = score

    print(f"\n{'='*50}")
    print("  FINAL SCORES (multi-agent)")
    print(f"{'='*50}")
    for name, score in multi_results.items():
        bar = "█" * int(score * 20)
        print(f"  {name:<8} {score:.4f}  {bar}")
    print(f"\n  Average: {sum(multi_results.values()) / len(multi_results):.4f}")
    print(f"{'='*50}")