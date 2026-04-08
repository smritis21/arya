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
    # Clamp INSIDE log_end — score is always strictly within (0, 1) when printed
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
        assignments_str = ", ".join(f"{a.sensor_id}→{a.target_id}" for a in actions) if actions else "none"
        print(f"[STEP] [{source.upper()}] Assignments: {assignments_str} | Reward: {reward:+.1f}", flush=True)

    raw_score = grade_episode(total_reward, info["step_count"], num_sensors=env.initial_sensor_count)
    # Clamp strictly within (0.0, 1.0) — validator rejects exactly 0.0 or 1.0
    score = min(max(float(raw_score), 0.01), 0.99)

    print(f"\n  Total Reward : {total_reward:.1f}")
    print(f"  Steps        : {info['step_count']}")
    print(f"  LLM steps    : {llm_steps}  |  Greedy fallback: {greedy_steps}")
    print(f"  Missed HIGH  : {len(info['missed_targets'])}")
    print(f"  SCORE        : {score:.4f}  (strictly in 0.01 – 0.99)")

    # [END] line with clamped score — validator reads this
    log_end(task=name, score=score, steps=steps_taken)

    return score


if __name__ == "__main__":
    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        print(f"[LLM] Using model: {MODEL_NAME}\n")
    else:
        client = None
        print("[WARN] HF_TOKEN not set — running in greedy fallback mode.\n")

    results = {}
    for task in TASKS:
        env = task["env_fn"]()
        env.seed = task["seed"]
        score = run_task(task["name"], env)
        results[task["name"]] = score

    print(f"\n{'='*50}")
    print("  FINAL SCORES")
    print(f"{'='*50}")
    for name, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {name:<8} {score:.4f}  {bar}")
    print(f"\n  Average: {sum(results.values()) / len(results):.4f}")
    print(f"{'='*50}")