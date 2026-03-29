<<<<<<< HEAD
=======
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

>>>>>>> round1-submission
import os
import json
from openai import OpenAI
from env import SentinelEnv
from env.models import Action
<<<<<<< HEAD

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def build_prompt(obs) -> str:
    sensors = "\n".join(
        f"  - {s.id} | type={s.type} | range={s.range} | available={s.available}"
        for s in obs.sensors
    )
    targets = "\n".join(
        f"  - {t.id} | priority={t.priority} | active={t.active}"
        for t in obs.targets if t.active
    )
    return f"""You are a sensor allocation agent. Assign ONE sensor to ONE target.

Timestep: {obs.timestep}

Sensors:
{sensors}

Active Targets (priority: 3=high, 2=medium, 1=low):
{targets}

Respond ONLY with valid JSON in this exact format:
{{"sensor_id": "<sensor_id>", "target_id": "<target_id>"}}
"""


def parse_action(response_text: str, obs) -> Action | None:
    try:
        # Extract JSON from response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        data = json.loads(response_text[start:end])
        action = Action(sensor_id=data["sensor_id"], target_id=data["target_id"])

        # Validate against current state
        valid_sensors = {s.id for s in obs.sensors if s.available}
        valid_targets = {t.id for t in obs.targets if t.active}
        if action.sensor_id in valid_sensors and action.target_id in valid_targets:
            return action
    except Exception:
        pass
    return None


def fallback_action(obs) -> Action | None:
    # Assign first available sensor to highest-priority active target
    available = [s for s in obs.sensors if s.available]
    active = sorted([t for t in obs.targets if t.active], key=lambda t: -t.priority)
    if available and active:
        return Action(sensor_id=available[0].id, target_id=active[0].id)
    return None


def run():
    env = SentinelEnv(max_steps=10, seed=42)
    obs = env.reset()
    print(f"[RESET] Step={obs.timestep} | Sensors={len(obs.sensors)} | Targets={len(obs.targets)}")

    done = False
    total_reward = 0.0

    while not done:
        prompt = build_prompt(obs)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip()
            action = parse_action(raw, obs)
            if action is None:
                print(f"  [WARN] Invalid LLM response, using fallback. Raw: {raw!r}")
                action = fallback_action(obs)
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}. Using fallback.")
            action = fallback_action(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(
            f"[STEP {info['step_count']}] "
            f"Action={action} | Reward={reward:.1f} | Done={done}"
        )

    print(f"\n[DONE] Total Reward={total_reward:.1f} | Steps={info['step_count']}")
    print(f"  Assignments : {info['assignments']}")
    print(f"  Missed High : {info['missed_targets']}")


if __name__ == "__main__":
    run()
=======
from tasks.grader import grade_episode
from tasks.easy_task import get_easy_env
from tasks.medium_task import get_medium_env
from tasks.hard_task import get_hard_env

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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


def get_actions(obs) -> tuple[list[Action], str]:
    if HF_TOKEN:
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


def run_task(name: str, env: SentinelEnv) -> float:
    print(f"\n{'='*50}")
    print(f"  TASK: {name.upper()}  |  max_steps={env.max_steps}  |  seed={env.seed}")
    print(f"{'='*50}")

    obs          = env.reset()
    total_reward = 0.0
    done         = False
    llm_steps    = 0
    greedy_steps = 0

    print(f"  [RESET] Sensors={len(obs.sensors)} | Targets={len(obs.targets)}")

    while not done:
        actions, source = get_actions(obs)
        obs, reward, done, info = env.step_batch(actions)
        total_reward += reward

        if source == "llm":
            llm_steps += 1
        else:
            greedy_steps += 1

        assignments_str = ", ".join(
            f"{a.sensor_id}→{a.target_id}" for a in actions
        ) if actions else "none"
        print(f"  [STEP {info['step_count']:>2}] [{source.upper():>6}] "
              f"Assignments: {assignments_str} | Reward: {reward:+.1f}")

    score = grade_episode(total_reward, info["step_count"])

    print(f"\n  Total Reward : {total_reward:.1f}")
    print(f"  Steps        : {info['step_count']}")
    print(f"  LLM steps    : {llm_steps}  |  Greedy fallback: {greedy_steps}")
    print(f"  Missed HIGH  : {len(info['missed_targets'])}")
    print(f"  SCORE        : {score:.4f}  (0.0 – 1.0)")

    return score


if __name__ == "__main__":
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set — running in greedy fallback mode.\n")
    else:
        print(f"[LLM] Using model: {MODEL_NAME}\n")

    results = {}
    for task in TASKS:
        env = task["env_fn"]()
        env.seed = task["seed"]   # fix seed for reproducibility
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
>>>>>>> round1-submission
