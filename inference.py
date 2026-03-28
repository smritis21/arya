import os
import json
from openai import OpenAI
from env import SentinelEnv
from env.models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
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
