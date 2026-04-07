"""
SentinelEnv server — Flask API + dashboard UI.
Run: python server.py
"""
import os
import json
import random as _random
from flask import Flask, request, jsonify, render_template
from env import SentinelEnv
from env.models import Action

app = Flask(__name__)

env = SentinelEnv(max_steps=10, seed=42)
obs = None
_target_positions: dict = {}

# ── LLM client (optional — falls back to greedy if token not set) ─────────────
_llm_client = None
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    try:
        from openai import OpenAI
        _llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        print(f"[LLM] Connected to {MODEL_NAME}")
    except Exception as e:
        print(f"[LLM] Failed to init client: {e}. Using greedy fallback.")
else:
    print("[LLM] No HF_TOKEN set — using greedy fallback.")


def _build_prompt(observation) -> str:
    sensors = "\n".join(
        f"  - id={s.id} type={s.type} range={s.range}km available={s.available}"
        for s in observation.sensors if s.available
    )
    targets = "\n".join(
        f"  - id={t.id} priority={t.priority} active={t.active}"
        for t in observation.targets if t.active
    )
    n = sum(1 for s in observation.sensors if s.available)
    return f"""You are a military sensor allocation AI. Assign ALL available sensors to threats.
Priority 3=HIGH (missile/critical), 2=MED (border movement), 1=LOW (airspace).
Always cover HIGH priority threats first. Each sensor must go to a DIFFERENT target.

Timestep: {observation.timestep}

Available Sensors ({n}):
{sensors}

Active Threats:
{targets}

Respond ONLY with a JSON array of assignments, one per available sensor:
[{{"sensor_id": "S1", "target_id": "T0_1"}}, {{"sensor_id": "S2", "target_id": "T0_2"}}]
"""


def _parse_llm_actions(text: str, observation) -> list[Action]:
    """Parse LLM response into validated Action list."""
    try:
        start = text.find("[")
        end   = text.rfind("]") + 1
        data  = json.loads(text[start:end])
        valid_sensors = {s.id for s in observation.sensors if s.available}
        valid_targets = {t.id for t in observation.targets if t.active}
        actions, used_sensors, used_targets = [], set(), set()
        for item in data:
            sid, tid = item.get("sensor_id"), item.get("target_id")
            if (sid in valid_sensors and tid in valid_targets
                    and sid not in used_sensors and tid not in used_targets):
                actions.append(Action(sensor_id=sid, target_id=tid))
                used_sensors.add(sid)
                used_targets.add(tid)
        return actions
    except Exception:
        return []


def _greedy_actions(observation) -> list[Action]:
    """Fallback: pair each sensor to highest-priority unassigned target."""
    available = [s for s in observation.sensors if s.available]
    targets   = sorted([t for t in observation.targets if t.active], key=lambda t: -t.priority)
    actions, used = [], set()
    for sensor in available:
        for target in targets:
            if target.id not in used:
                actions.append(Action(sensor_id=sensor.id, target_id=target.id))
                used.add(target.id)
                break
    return actions


def _get_actions(observation) -> tuple[list[Action], str]:
    """Try LLM first, fall back to greedy. Returns (actions, source)."""
    if _llm_client:
        try:
            prompt   = _build_prompt(observation)
            response = _llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0
            )
            raw     = response.choices[0].message.content.strip()
            actions = _parse_llm_actions(raw, observation)
            if actions:
                return actions, "llm"
            print(f"[LLM] Bad response, falling back. Raw: {raw!r}")
        except Exception as e:
            print(f"[LLM] Error: {e}. Falling back to greedy.")
    return _greedy_actions(observation), "greedy"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return jsonify({
        "status": "ok",
        "obs_ready": obs is not None,
        "llm_enabled": _llm_client is not None,
        "model": MODEL_NAME if _llm_client else None
    })


@app.post("/reset")
def reset():
    global obs, _target_positions
    body      = request.get_json(silent=True) or {}
    seed      = body.get("seed") or _random.randint(1, 99999)
    max_steps = body.get("max_steps", 10)
    env.seed      = seed
    env.max_steps = max_steps
    _target_positions = {}
    obs = env.reset()
    return jsonify({**obs.model_dump(), "seed": seed})


@app.post("/step")
def step():
    global obs
    if obs is None:
        return jsonify({"error": "Call /reset first"}), 400
    body      = request.get_json(silent=True) or {}
    sensor_id = body.get("sensor_id")
    target_id = body.get("target_id")
    if not sensor_id or not target_id:
        return jsonify({"error": "Provide sensor_id and target_id"}), 400
    action = Action(sensor_id=sensor_id, target_id=target_id)
    obs, reward, done, info = env.step(action)
    return jsonify({"observation": obs.model_dump(), "reward": reward, "done": done, "info": info})


@app.post("/step/auto")
def step_auto():
    global obs
    if obs is None:
        return jsonify({"error": "Call /reset first"}), 400

    available = [s for s in obs.sensors if s.available]
    active    = [t for t in obs.targets if t.active]

    if not available or not active:
        obs, reward, done, info = env.step_batch([])
        return jsonify({"actions": [], "action": None, "agent": "idle",
                        "observation": obs.model_dump(), "reward": reward, "done": done, "info": info})

    actions, source = _get_actions(obs)
    obs, total_reward, done, info = env.step_batch(actions)

    return jsonify({
        "actions": [a.model_dump() for a in actions],
        "action":  actions[0].model_dump() if actions else None,
        "agent":   source,   # "llm" or "greedy" — frontend can show this
        "observation": obs.model_dump(),
        "reward":  total_reward,
        "done":    done,
        "info":    info
    })


@app.get("/state")
def state():
    if obs is None:
        return jsonify({"error": "Call /reset first"}), 400
    return jsonify(obs.model_dump())


@app.post("/targets/custom")
def register_custom_target():
    global obs
    if obs is None:
        return jsonify({"error": "Call /reset first"}), 400
    body     = request.get_json(silent=True) or {}
    tid      = body.get("id")
    priority = body.get("priority", 2)
    lat      = body.get("lat")
    lon      = body.get("lon")
    if not tid or lat is None or lon is None:
        return jsonify({"error": "Provide id, lat, lon"}), 400
    from env.models import Target
    _target_positions[tid] = [lat, lon]
    env.targets.append(Target(id=tid, priority=priority, active=True))
    obs = env.state()
    return jsonify({"ok": True, "id": tid})


@app.post("/grade")
def grade():
    from tasks.grader import grade_episode
    body  = request.get_json(silent=True) or {}
    steps = body.get("max_steps", env.max_steps)
    seed  = body.get("seed") or _random.randint(1, 99999)
    g_env = SentinelEnv(max_steps=steps, seed=seed)
    g_obs = g_env.reset()
    total_reward, done = 0.0, False
    while not done:
        actions, _ = _get_actions(g_obs)
        g_obs, reward, done, info = g_env.step_batch(actions)
        total_reward += reward
    score = grade_episode(total_reward, info["step_count"], num_sensors=g_env.initial_sensor_count)
    score = max(0.01, min(0.99, score))
    return jsonify({"score": score, "total_reward": total_reward,
                    "steps": info["step_count"], "seed": seed})


@app.get("/")
@app.get("/ui")
def ui():
    return render_template("dashboard.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
