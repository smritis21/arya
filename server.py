"""
SentinelEnv / Arya-X server — Flask API + dashboard UI.
Single-agent endpoints preserved; multi-agent endpoints added under /reset_multi,
/step_multi, /auto_multi.
Run: python server.py
"""
import os
import json
import random as _random
from flask import Flask, request, jsonify, render_template
from env import SentinelEnv
from env.models import Action
from env.multiagent import AryaXEnv, Proposal, AGENT_TYPES

app = Flask(__name__)

# ── Single-agent env (existing) ───────────────────────────────────────────────
env = SentinelEnv(max_steps=10, seed=42)
obs = None
_target_positions: dict = {}

# ── Multi-agent env ───────────────────────────────────────────────────────────
mx_env = AryaXEnv(max_steps=10, seed=42)
mx_obs = None   # Dict[str, AgentObservation] | None

# ── LLM client (optional) ─────────────────────────────────────────────────────
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


# ── Single-agent helpers (unchanged) ─────────────────────────────────────────
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


# ── Multi-agent helpers ───────────────────────────────────────────────────────
def _build_multi_prompt(agent_id: str, agent_obs) -> str:
    my_sensors = [
        s for s in agent_obs.sensors
        if s["available"] and (agent_id == "command" or s["type"] == agent_id)
    ]
    sensors = "\n".join(
        f"  - id={s['id']} type={s['type']} range={s['range']}km"
        for s in my_sensors
    )
    targets = "\n".join(
        f"  - id={t['id']} priority={t['priority']}"
        for t in agent_obs.targets if t["active"]
    )
    if not my_sensors:
        return ""
    sensor_ids = [s['id'] for s in my_sensors]
    target_ids = [t['id'] for t in agent_obs.targets if t['active']]
    return f"""You are the {agent_id} agent. Assign ONLY your sensors to threats.
Your sensors: {sensor_ids}
Active threat IDs: {target_ids}
Priority 3=HIGH, 2=MED, 1=LOW. Cover HIGH first. Each sensor to a DIFFERENT target.

Timestep: {agent_obs.timestep}

Your Sensors:
{sensors}

Active Threats:
{targets}

Respond ONLY with a JSON array using ONLY the sensor IDs and target IDs listed above:
[{{"sensor_id": "{sensor_ids[0]}", "target_id": "{target_ids[0] if target_ids else 'T0_1'}"}}]
"""


def _greedy_proposals(agent_id: str, agent_obs, used_sensors: set, used_targets: set = None) -> list[Proposal]:
    """Greedy proposals for one agent — only claim sensors matching agent type."""
    if used_targets is None:
        used_targets = set()
    my_sensors = [
        s for s in agent_obs.sensors
        if s["available"] and s["id"] not in used_sensors
        and (agent_id == "command" or s["type"] == agent_id)
    ]
    targets = sorted(
        [t for t in agent_obs.targets if t["active"]],
        key=lambda t: -t["priority"]
    )
    proposals = []
    for sensor in my_sensors:
        for target in targets:
            if target["id"] not in used_targets:
                proposals.append(Proposal(
                    agent_id=agent_id,
                    sensor_id=sensor["id"],
                    target_id=target["id"]
                ))
                used_targets.add(target["id"])
                break
    return proposals


def _get_multi_proposals(agent_obs_map) -> tuple[list[Proposal], str]:
    """Build proposals from all agents. Returns (proposals, source)."""
    import sys
    proposals: list[Proposal] = []
    used_sensors: set = set()
    print(f"[_get_multi_proposals] called, llm_client={_llm_client is not None}", flush=True)

    if _llm_client:
        try:
            for agent_id in AGENT_TYPES:
                agent_obs = agent_obs_map[agent_id]
                prompt = _build_multi_prompt(agent_id, agent_obs)
                if not prompt:
                    continue
                print(f"[LLM multi] calling LLM for {agent_id}", flush=True)
                response = _llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=128,
                    temperature=0.0
                )
                raw = response.choices[0].message.content.strip()
                print(f"[LLM multi] {agent_id} response: {raw[:200]}", flush=True)
                start, end = raw.find("["), raw.rfind("]") + 1
                if start == -1 or end == 0:
                    print(f"[LLM multi] {agent_id} no JSON array found", flush=True)
                    continue
                items = json.loads(raw[start:end])
                valid_sensors = {
                    s["id"] for s in agent_obs.sensors
                    if s["available"] and (agent_id == "command" or s["type"] == agent_id)
                }
                valid_targets = {t["id"] for t in agent_obs.targets if t["active"]}
                for item in items:
                    sid, tid = item.get("sensor_id"), item.get("target_id")
                    if sid in valid_sensors and tid in valid_targets and sid not in used_sensors:
                        proposals.append(Proposal(agent_id=agent_id, sensor_id=sid, target_id=tid))
                        used_sensors.add(sid)
            if proposals:
                print(f"[LLM multi] Generated {len(proposals)} proposals", flush=True)
                return proposals, "llm"
            print("[LLM multi] No valid proposals, falling back to greedy", flush=True)
        except Exception as e:
            print(f"[LLM multi] Error: {e}. Falling back to greedy.", flush=True)

    # Greedy fallback — partition both sensors and targets across agents
    used_sensors = set()
    used_targets: set = set()
    for agent_id in AGENT_TYPES:
        agent_obs = agent_obs_map[agent_id]
        for p in _greedy_proposals(agent_id, agent_obs, used_sensors, used_targets):
            proposals.append(p)
            used_sensors.add(p.sensor_id)
            used_targets.add(p.target_id)

    return proposals, "greedy"


# ── Single-agent routes (unchanged) ──────────────────────────────────────────
@app.get("/status")
def status():
    return jsonify({
        "status":       "ok",
        "obs_ready":    obs is not None,
        "mx_obs_ready": mx_obs is not None,
        "llm_enabled":  _llm_client is not None,
        "model":        MODEL_NAME if _llm_client else None
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
        "actions":     [a.model_dump() for a in actions],
        "action":      actions[0].model_dump() if actions else None,
        "agent":       source,
        "observation": obs.model_dump(),
        "reward":      total_reward,
        "done":        done,
        "info":        info
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


# ── Multi-agent routes ────────────────────────────────────────────────────────
@app.post("/reset_multi")
def reset_multi():
    global mx_obs
    body      = request.get_json(silent=True) or {}
    seed      = body.get("seed") or _random.randint(1, 99999)
    max_steps = body.get("max_steps", 10)
    mx_env.seed      = seed
    mx_env.max_steps = max_steps
    mx_obs = mx_env.reset()
    return jsonify({
        "seed":            seed,
        "max_steps":       max_steps,
        "observations":    {k: v.to_dict() for k, v in mx_obs.items()},
        "conflict_rate":   0.0,
        "per_agent_rewards": {a: 0.0 for a in AGENT_TYPES},
    })


@app.post("/step_multi")
def step_multi():
    global mx_obs
    if mx_obs is None:
        return jsonify({"error": "Call /reset_multi first"}), 400

    body = request.get_json(silent=True) or {}
    raw_proposals = body.get("proposals", [])

    proposals = [
        Proposal(
            agent_id=p["agent_id"],
            sensor_id=p["sensor_id"],
            target_id=p["target_id"]
        )
        for p in raw_proposals
        if p.get("agent_id") and p.get("sensor_id") and p.get("target_id")
    ]

    # Step env with proposals — env runs negotiation internally
    mx_obs, step_rewards, done, info = mx_env.step_multiagent(proposals)

    # Extract conflict info from env's internal negotiation layer
    conflict_rate = mx_env.negotiation.get_conflict_rate()
    conflicts = info.get("conflicts", [])

    return jsonify({
        "observations":      {k: v.to_dict() for k, v in mx_obs.items()},
        "step_rewards":      step_rewards,
        "per_agent_rewards": dict(mx_env._agent_rewards),
        "done":              done,
        "info":              info,
        "conflict_rate":     round(conflict_rate, 4),
        "conflicts":         conflicts,
        "override_invoked":  False,
    })


@app.post("/auto_multi")
def auto_multi():
    global mx_obs
    print(f"[auto_multi] called, mx_obs is None: {mx_obs is None}", flush=True)
    if mx_obs is None:
        return jsonify({"error": "Call /reset_multi first"}), 400

    proposals, source = _get_multi_proposals(mx_obs)

    # Step env with proposals — env runs negotiation internally
    mx_obs, step_rewards, done, info = mx_env.step_multiagent(proposals)

    # Extract conflict info from env's internal negotiation layer
    conflict_rate = mx_env.negotiation.get_conflict_rate()
    conflicts = info.get("conflicts", [])

    return jsonify({
        "proposals":         [{"agent_id": p.agent_id, "sensor_id": p.sensor_id,
                               "target_id": p.target_id} for p in proposals],
        "agent":             source,
        "observations":      {k: v.to_dict() for k, v in mx_obs.items()},
        "step_rewards":      step_rewards,
        "per_agent_rewards": dict(mx_env._agent_rewards),
        "done":              done,
        "info":              info,
        "conflict_rate":     round(conflict_rate, 4),
        "conflicts":         conflicts,
        "override_invoked":  False,
    })


# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/ui")
def ui():
    return render_template("dashboard.html")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
