"""
Local test server for SentinelEnv.
Run:  python server.py
Test: curl http://localhost:5000/reset
"""
from flask import Flask, request, jsonify
from env import SentinelEnv
from env.models import Action

app = Flask(__name__)

env = SentinelEnv(max_steps=10, seed=42)
obs = None


@app.get("/status")
def status():
    return jsonify({"status": "ok", "obs_ready": obs is not None})


@app.post("/reset")
def reset():
    global obs
    seed = request.json.get("seed", 42) if request.is_json else 42
    max_steps = request.json.get("max_steps", 10) if request.is_json else 10
    env.seed = seed
    env.max_steps = max_steps
    obs = env.reset()
    return jsonify(obs.model_dump())


@app.post("/step")
def step():
    global obs
    if obs is None:
        return jsonify({"error": "Call /reset first"}), 400
    body = request.get_json(silent=True) or {}
    sensor_id = body.get("sensor_id")
    target_id = body.get("target_id")
    if not sensor_id or not target_id:
        return jsonify({"error": "Provide sensor_id and target_id"}), 400

    action = Action(sensor_id=sensor_id, target_id=target_id)
    obs, reward, done, info = env.step(action)
    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    })


@app.post("/step/auto")
def step_auto():
    global obs
    if obs is None:
        return jsonify({"error": "Call /reset first"}), 400
    # Pick best available sensor → highest priority active target
    sensor = next((s for s in obs.sensors if s.available), None)
    target = sorted([t for t in obs.targets if t.active], key=lambda t: -t.priority)
    target = target[0] if target else None
    if not sensor or not target:
        return jsonify({"error": "No available sensors or active targets"}), 400
    action = Action(sensor_id=sensor.id, target_id=target.id)
    obs, reward, done, info = env.step(action)
    return jsonify({
        "action": action.model_dump(),
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    })


@app.get("/state")
def state():
    if obs is None:
        return jsonify({"error": "Call /reset first"}), 400
    return jsonify(obs.model_dump())


@app.post("/grade")
def grade():
    """Run a full episode with fallback policy and return a graded score."""
    from tasks.grader import grade_episode
    g_env = SentinelEnv(max_steps=10, seed=42)
    g_obs = g_env.reset()
    total_reward, done = 0.0, False
    while not done:
        sensor = next((s for s in g_obs.sensors if s.available), None)
        target = sorted([t for t in g_obs.targets if t.active], key=lambda t: -t.priority)
        target = target[0] if target else None
        action = Action(sensor_id=sensor.id, target_id=target.id) if sensor and target else None
        g_obs, reward, done, info = g_env.step(action)
        total_reward += reward
    score = grade_episode(total_reward, info["step_count"])
    return jsonify({"score": score, "total_reward": total_reward, "steps": info["step_count"]})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
