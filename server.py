"""
Local test server for SentinelEnv.
Run:  python server.py
Test: curl http://localhost:5000/reset
"""
from flask import Flask, request, jsonify, render_template_string
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
    body = request.get_json(silent=True) or {}
    steps = body.get("max_steps", env.max_steps)
    seed = body.get("seed", env.seed)
    g_env = SentinelEnv(max_steps=steps, seed=seed)
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


UI = """
<!DOCTYPE html>
<html>
<head>
<title>SentinelEnv Dashboard</title>
<style>
  :root {
    --bg: #f6f8fa;
    --bg-card: #ffffff;
    --border: #d0d7de;
    --text: #1f2328;
    --text-muted: #656d76;
    --accent: #0969da;
    --green: #1a7f37;
    --green-bg: #dafbe1;
    --red: #cf222e;
    --red-bg: #ffebe9;
    --yellow: #9a6700;
    --yellow-bg: #fff8c5;
    --sensor-avail-bg: #ddf4ff;
    --sensor-avail-border: #54aeff;
    --sensor-busy-bg: #f6f8fa;
    --sensor-busy-border: #d0d7de;
    --score-track: #eaeef2;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 24px; }
  h1 { color: var(--accent); margin-bottom: 20px; font-size: 1.3em; }
  h2 { color: var(--text-muted); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; font-weight: 600; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card.full { grid-column: 1 / -1; }
  button { background: var(--green); color: #fff; border: none; padding: 7px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85em; margin: 3px 3px 3px 0; font-family: inherit; }
  button:hover { opacity: 0.85; }
  button.secondary { background: var(--bg); color: var(--text); border: 1px solid var(--border); }
  button.danger { background: var(--red); }
  select { background: var(--bg); color: var(--text); border: 1px solid var(--border); padding: 6px 10px; border-radius: 6px; font-size: 0.85em; margin: 3px 3px 3px 0; font-family: inherit; }
  .sensor { display: inline-block; padding: 4px 10px; border-radius: 4px; margin: 3px; font-size: 0.82em; }
  .sensor.available { background: var(--sensor-avail-bg); border: 1px solid var(--sensor-avail-border); color: var(--accent); }
  .sensor.busy { background: var(--sensor-busy-bg); border: 1px solid var(--sensor-busy-border); color: var(--text-muted); }
  .target { display: flex; align-items: center; gap: 8px; padding: 6px 10px; border-radius: 4px; margin: 3px 0; font-size: 0.85em; }
  .p3 { background: var(--red-bg); border-left: 3px solid var(--red); }
  .p2 { background: var(--yellow-bg); border-left: 3px solid var(--yellow); }
  .p1 { background: var(--green-bg); border-left: 3px solid var(--green); }
  .badge { font-size: 0.72em; padding: 2px 7px; border-radius: 10px; font-weight: 600; }
  .badge.high { background: var(--red); color: #fff; }
  .badge.med { background: var(--yellow); color: #fff; }
  .badge.low { background: var(--green); color: #fff; }
  .log { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px; height: 220px; overflow-y: auto; font-size: 0.8em; font-family: monospace; }
  .log-entry { padding: 3px 0; border-bottom: 1px solid var(--border); }
  .reward-pos { color: var(--green); font-weight: 600; }
  .reward-neg { color: var(--red); font-weight: 600; }
  .reward-zero { color: var(--text-muted); }
  .score-bar { height: 10px; background: var(--score-track); border-radius: 6px; margin-top: 8px; overflow: hidden; }
  .score-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--green)); transition: width 0.5s; }
  .stat { font-size: 1.6em; color: var(--accent); font-weight: 700; }
  .stat-label { font-size: 0.75em; color: var(--text-muted); margin-top: 2px; }
  .task-btn.easy { background: var(--green); }
  .task-btn.medium { background: var(--yellow); }
  .task-btn.hard { background: var(--red); }
  #stepInfo { color: var(--text-muted); font-size: 0.82em; margin-top: 8px; }
  #taskInfo { color: var(--text-muted); font-size: 0.82em; margin-top: 10px; }
</style>
</head>
<body>
<h1>🛰️ SentinelEnv — Sensor Allocation Dashboard</h1>

<div class="grid">
  <!-- Task Selection -->
  <div class="card">
    <h2>Task</h2>
    <div style="margin-top:10px">
      <button class="task-btn easy" onclick="resetTask(20,42,'Easy')">Easy (20 steps)</button>
      <button class="task-btn medium" onclick="resetTask(40,7,'Medium')">Medium (40 steps)</button>
      <button class="task-btn hard" onclick="resetTask(60,13,'Hard')">Hard (60 steps)</button>
    </div>
    <div id="taskInfo">No task loaded</div>
  </div>

  <!-- Stats -->
  <div class="card">
    <h2>Episode Stats</h2>
    <div style="display:flex;gap:24px;margin-top:10px">
      <div><div class="stat" id="statStep">-</div><div class="stat-label">Step</div></div>
      <div><div class="stat" id="statReward">-</div><div class="stat-label">Total Reward</div></div>
      <div><div class="stat" id="statScore">-</div><div class="stat-label">Score</div></div>
    </div>
    <div class="score-bar"><div class="score-fill" id="scoreBar" style="width:0%"></div></div>
    <div id="stepInfo">Reset to start</div>
  </div>

  <!-- Grade -->
  <div class="card">
    <h2>Full Episode Grade</h2>
    <button onclick="runGrade()" style="margin-top:10px">▶ Run Grader</button>
    <div id="gradeResult" class="reward-zero" style="margin-top:10px;font-size:0.85em">—</div>
  </div>

  <!-- Sensors -->
  <div class="card">
    <h2>Sensors</h2>
    <div id="sensors" style="margin-top:8px">—</div>
  </div>

  <!-- Targets -->
  <div class="card">
    <h2>Active Targets</h2>
    <div id="targets" style="margin-top:8px">—</div>
  </div>

  <!-- Manual Action -->
  <div class="card">
    <h2>Manual Action</h2>
    <div style="margin-top:10px">
      <select id="sensorSel"><option>— sensor —</option></select>
      <select id="targetSel"><option>— target —</option></select>
    </div>
    <div style="margin-top:8px">
      <button onclick="manualStep()">Assign</button>
      <button onclick="autoStep()">Auto Step</button>
      <button onclick="runAll()" class="secondary">Run All Steps</button>
    </div>
  </div>

  <!-- Log -->
  <div class="card full">
    <h2>Step Log</h2>
    <div class="log" id="log"></div>
  </div>
</div>

<script>
let totalReward = 0, stepCount = 0, maxSteps = 10, done = false, currentTask = 'Easy';

async function api(method, path, body) {
  const r = await fetch(path, {
    method, headers: {'Content-Type':'application/json'},
    body: body ? JSON.stringify(body) : undefined
  });
  return r.json();
}

function renderObs(obs) {
  // Sensors
  document.getElementById('sensors').innerHTML = obs.sensors.map(s =>
    `<span class="sensor ${s.available?'available':'busy'}">
      ${s.id} <small>${s.type}</small> ${s.available?'✓':'✗'}
    </span>`
  ).join('');

  // Targets
  const active = obs.targets.filter(t => t.active);
  document.getElementById('targets').innerHTML = active.length ? active.map(t => {
    const cls = t.priority===3?'p3':t.priority===2?'p2':'p1';
    const badge = t.priority===3?'high':t.priority===2?'med':'low';
    const label = t.priority===3?'HIGH':t.priority===2?'MED':'LOW';
    return `<div class="target ${cls}">${t.id} <span class="badge ${badge}">${label}</span></div>`;
  }).join('') : '<span class="reward-zero">No active targets</span>';

  // Selects
  const ss = document.getElementById('sensorSel');
  const ts = document.getElementById('targetSel');
  ss.innerHTML = obs.sensors.filter(s=>s.available).map(s=>`<option value="${s.id}">${s.id} (${s.type})</option>`).join('');
  ts.innerHTML = active.sort((a,b)=>b.priority-a.priority).map(t=>{
    const label = t.priority===3?'🔴':t.priority===2?'🟡':'🟢';
    return `<option value="${t.id}">${label} ${t.id} (p${t.priority})</option>`;
  }).join('');

  document.getElementById('statStep').textContent = obs.timestep;
}

function logEntry(text, rewardClass) {
  const log = document.getElementById('log');
  const d = document.createElement('div');
  d.className = 'log-entry';
  d.innerHTML = `<span class="reward-zero">[${new Date().toLocaleTimeString()}]</span> <span class="${rewardClass}">${text}</span>`;
  log.prepend(d);
}

function updateStats(reward, info) {
  totalReward += reward;
  stepCount = info.step_count;
  document.getElementById('statReward').textContent = totalReward.toFixed(1);
  const score = Math.max(0, Math.min(1, (totalReward - stepCount*-12) / (stepCount*10 - stepCount*-12)));
  document.getElementById('statScore').textContent = score.toFixed(3);
  document.getElementById('scoreBar').style.width = (score*100)+'%';
  document.getElementById('stepInfo').textContent =
    `Step ${stepCount}/${maxSteps} | Missed: ${info.missed_targets.length} high-priority`;
}

async function resetTask(steps, seed, name) {
  totalReward = 0; stepCount = 0; maxSteps = steps; done = false; currentTask = name;
  document.getElementById('log').innerHTML = '';
  document.getElementById('statReward').textContent = '0';
  document.getElementById('statScore').textContent = '0';
  document.getElementById('scoreBar').style.width = '0%';
  const obs = await api('POST', '/reset', {max_steps: steps, seed});
  renderObs(obs);
  document.getElementById('taskInfo').textContent = `${name} | ${steps} steps | seed=${seed}`;
  logEntry(`🔄 Reset — ${name} task | ${obs.sensors.length} sensors | ${obs.targets.length} targets`, 'reward-zero');
}

async function manualStep() {
  if (done) return logEntry('Episode done — reset first', 'reward-neg');
  const sensor_id = document.getElementById('sensorSel').value;
  const target_id = document.getElementById('targetSel').value;
  if (!sensor_id || !target_id) return;
  const r = await api('POST', '/step', {sensor_id, target_id});
  if (r.error) return logEntry('Error: '+r.error, 'reward-neg');
  done = r.done;
  renderObs(r.observation);
  updateStats(r.reward, r.info);
  const rc = r.reward>0?'reward-pos':r.reward<0?'reward-neg':'reward-zero';
  logEntry(`${sensor_id} → ${target_id} | reward: ${r.reward>0?'+':''}${r.reward} ${r.done?'✅ DONE':''}`, rc);
}

async function autoStep() {
  if (done) return logEntry('Episode done — reset first', 'reward-neg');
  const r = await api('POST', '/step/auto');
  if (r.error) return logEntry('Error: '+r.error, 'reward-neg');
  done = r.done;
  renderObs(r.observation);
  updateStats(r.reward, r.info);
  const rc = r.reward>0?'reward-pos':r.reward<0?'reward-neg':'reward-zero';
  logEntry(`AUTO: ${r.action.sensor_id} → ${r.action.target_id} | reward: ${r.reward>0?'+':''}${r.reward} ${r.done?'✅ DONE':''}`, rc);
}

async function runAll() {
  if (done) await resetTask(maxSteps, 42, currentTask);
  while (!done) { await autoStep(); await new Promise(r=>setTimeout(r,300)); }
}

async function runGrade() {
  document.getElementById('gradeResult').textContent = 'Running...';
  const r = await api('POST', '/grade', {max_steps: maxSteps});
  const pct = (r.score*100).toFixed(1);
  document.getElementById('gradeResult').innerHTML =
    `Score: <strong class="reward-pos">${r.score}</strong> (${pct}%)<br>Reward: ${r.total_reward} | Steps: ${r.steps}`;
}

// Auto-load easy task on page open
resetTask(20, 42, 'Easy');
</script>
</body>
</html>
"""

@app.get("/ui")
def ui():
    return render_template_string(UI)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
