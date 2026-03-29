// ── State ─────────────────────────────────────────────────────────────────────
let totalReward = 0, stepCount = 0, maxSteps = 20, done = false;
let currentTask = 'Easy', currentSeed = 42;
let sensors = [], envTargets = [], customTargets = [];
let episodeLog = [], obsLoaded = false;

// Leaflet map + layer references
let map = null;
const sensorMarkers = {};   // id → L.marker
const targetMarkers = {};   // id → L.marker
const arcLines = [];        // L.polyline[]

// Snapshot of all target metadata (priority etc.) keyed by id — survives deactivation
const targetMeta = {};

// ── Fixed sensor positions (lat, lon) ─────────────────────────────────────────
const SENSOR_POS = {
  S1: [28.6, 77.2],   // Delhi
  S2: [19.0, 72.8],   // Mumbai
  S3: [13.0, 80.3],   // Chennai
  S4: [22.5, 88.3],   // Kolkata
  S5: [17.4, 78.5],   // Hyderabad
};
const CITY_NAMES = {
  S1: 'Delhi', S2: 'Mumbai', S3: 'Chennai', S4: 'Kolkata', S5: 'Hyderabad'
};
const customSensorPos = {};   // id → [lat, lon] when dragged

// ── Threat zones for env targets ──────────────────────────────────────────────
const THREAT_ZONES = [
  [34.5, 74.0], [32.0, 77.5], [28.0, 97.5], [23.5, 91.5],
  [22.0, 69.0], [27.5, 88.5], [30.5, 71.0], [15.0, 74.0]
];

const allTargetPos = {};   // id → [lat, lon]
let nextCustomId = 1;

function getTargetPos(id) {
  if (allTargetPos[id]) return allTargetPos[id];
  let h = 0;
  for (const c of id) h = (h * 31 + c.charCodeAt(0)) & 0xffff;
  const base = THREAT_ZONES[h % THREAT_ZONES.length];
  const pos = [base[0] + (((h >> 4) & 0xff) / 255 - 0.5) * 3,
               base[1] + (((h >> 8) & 0xff) / 255 - 0.5) * 3];
  allTargetPos[id] = pos;
  return pos;
}

function getSensorPos(id) { return customSensorPos[id] || SENSOR_POS[id] || [20, 78]; }

// ── Leaflet init ──────────────────────────────────────────────────────────────
function initMap() {
  map = L.map('map', { center: [22, 80], zoom: 5, zoomControl: true });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 18
  }).addTo(map);
}

// ── Marker helpers ────────────────────────────────────────────────────────────
function sensorIcon(type, available) {
  const colors = { satellite: '#395144', drone: '#4E6C50', radar: '#AA8B56' };
  const c = colors[type] || '#395144';
  const opacity = available ? 1 : 0.4;
  const letter = type[0].toUpperCase();
  return L.divIcon({
    className: '',
    html: `<div style="width:28px;height:28px;border-radius:50%;background:${c};opacity:${opacity};
      border:2px solid #F0EBCE;display:flex;align-items:center;justify-content:center;
      color:#F0EBCE;font-family:'Courier New',monospace;font-size:11px;font-weight:700;
      box-shadow:0 2px 6px rgba(0,0,0,0.4)">${letter}</div>`,
    iconSize: [28, 28], iconAnchor: [14, 14]
  });
}

function threatIcon(priority) {
  const colors = { 3: '#8B2E2E', 2: '#A0522D', 1: '#4E6C50' };
  const sizes  = { 3: 26, 2: 22, 1: 18 };
  const c = colors[priority] || '#4E6C50';
  const s = sizes[priority] || 18;
  const pulse = priority === 3
    ? `<div style="position:absolute;inset:-6px;border-radius:50%;border:2px solid ${c};opacity:0.5;animation:ping 1.2s infinite"></div>` : '';
  return L.divIcon({
    className: '',
    html: `<div style="position:relative;width:${s}px;height:${s}px">
      ${pulse}
      <div style="width:0;height:0;border-left:${s/2}px solid transparent;
        border-right:${s/2}px solid transparent;border-bottom:${s}px solid ${c};
        filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5))"></div>
    </div>`,
    iconSize: [s, s], iconAnchor: [s/2, s]
  });
}

// ── Render sensors on map ─────────────────────────────────────────────────────
function renderSensors(data) {
  sensors = data;

  data.forEach(s => {
    const pos = getSensorPos(s.id);
    if (sensorMarkers[s.id]) {
      sensorMarkers[s.id].setIcon(sensorIcon(s.type, s.available));
      // Don't move marker if user has dragged it — position already updated
    } else {
      const m = L.marker(pos, {
        icon: sensorIcon(s.type, s.available),
        draggable: true,
        title: `${s.id} — ${s.type}`
      }).addTo(map);
      m.bindTooltip(`<b>${s.id}</b><br>${s.type}<br>${CITY_NAMES[s.id] || ''}`, { direction: 'top' });
      m.on('dragend', async e => {
        const ll = e.target.getLatLng();
        customSensorPos[s.id] = [ll.lat, ll.lng];
        // Reverse geocode to get place name
        let placeName = `${ll.lat.toFixed(1)}°N, ${ll.lng.toFixed(1)}°E`;
        try {
          const geo = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${ll.lat}&lon=${ll.lng}&format=json`,
            { headers: { 'Accept-Language': 'en' } }
          ).then(r => r.json());
          const addr = geo.address || {};
          placeName = addr.city || addr.town || addr.village || addr.county || addr.state || placeName;
        } catch (_) {}
        CITY_NAMES[s.id] = placeName;
        m.setTooltipContent(`<b>${s.id}</b><br>${s.type}<br>${placeName}`);
        // Refresh sidebar so dropdown shows new location name
        renderSensors(sensors);
        addLog(`${s.type} sensor moved to ${placeName}`, 'log-neu');
      });
      sensorMarkers[s.id] = m;
    }
  });

  // Update sidebar sensor list
  document.getElementById('sensorList').innerHTML = data.map(s => {
    const dc = s.type === 'satellite' ? 'sat' : s.type === 'drone' ? 'drone' : 'radar';
    return `<div class="sensor-item ${s.available ? 'available' : 'busy'}">
      <span class="sensor-dot ${dc}"></span>
      <span class="sensor-id">${s.id}</span>
      <span class="sensor-type">${s.type}</span>
      <span class="sensor-status ${s.available ? 'on' : 'off'}">${s.available ? 'READY' : 'BUSY'}</span>
    </div>`;
  }).join('');

  document.getElementById('sensorSel').innerHTML = data.filter(s => s.available)
    .map(s => `<option value="${s.id}">${s.id} — ${s.type} (${CITY_NAMES[s.id] || ''})</option>`).join('');
}

// ── Render targets on map ─────────────────────────────────────────────────────
function renderEnvTargets(data) {
  // Snapshot metadata before targets get deactivated
  data.forEach(t => {
    targetMeta[t.id] = { priority: t.priority };
    if (!allTargetPos[t.id]) getTargetPos(t.id);
  });

  // Remove markers for targets no longer in observation
  const activeIds = new Set(data.map(t => t.id));
  Object.keys(targetMarkers).forEach(id => {
    if (!id.startsWith('C') && !activeIds.has(id)) {
      map.removeLayer(targetMarkers[id]);
      delete targetMarkers[id];
    }
  });

  data.filter(t => t.active).forEach(t => {
    const pos = getTargetPos(t.id);
    if (targetMarkers[t.id]) {
      targetMarkers[t.id].setIcon(threatIcon(t.priority));
    } else {
      const lvl = t.priority === 3 ? 'HIGH' : t.priority === 2 ? 'MED' : 'LOW';
      const m = L.marker(pos, { icon: threatIcon(t.priority), title: t.id }).addTo(map);
      m.bindTooltip(`<b>${t.id}</b><br>Priority: ${lvl}`, { direction: 'top' });
      targetMarkers[t.id] = m;
    }
  });

  envTargets = data;
  refreshThreatPanel();
}

// ── Arc (projectile line) ─────────────────────────────────────────────────────
function spawnArc(sensorId, targetId, reward) {
  const sp = getSensorPos(sensorId);
  const tp = getTargetPos(targetId);
  const color = reward > 0 ? '#4E6C50' : '#8B2E2E';
  const line = L.polyline([sp, tp], {
    color, weight: 2, dashArray: '6 4', opacity: 0.85
  }).addTo(map);
  arcLines.push(line);
  // Fade out after 3s
  setTimeout(() => { map.removeLayer(line); }, 3000);
}

// ── Map click to place custom threats ────────────────────────────────────────
let activeTool = 'move';

function setTool(tool) {
  activeTool = tool;
  document.querySelectorAll('.map-tool').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('tool-' + tool);
  if (btn) btn.classList.add('active');
  const hints = {
    'move': 'Drag a sensor marker to reposition it',
    'threat-h': 'Click on map to place a HIGH priority threat',
    'threat-m': 'Click on map to place a MED priority threat',
    'threat-l': 'Click on map to place a LOW priority threat',
  };
  document.getElementById('toolHint').textContent = hints[tool] || '';
  map.getContainer().style.cursor = tool === 'move' ? '' : 'crosshair';
}

// ── API ───────────────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const r = await fetch(path, {
    method, headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  });
  return r.json();
}

// ── Label helpers (for summary) ───────────────────────────────────────────────
function sensorLabel(id) {
  const s = sensors.find(x => x.id === id);
  const city = CITY_NAMES[id] || id;
  return (s ? s.type : 'sensor') + ' based in ' + city;
}

function targetLabel(id) {
  const meta = targetMeta[id] || customTargets.find(x => x.id === id);
  const priority = meta ? meta.priority : null;
  const pos = getTargetPos(id);
  const lvl = priority === 3 ? 'high-priority' : priority === 2 ? 'medium-priority' : priority === 1 ? 'low-priority' : 'unknown';
  return `${lvl} threat at ${pos[0].toFixed(1)}°N, ${pos[1].toFixed(1)}°E`;
}

// ── Threat panel ──────────────────────────────────────────────────────────────
function refreshThreatPanel() {
  const all = [...envTargets, ...customTargets].filter(t => t.active);
  document.getElementById('threatList').innerHTML = all.length
    ? all.map(t => {
        const c = t.priority === 3 ? 'p3' : t.priority === 2 ? 'p2' : 'p1';
        const b = t.priority === 3 ? 'h' : t.priority === 2 ? 'm' : 'l';
        const l = t.priority === 3 ? 'HIGH' : t.priority === 2 ? 'MED' : 'LOW';
        const tag = t.id.startsWith('C') ? '<span style="font-size:0.65em;opacity:0.7"> CUSTOM</span>' : '';
        return `<div class="threat-item ${c}">&#9650; <strong>${t.id}</strong>${tag}
          <span class="threat-badge ${b}">${l}</span></div>`;
      }).join('')
    : '<div class="empty-state">No active threats</div>';

  document.getElementById('targetSel').innerHTML = all
    .sort((a, b) => b.priority - a.priority)
    .map(t => `<option value="${t.id}">${t.id} (p${t.priority}${t.id.startsWith('C') ? ', custom' : ''})</option>`).join('');
}

// ── Log ───────────────────────────────────────────────────────────────────────
function addLog(text, cls) {
  const box = document.getElementById('logBox');
  const d = document.createElement('div');
  d.className = 'log-entry';
  d.innerHTML = `<span class="log-time">${new Date().toLocaleTimeString()}</span><span class="${cls}">${text}</span>`;
  box.prepend(d);
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(reward, info) {
  totalReward += reward;
  stepCount = info.step_count;
  document.getElementById('statStep').textContent     = stepCount;
  document.getElementById('statReward').textContent   = totalReward.toFixed(1);
  document.getElementById('statMissed').textContent   = info.missed_targets.length;
  document.getElementById('statAssigned').textContent = info.assignments.length;
  const best = stepCount * 2, worst = stepCount * -2;
  const score = best === worst ? 0 : Math.max(0, Math.min(1, (totalReward - worst) / (best - worst)));
  document.getElementById('scoreBar').style.width = (score * 100) + '%';
  document.getElementById('scorePct').textContent = score.toFixed(3);
}

function setActiveTask(name) {
  ['easy', 'medium', 'hard'].forEach(n => document.getElementById('btn-' + n).classList.remove('active'));
  document.getElementById('btn-' + name.toLowerCase()).classList.add('active');
}

// ── Episode summary ───────────────────────────────────────────────────────────
function showSummary(info) {
  const best = stepCount * 2, worst = stepCount * -2;
  const score = best === worst ? 0 : Math.max(0, Math.min(1, (totalReward - worst) / (best - worst)));
  const pct = (score * 100).toFixed(1);

  const assignmentLines = info.assignments.map((a, i) =>
    `Step ${i + 1}: The ${sensorLabel(a.sensor)} was assigned to track the ${targetLabel(a.target)}.`
  ).join('\n');

  const missedLines = info.missed_targets.length
    ? `During the episode, ${info.missed_targets.length} high-priority threat(s) went untracked, which significantly impacted the score.`
    : 'All high-priority threats were successfully tracked — no critical blind spots.';

  const verdict = score >= 0.8 ? 'Excellent performance.'
    : score >= 0.5 ? 'Moderate performance — some threats were missed.'
    : 'Poor performance — too many high-priority threats went untracked.';

  const summary = `EPISODE SUMMARY — ${currentTask.toUpperCase()} MISSION
${'─'.repeat(50)}
The mission ran for ${stepCount} steps with ${sensors.length} sensor(s) deployed.

ASSIGNMENTS:
${assignmentLines}

THREAT COVERAGE:
${missedLines}

FINAL SCORE: ${score.toFixed(4)} (${pct}%)
Total reward accumulated: ${totalReward.toFixed(1)}

VERDICT: ${verdict}`;

  document.getElementById('summaryText').textContent = summary;
  document.getElementById('summaryModal').style.display = 'flex';
}

// ── Actions ───────────────────────────────────────────────────────────────────
async function loadTask(steps, seed, name, threatLevel) {
  totalReward = 0; stepCount = 0; maxSteps = steps; done = false;
  currentTask = name; currentSeed = seed;
  episodeLog = [];
  customTargets = []; nextCustomId = 1;
  envTargets = [];
  Object.keys(allTargetPos).forEach(k => delete allTargetPos[k]);
  Object.keys(targetMeta).forEach(k => delete targetMeta[k]);

  // Clear map layers
  Object.values(sensorMarkers).forEach(m => map.removeLayer(m));
  Object.keys(sensorMarkers).forEach(k => delete sensorMarkers[k]);
  Object.values(targetMarkers).forEach(m => map.removeLayer(m));
  Object.keys(targetMarkers).forEach(k => delete targetMarkers[k]);
  arcLines.forEach(l => map.removeLayer(l));
  arcLines.length = 0;

  document.getElementById('logBox').innerHTML = '';
  ['statStep', 'statReward', 'statMissed', 'statAssigned'].forEach(id =>
    document.getElementById(id).textContent = '0');
  document.getElementById('scoreBar').style.width = '0%';
  document.getElementById('scorePct').textContent = '0.000';
  document.getElementById('gradeVal').textContent = '--';
  document.getElementById('gradeSub').textContent = 'Run grader to evaluate';
  setActiveTask(name);

  document.querySelectorAll('.btn-assign,.btn-auto,.btn-run,.btn-grade').forEach(b => b.disabled = true);
  const obs = await api('POST', '/reset', { max_steps: steps });
  currentSeed = obs.seed || seed;
  document.querySelectorAll('.btn-assign,.btn-auto,.btn-run,.btn-grade').forEach(b => b.disabled = false);

  renderSensors(obs.sensors);
  renderEnvTargets(obs.targets);

  document.getElementById('taskLabel').textContent    = name.toUpperCase() + ' — ' + threatLevel;
  document.getElementById('statusPill').innerHTML     = '&#9679; ' + threatLevel;
  document.getElementById('statusPill').className     = 'status-pill active';
  document.getElementById('mapOverlay').style.display = 'none';
  document.getElementById('toolHint').textContent     = 'Drag sensors to reposition | Use toolbar to place custom threats';

  obsLoaded = true;
  addLog(`Mission started: ${name} task with ${obs.sensors.length} sensors and ${obs.targets.length} initial threats.`, 'log-neu');
}

async function manualStep() {
  if (!obsLoaded) return addLog('Load a mission first', 'log-neg');
  if (done) return addLog('Episode complete — reload mission', 'log-neg');
  const sensor_id = document.getElementById('sensorSel').value;
  const target_id = document.getElementById('targetSel').value;
  if (!sensor_id || !target_id) return;
  const r = await api('POST', '/step', { sensor_id, target_id });
  if (r.error) return addLog('ERROR: ' + r.error, 'log-neg');
  done = r.done;
  spawnArc(sensor_id, target_id, r.reward);
  renderSensors(r.observation.sensors);
  renderEnvTargets(r.observation.targets);
  updateStats(r.reward, r.info);
  const c = r.reward > 0 ? 'log-pos' : r.reward < 0 ? 'log-neg' : 'log-neu';
  addLog(`The ${sensorLabel(sensor_id)} was directed to the ${targetLabel(target_id)}. Reward: ${r.reward > 0 ? '+' : ''}${r.reward}${r.done ? ' — Mission complete.' : ''}`, c);
  if (r.done) showSummary(r.info);
}

async function autoStep() {
  if (!obsLoaded) return addLog('Load a mission first', 'log-neg');
  if (done) return addLog('Episode complete — reload mission', 'log-neg');
  const r = await api('POST', '/step/auto');
  if (r.error) return addLog('ERROR: ' + r.error, 'log-neg');
  done = r.done;
  const actions = r.actions || (r.action ? [r.action] : []);
  if (actions.length === 0) {
    if (r.done) showSummary(r.info);
    return;
  }
  actions.forEach(a => spawnArc(a.sensor_id, a.target_id, r.reward));
  renderSensors(r.observation.sensors);
  renderEnvTargets(r.observation.targets);
  updateStats(r.reward, r.info);
  const c = r.reward > 0 ? 'log-pos' : r.reward < 0 ? 'log-neg' : 'log-neu';
  const agentTag = r.agent === 'llm' ? ' <span style="color:#AA8B56">[LLM]</span>' : ' <span style="opacity:0.5">[greedy]</span>';
  const desc = actions.map(a =>
    `The ${sensorLabel(a.sensor_id)} → ${targetLabel(a.target_id)}`
  ).join(' | ');
  addLog(`[Auto]${agentTag} ${desc}. Reward: ${r.reward > 0 ? '+' : ''}${r.reward}${r.done ? ' — Mission complete.' : ''}`, c);
  if (r.done) showSummary(r.info);
}

async function runAll() {
  if (!obsLoaded) return addLog('Load a mission first', 'log-neg');
  if (done) {
    await loadTask(maxSteps, currentSeed, currentTask, 'RUNNING');
  }
  while (!done) {
    await autoStep();
    await new Promise(r => setTimeout(r, 300));
    if (done) break;
  }
}

async function runGrade() {
  document.getElementById('gradeVal').textContent = '...';
  document.getElementById('gradeSub').textContent = 'Computing...';
  const r = await api('POST', '/grade', { max_steps: maxSteps });
  const pct = (r.score * 100).toFixed(1);
  document.getElementById('gradeVal').textContent = r.score.toFixed(4);
  document.getElementById('gradeSub').innerHTML = pct + '% &bull; ' + r.steps + ' steps &bull; reward ' + r.total_reward;
}

function closeSummary() {
  document.getElementById('summaryModal').style.display = 'none';
}

// ── Clock ─────────────────────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-IN', { hour12: false });
}
setInterval(updateClock, 1000); updateClock();

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  initMap();

  // Place custom threats on map click
  map.on('click', e => {
    if (!obsLoaded || activeTool === 'move') return;
    const lat = e.latlng.lat, lon = e.latlng.lng;
    const priority = activeTool === 'threat-h' ? 3 : activeTool === 'threat-m' ? 2 : 1;
    const id = 'C' + nextCustomId++;
    allTargetPos[id] = [lat, lon];
    targetMeta[id] = { priority };
    customTargets.push({ id, priority, active: true });
    api('POST', '/targets/custom', { id, priority, lat, lon });

    const lvl = priority === 3 ? 'HIGH' : priority === 2 ? 'MED' : 'LOW';
    const m = L.marker([lat, lon], { icon: threatIcon(priority) }).addTo(map);
    m.bindTooltip(`<b>${id}</b><br>Priority: ${lvl} (custom)`, { direction: 'top' });
    targetMarkers[id] = m;

    refreshThreatPanel();
    addLog(`Custom ${lvl} threat placed at ${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E`, 'log-neu');
  });

  setTool('move');
});
