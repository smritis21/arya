// ── Mode ───────────────────────────────────────────────────────────────────────
let currentMode = 'single';   // 'single' | 'multi'

// ── State ─────────────────────────────────────────────────────────────────────
let totalReward = 0, stepCount = 0, maxSteps = 20, done = false;
let currentTask = 'Easy', currentSeed = 42;
let sensors = [], envTargets = [], customTargets = [];
let episodeLog = [], obsLoaded = false;

// Multi-agent state
let mxDone = false;
let mxStepCount = 0;
let agentRewardsCumulative = { satellite: 0, drone: 0, radar: 0, command: 0 };
let conflictLog = [];
let activeThreats = [];

// Training comparison baseline
let baselineMetrics = null;   // captured from first (greedy) episode
let currentEpisodeN = 0;      // episode counter per task load

// Leaflet map + layer references
let map = null;
const sensorMarkers = {};      // id → L.marker
const targetMarkers = {};      // id → L.marker
const arcLines = [];           // L.polyline[]
const agentArcLines = [];      // agent-colored arcs for multi mode

// Snapshot of all target metadata (priority etc.) keyed by id — survives deactivation
const targetMeta = {};

// ── Fixed sensor positions (lat, lon) ─────────────────────────────────────────
const SENSOR_POS = {
  S1: [28.6, 77.2],   // Delhi
  S2: [19.0, 72.8],   // Mumbai
  S3: [13.0, 80.3],   // Chennai
  S4: [22.5, 88.3],   // Kolkata
  S5: [17.4, 78.5],   // Hyderabad
  S6: [26.9, 75.8],   // Jaipur
  S7: [23.0, 72.6],   // Ahmedabad
  S8: [12.9, 74.8],   // Mangalore
};
const CITY_NAMES = {
  S1: 'Delhi', S2: 'Mumbai', S3: 'Chennai', S4: 'Kolkata', S5: 'Hyderabad',
  S6: 'Jaipur', S7: 'Ahmedabad', S8: 'Mangalore'
};
const customSensorPos = {};    // id → [lat, lon] when dragged

// ── Threat zones for env targets ──────────────────────────────────────────────
const THREAT_ZONES = [
  [34.5, 74.0], [32.0, 77.5], [28.0, 97.5], [23.5, 91.5],
  [22.0, 69.0], [27.5, 88.5], [30.5, 71.0], [15.0, 74.0]
];

const allTargetPos = {};       // id → [lat, lon]
let nextCustomId = 1;

// ── Agent color palette ────────────────────────────────────────────────────────
const AGENT_COLORS = {
  satellite: '#2563eb',   // blue
  drone:     '#16a34a',   // green
  radar:     '#d97706',   // amber
  command:   '#9333ea',   // purple
};

// ── Helpers ───────────────────────────────────────────────────────────────────
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
function sensorIcon(type, available, agentColor) {
  const colors = { satellite: '#395144', drone: '#4E6C50', radar: '#AA8B56' };
  const c = agentColor || colors[type] || '#395144';
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

function threatIcon(priority, hasConflict) {
  const colors = { 3: '#8B2E2E', 2: '#A0522D', 1: '#4E6C50' };
  const sizes  = { 3: 26, 2: 22, 1: 18 };
  const c = colors[priority] || '#4E6C50';
  const s = sizes[priority] || 18;
  const pulse = priority === 3
    ? `<div style="position:absolute;inset:-6px;border-radius:50%;border:2px solid ${c};opacity:0.5;animation:ping 1.2s infinite"></div>` : '';
  const conflictRing = hasConflict
    ? `<div style="position:absolute;inset:-10px;border-radius:50%;border:2px dashed #e11d48;opacity:0.9;animation:ping 0.8s infinite"></div>` : '';
  return L.divIcon({
    className: '',
    html: `<div style="position:relative;width:${s}px;height:${s}px">
      ${conflictRing}${pulse}
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
        renderSensors(sensors);
        addLog(`${s.type} sensor moved to ${placeName}`, 'log-neu');
      });
      sensorMarkers[s.id] = m;
    }
  });

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
function renderEnvTargets(data, conflictTargetIds) {
  const conflictSet = new Set(conflictTargetIds || []);

  // Build set of currently active target IDs from server response
  const activeFromServer = new Set(
    data.filter(t => t.active !== false).map(t => t.id)
  );

  // Update metadata for all incoming targets
  data.forEach(t => {
    targetMeta[t.id] = { priority: t.priority };
    if (!allTargetPos[t.id]) getTargetPos(t.id);
  });

  // Remove markers for targets no longer active
  Object.keys(targetMarkers).forEach(id => {
    if (!activeFromServer.has(id)) {
      map.removeLayer(targetMarkers[id]);
      delete targetMarkers[id];
    }
  });

  // Add or update markers for active targets
  activeFromServer.forEach(id => {
    const t = data.find(x => x.id === id);
    if (!t) return;
    const pos = getTargetPos(t.id);
    const hasConflict = conflictSet.has(t.id);
    if (targetMarkers[t.id]) {
      targetMarkers[t.id].setIcon(threatIcon(t.priority, hasConflict));
    } else {
      const lvl = t.priority === 3 ? 'HIGH' : t.priority === 2 ? 'MED' : 'LOW';
      const m = L.marker(pos, { icon: threatIcon(t.priority, hasConflict), title: t.id }).addTo(map);
      m.bindTooltip(`<b>${t.id}</b><br>Priority: ${lvl}${hasConflict ? '<br><span style="color:#e11d48">⚡ CONFLICT</span>' : ''}`, { direction: 'top' });
      targetMarkers[t.id] = m;
    }
  });

  // Sync envTargets to only active ones
  envTargets = data.filter(t => activeFromServer.has(t.id));
  activeThreats = [...envTargets];
  refreshThreatPanel();
}

// ── Arc helpers ───────────────────────────────────────────────────────────────
function spawnArc(sensorId, targetId, reward) {
  const sp = getSensorPos(sensorId);
  const tp = getTargetPos(targetId);
  if (!sp || !tp) return; // FIX 7: Validation

  const color = reward > 0 ? '#4E6C50' : '#8B2E2E';
  const line = L.polyline([sp, tp], { 
      color, weight: 3, 
      dashArray: null, // FIX 9: Solid for final assignment
      opacity: 0.9 
  }).addTo(map);
  
  arcLines.push(line);
  
  // FIX 8: Fade effect
  setTimeout(() => {
    line.setStyle({ opacity: 0.3 });
    setTimeout(() => { map.removeLayer(line); }, 1500);
  }, 1000);
}

function spawnAgentArc(sensorId, targetId, agentId) {
  const sp = getSensorPos(sensorId);
  const tp = getTargetPos(targetId);
  if (!sp || !tp) return; // FIX 7: Validation

  const color = AGENT_COLORS[agentId] || '#888';
  const line = L.polyline([sp, tp], { 
      color, weight: 2.5, 
      dashArray: '5,5', // FIX 9: Dashed for proposal
      opacity: 0.9 
  }).addTo(map);

  agentArcLines.push(line);
  
  // FIX 8: Fade effect
  setTimeout(() => {
    line.setStyle({ opacity: 0.3 });
    setTimeout(() => { try { map.removeLayer(line); } catch(_){} }, 1500);
  }, 1000);
}

// ── Map click to place custom threats ─────────────────────────────────────────
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

// ── Mode toggle ───────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.getElementById('btnSingle').classList.toggle('active', mode === 'single');
  document.getElementById('btnMulti').classList.toggle('active', mode === 'multi');

  document.getElementById('singleControls').style.display  = mode === 'single' ? '' : 'none';
  document.getElementById('multiControls').style.display   = mode === 'multi'  ? '' : 'none';
  document.getElementById('multiMetrics').style.display    = mode === 'multi'  ? '' : 'none';
  document.getElementById('conflictPanel').style.display   = mode === 'multi'  ? '' : 'none';

  document.querySelectorAll('.multi-legend').forEach(el => {
    el.style.display = mode === 'multi' ? 'flex' : 'none';
  });

  addLog(`Switched to ${mode === 'multi' ? 'MULTI-AGENT' : 'SINGLE-AGENT'} mode`, 'log-neu');

  if (mode === 'multi' && obsLoaded) {
    const taskSteps = { 'Easy': 20, 'Medium': 40, 'Hard': 60 };
    const taskSeeds = { 'Easy': 42, 'Medium': 7, 'Hard': 13 };
    const taskThreats = { 'Easy': 'MONITORING', 'Medium': 'ELEVATED', 'Hard': 'CRITICAL' };
    loadTask(taskSteps[currentTask] || 20, taskSeeds[currentTask] || 42, currentTask, taskThreats[currentTask] || 'MONITORING');
  }
}

// ── Label helpers ─────────────────────────────────────────────────────────────
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

// ── Conflict log ──────────────────────────────────────────────────────────────
function addConflictEntry(conflicts, step) {
  const box = document.getElementById('conflictLog');
  if (!conflicts || conflicts.length === 0) {
    const d = document.createElement('div');
    d.className = 'conflict-entry conflict-ok';
    d.innerHTML = `<span class="conflict-step">Step ${step}</span><span class="conflict-none">✓ No conflicts</span>`;
    box.prepend(d);
    return;
  }
  conflicts.forEach(c => {
    const d = document.createElement('div');
    d.className = 'conflict-entry conflict-bad';
    const agents = c.agents && c.agents.length ? c.agents.join(', ') : '—';
    const target = c.target_id ? ` → ${c.target_id}` : '';
    const sensor = c.sensor_id ? ` [${c.sensor_id}]` : '';
    d.innerHTML = `<span class="conflict-step">Step ${step}</span>
      <span class="conflict-type">${c.type}</span>
      <span class="conflict-agents">${agents}${sensor}${target}</span>`;
    box.prepend(d);
  });
}

// ── General log ───────────────────────────────────────────────────────────────
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

function updateMultiStats(stepRewards, agentRewards, conflictRate, conflicts, step) {
  // Cumulative agent rewards — use server cumulative if available, else accumulate locally
  const rewardSource = (agentRewards && Object.values(agentRewards).some(v => v !== 0))
    ? agentRewards : null;
  if (rewardSource) {
    Object.keys(rewardSource).forEach(a => { agentRewardsCumulative[a] = rewardSource[a]; });
  } else {
    Object.keys(stepRewards).forEach(a => { agentRewardsCumulative[a] = (agentRewardsCumulative[a] || 0) + (stepRewards[a] || 0); });
  }
  document.getElementById('rewardSatellite').textContent = agentRewardsCumulative.satellite.toFixed(1);
  document.getElementById('rewardDrone').textContent     = agentRewardsCumulative.drone.toFixed(1);
  document.getElementById('rewardRadar').textContent     = agentRewardsCumulative.radar.toFixed(1);
  document.getElementById('rewardCommand').textContent   = agentRewardsCumulative.command.toFixed(1);

  // Conflict rate — color coding
  const crEl = document.getElementById('conflictRate');
  const csEl = document.getElementById('coordinationScore');
  const coordScore = 1.0 - conflictRate;
  
  crEl.textContent = conflictRate.toFixed(3);
  crEl.className = 'conflict-rate-val' + (conflictRate > 0.5 ? ' cr-high' : conflictRate > 0.2 ? ' cr-med' : ' cr-ok');
  
  if (csEl) {
    csEl.textContent = coordScore.toFixed(3);
  }
  
  // FIX 13: Console log
  console.log(`Conflict Rate: ${conflictRate}, Coordination Score: ${coordScore}`);

  // Step counter (share the same step display)
  mxStepCount = step;
  document.getElementById('statStep').textContent = step;

  // Conflict log entry
  addConflictEntry(conflicts, step);
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

function showMultiSummary(info, conflictRateOverride, agentRewardsOverride) {
  const ar = agentRewardsOverride || agentRewardsCumulative;
  const totalCumReward = Object.values(ar).reduce((a, b) => a + b, 0);
  const conflictRateFinal = conflictRateOverride ?? info.conflict_rate ?? 0;
  const coordScore = parseFloat((1.0 - conflictRateFinal).toFixed(3));

  currentEpisodeN++;
  if (currentEpisodeN === 1) {
    baselineMetrics = { conflictRate: conflictRateFinal, reward: totalCumReward, coord: coordScore };
  }

  // Use hardcoded greedy baseline for comparison (from README benchmarks)
  const GREEDY = { Easy: { conflictRate: 0.55, reward: -7.0, coord: 0.45 }, Medium: { conflictRate: 0.65, reward: -15.0, coord: 0.35 }, Hard: { conflictRate: 0.75, reward: -30.0, coord: 0.25 } };
  const bl = (currentEpisodeN > 1 && baselineMetrics) ? baselineMetrics : (GREEDY[currentTask] || GREEDY.Easy);

  const panel = document.getElementById('trainingComparison');
  if (panel) {
    panel.style.display = 'block';
    document.getElementById('cmpConflictBefore').textContent = bl.conflictRate.toFixed(3);
    document.getElementById('cmpConflictAfter').textContent  = conflictRateFinal.toFixed(3);
    document.getElementById('cmpRewardBefore').textContent   = bl.reward.toFixed(1);
    document.getElementById('cmpRewardAfter').textContent    = totalCumReward.toFixed(1);
    document.getElementById('cmpCoordBefore').textContent    = bl.coord.toFixed(3);
    document.getElementById('cmpCoordAfter').textContent     = coordScore.toFixed(3);
    document.getElementById('cmpConflictAfter').style.color  = conflictRateFinal < bl.conflictRate ? '#16a34a' : '#e11d48';
    document.getElementById('cmpRewardAfter').style.color    = totalCumReward > bl.reward ? '#16a34a' : '#e11d48';
    document.getElementById('cmpCoordAfter').style.color     = coordScore > bl.coord ? '#16a34a' : '#e11d48';
  }

  const assignmentLines = (info.assignments || []).map((a, i) =>
    `Step ${i + 1}: [${(a.agent || '?').toUpperCase()}] ${sensorLabel(a.sensor)} → ${targetLabel(a.target)}`
  ).join('\n');

  const verdict = conflictRateFinal < 0.1 ? 'Excellent coordination — minimal conflicts.'
    : conflictRateFinal < 0.3 ? 'Good coordination — moderate conflicts resolved.'
    : 'High conflict rate — agents need better negotiation.';

  const agentLines = Object.entries(ar).map(([a, r]) => `  ${a.padEnd(12)}: ${r.toFixed(2)}`).join('\n');

  const comparisonLines = `
TRAINING IMPACT (Greedy Baseline → Current):
  Conflict Rate : ${bl.conflictRate.toFixed(3)} → ${conflictRateFinal.toFixed(3)}  ${conflictRateFinal < bl.conflictRate ? '▼ IMPROVED' : '▲ WORSE'}
  Total Reward  : ${bl.reward.toFixed(1)} → ${totalCumReward.toFixed(1)}  ${totalCumReward > bl.reward ? '▲ IMPROVED' : '▼ WORSE'}
  Coord Score   : ${bl.coord.toFixed(3)} → ${coordScore.toFixed(3)}  ${coordScore > bl.coord ? '▲ IMPROVED' : '▼ WORSE'}`;

  const summary = `EPISODE SUMMARY — MULTI-AGENT MODE
${'─'.repeat(50)}
Mission: ${currentTask.toUpperCase()} | Agents: satellite, drone, radar, command
Steps completed: ${mxStepCount}

PER-AGENT CUMULATIVE REWARDS:
${agentLines}

TOTAL REWARD: ${totalCumReward.toFixed(2)}
CONFLICT RATE: ${conflictRateFinal.toFixed(4)} (${(conflictRateFinal * 100).toFixed(1)}% of steps had conflicts)
${comparisonLines}

ASSIGNMENTS:
${assignmentLines}

VERDICT: ${verdict}`;

  document.getElementById('summaryText').textContent = summary;
  document.getElementById('summaryModal').style.display = 'flex';
}

// ── Single-agent actions ───────────────────────────────────────────────────────
async function loadTask(steps, seed, name, threatLevel) {
  totalReward = 0; stepCount = 0; maxSteps = steps; done = false;
  mxDone = false; mxStepCount = 0;
  currentTask = name; currentSeed = seed;
  episodeLog = [];
  customTargets = []; nextCustomId = 1;
  envTargets = [];
  agentRewardsCumulative = { satellite: 0, drone: 0, radar: 0, command: 0 };
  conflictLog = [];
  currentEpisodeN = 0;
  baselineMetrics = null;
  const _tc = document.getElementById('trainingComparison');
  if (_tc) _tc.style.display = 'none';

  Object.keys(allTargetPos).forEach(k => delete allTargetPos[k]);
  Object.keys(targetMeta).forEach(k => delete targetMeta[k]);

  Object.values(sensorMarkers).forEach(m => map.removeLayer(m));
  Object.keys(sensorMarkers).forEach(k => delete sensorMarkers[k]);
  Object.values(targetMarkers).forEach(m => map.removeLayer(m));
  Object.keys(targetMarkers).forEach(k => delete targetMarkers[k]);
  arcLines.forEach(l => map.removeLayer(l)); arcLines.length = 0;
  agentArcLines.forEach(l => { try { map.removeLayer(l); } catch(_){} }); agentArcLines.length = 0;

  document.getElementById('logBox').innerHTML = '';
  document.getElementById('conflictLog').innerHTML = '';
  ['statStep', 'statReward', 'statMissed', 'statAssigned'].forEach(id =>
    document.getElementById(id).textContent = '0');
  document.getElementById('scoreBar').style.width = '0%';
  document.getElementById('scorePct').textContent = '0.000';
  document.getElementById('gradeVal').textContent = '--';
  document.getElementById('gradeSub').textContent = 'Run grader to evaluate';
  ['rewardSatellite','rewardDrone','rewardRadar','rewardCommand'].forEach(id =>
    document.getElementById(id).textContent = '0.0');
  document.getElementById('conflictRate').textContent = '0.000';
  setActiveTask(name);

  document.querySelectorAll('.btn-assign,.btn-auto,.btn-run,.btn-grade').forEach(b => b.disabled = true);

  let obs;
  if (currentMode === 'multi') {
    const densityMap = { 20: 1.5, 40: 2.5, 60: 4.0 };
    const failureMap = { 20: 0.0, 40: 0.05, 60: 0.15 };
    obs = await api('POST', '/reset_multi', {
      max_steps: steps,
      seed: seed,
      density_factor: densityMap[steps] || 1.5,
      failure_prob: failureMap[steps] || 0.0,
      conflict_injection: steps >= 40,
    });
    currentSeed = obs.seed || seed;
    // Use command agent obs — it sees ALL sensors and ALL targets
    const cmdObs = obs.observations ? obs.observations['command'] : null;
    if (cmdObs) {
      renderSensors(cmdObs.sensors);
      renderEnvTargets(cmdObs.targets);
    }
    addLog(`[MULTI] Mission started: ${name} — 4 agents active, ${steps} steps`, 'log-neu');
  } else {
    obs = await api('POST', '/reset', { max_steps: steps });
    currentSeed = obs.seed || seed;
    renderSensors(obs.sensors);
    renderEnvTargets(obs.targets);
    addLog(`Mission started: ${name} task with ${obs.sensors.length} sensors and ${obs.targets.length} initial threats.`, 'log-neu');
  }

  document.querySelectorAll('.btn-assign,.btn-auto,.btn-run,.btn-grade').forEach(b => b.disabled = false);

  document.getElementById('taskLabel').textContent    = name.toUpperCase() + ' — ' + threatLevel;
  document.getElementById('statusPill').innerHTML     = '&#9679; ' + threatLevel;
  document.getElementById('statusPill').className     = 'status-pill active';
  document.getElementById('mapOverlay').style.display = 'none';
  document.getElementById('toolHint').textContent     = 'Drag sensors to reposition | Use toolbar to place custom threats';

  obsLoaded = true;
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
  if (actions.length === 0) { if (r.done) showSummary(r.info); return; }
  actions.forEach(a => spawnArc(a.sensor_id, a.target_id, r.reward));
  renderSensors(r.observation.sensors);
  renderEnvTargets(r.observation.targets);
  updateStats(r.reward, r.info);
  const c = r.reward > 0 ? 'log-pos' : r.reward < 0 ? 'log-neg' : 'log-neu';
  const agentTag = r.agent === 'llm' ? ' <span style="color:#AA8B56">[LLM]</span>' : ' <span style="opacity:0.5">[greedy]</span>';
  const desc = actions.map(a => `The ${sensorLabel(a.sensor_id)} → ${targetLabel(a.target_id)}`).join(' | ');
  addLog(`[Auto]${agentTag} ${desc}. Reward: ${r.reward > 0 ? '+' : ''}${r.reward}${r.done ? ' — Mission complete.' : ''}`, c);
  if (r.done) showSummary(r.info);
}

async function runAll() {
  if (!obsLoaded) return addLog('Load a mission first', 'log-neg');
  if (done) { await loadTask(maxSteps, currentSeed, currentTask, 'RUNNING'); }
  while (!done) {
    await autoStep();
    await new Promise(r => setTimeout(r, 1200)); // FIX 10: Step delay
    if (done) break;
  }
}

// ── Multi-agent actions ───────────────────────────────────────────────────────
async function autoMultiStep() {
  if (!obsLoaded) return;
  if (mxDone) return addLog('Episode complete — reload mission', 'log-neg');

  const r = await api('POST', '/auto_multi');
  if (r.error) return addLog('ERROR: ' + r.error, 'log-neg');
  mxDone = r.done;

  // Draw agent-colored arcs for proposals
  const proposals = r.proposals || [];
  proposals.forEach(p => spawnAgentArc(p.sensor_id, p.target_id, p.agent_id));

  // Use first agent's observation for display
  const agentObs = r.observations;
  // Always use command obs — sees all sensors + all targets
  const cmdObs = agentObs ? agentObs['command'] : null;

  // Collect conflict target IDs for overlay
  const conflictTargetIds = (r.conflicts || [])
    .filter(c => c.target_id).map(c => c.target_id);

  if (cmdObs) {
    renderSensors(cmdObs.sensors);
    renderEnvTargets(cmdObs.targets, conflictTargetIds);
  }

  // Update step / stats
  if (r.info) {
    updateMultiStats(r.step_rewards || {}, r.agent_rewards || {}, r.conflict_rate || 0, r.conflicts || [], r.info.step_count);
    const totalStepRwd = Object.values(r.step_rewards || {}).reduce((a,b)=>a+b,0);
    totalReward += totalStepRwd;
    document.getElementById('statReward').textContent = totalReward.toFixed(1);
    document.getElementById('statMissed').textContent = (r.info.missed_targets || []).length;
    document.getElementById('statAssigned').textContent = (r.info.assignments || []).length;
  }

  const source = r.agent === 'llm'
    ? '<span style="color:#AA8B56">[LLM]</span>'
    : '<span style="opacity:0.5">[greedy]</span>';
  const propDesc = proposals.map(p => {
    const agColor = AGENT_COLORS[p.agent_id] || '#888';
    return `<span style="color:${agColor}">[${p.agent_id}]</span> ${p.sensor_id}→${p.target_id}`;
  }).join(' ');

  const conflictStr = r.conflicts && r.conflicts.length
    ? ` <span class="conflict-badge">⚡ ${r.conflicts.length} conflict(s)</span>`
    : '';

  addLog(`[Multi] ${source} ${propDesc || 'no proposals'}${conflictStr}${r.done ? ' — Episode done.' : ''}`, 'log-neu');

  if (r.done) showMultiSummary(r.info, r.conflict_rate, r.agent_rewards);
}

async function runAllMulti() {
  if (!obsLoaded) return addLog('Load a mission first', 'log-neg');
  if (mxDone) { await loadTask(maxSteps, currentSeed, currentTask, 'RUNNING'); }
  while (!mxDone) {
    await autoMultiStep();
    await new Promise(r => setTimeout(r, 1200)); // FIX 10: Step delay
    if (mxDone) break;
  }
}

// ── Grader ────────────────────────────────────────────────────────────────────
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
    const m = L.marker([lat, lon], { icon: threatIcon(priority, false) }).addTo(map);
    m.bindTooltip(`<b>${id}</b><br>Priority: ${lvl} (custom)`, { direction: 'top' });
    targetMarkers[id] = m;

    refreshThreatPanel();
    addLog(`Custom ${lvl} threat placed at ${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E`, 'log-neu');
  });

  setTool('move');
  setMode('single');
  loadTask(20, 42, 'Easy', 'MONITORING');
});
