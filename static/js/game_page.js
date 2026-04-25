(function () {
  const missions = [
    { id: 1, name: "Sentinel Patrol", difficulty: "easy", story: "Initial surveillance sweep across low-threat sectors.", objective: "Secure low-threat zone with zero missed critical targets.", maxSteps: 20, seed: 42, unlocked: true, sector: "X-RAY 11" },
    { id: 2, name: "Border Surge", difficulty: "medium", story: "Multiple threats detected across sectors. Drone reliability compromised.", objective: "Cover all P3 threats while containing conflict escalation.", maxSteps: 40, seed: 7, unlocked: false, sector: "X-RAY 44" },
    { id: 3, name: "Storm Protocol", difficulty: "hard", story: "Escalating high-threat activity under degraded weather and sensor reliability.", objective: "Maintain coverage under disruption and maximize coordination score.", maxSteps: 60, seed: 13, unlocked: false, sector: "X-RAY 78" }
  ];

  const gameState = {
    running: false,
    mode: "manual",
    step: 0,
    score: 0,
    activeMission: null,
    done: false,
    autoLoop: null,
    timerLoop: null,
    startedAtMs: 0
  };

  function byId(id) { return document.getElementById(id); }
  function firstEl(ids) {
    for (const id of ids) {
      const el = byId(id);
      if (el) return el;
    }
    return null;
  }

  function debugElements() {
    console.log("DEBUG ELEMENTS:");
    ["startBtn", "manualBtn", "autoBtn", "endBtn", "sensorSelect", "targetSelect"].forEach((id) => {
      console.log(id, byId(id));
    });
    console.log("DEBUG FALLBACK ELEMENTS:");
    console.log("gmStartBtn", byId("gmStartBtn"));
    console.log("gmManualBtn", byId("gmManualBtn"));
    console.log("gmAutoBtn", byId("gmAutoBtn"));
    console.log("gmEndBtn", byId("gmEndBtn"));
    console.log("sensorSel", byId("sensorSel"));
    console.log("targetSel", byId("targetSel"));
  }

  function loadProgress() {
    try {
      const raw = localStorage.getItem("arya_game_progress_v1");
      if (!raw) return;
      const parsed = JSON.parse(raw);
      missions.forEach((m) => {
        if (typeof parsed[String(m.id)] === "boolean") m.unlocked = parsed[String(m.id)];
      });
      missions[0].unlocked = true;
    } catch (_) {}
  }

  function saveProgress() {
    const payload = {};
    missions.forEach((m) => { payload[String(m.id)] = m.unlocked; });
    localStorage.setItem("arya_game_progress_v1", JSON.stringify(payload));
  }

  function fixMap() {
    if (window.map) {
      setTimeout(() => {
        window.map.invalidateSize();
        console.log("Map resized");
      }, 500);
    }
  }

  function appendCommentary(text, tone) {
    const feed = byId("gmCommentaryFeed");
    if (!feed) return;
    const row = document.createElement("div");
    const cls = tone === "warn" ? "text-secondary font-bold" : tone === "good" ? "text-primary-container font-bold" : "text-tertiary";
    row.className = "flex gap-2";
    row.innerHTML = `<span class="opacity-50">${new Date().toLocaleTimeString()}</span><span class="${cls}">${text}</span>`;
    feed.appendChild(row);
    feed.scrollTop = feed.scrollHeight;
  }

  async function safeApi(path, data, method) {
    const httpMethod = method || "POST";
    try {
      const res = await api(httpMethod, path, data);
      console.log("API OK:", path, res);
      return res;
    } catch (e) {
      console.error("API FAILED:", path, e);
      alert("API error");
      return null;
    }
  }

  function renderMissionButtons() {
    document.querySelectorAll(".gm-mission-btn").forEach((btn) => {
      const id = Number(btn.dataset.mission);
      const m = missions.find((x) => x.id === id);
      if (!m) return;
      btn.disabled = !m.unlocked;
      btn.classList.toggle("opacity-40", !m.unlocked);
      btn.classList.toggle("cursor-not-allowed", !m.unlocked);
      btn.classList.toggle("bg-primary-container", gameState.activeMission && gameState.activeMission.id === m.id);
      btn.classList.toggle("text-white", gameState.activeMission && gameState.activeMission.id === m.id);
      btn.textContent = `M${m.id}${m.unlocked ? "" : " LOCK"}`;
    });
  }

  function setMission(mission) {
    if (!mission || !mission.unlocked) return;
    gameState.activeMission = mission;
    byId("gmMissionTitle").textContent = mission.name.toUpperCase();
    byId("gmMissionObjective").textContent = mission.objective;
    byId("gmMissionStory").innerHTML = `<p>${mission.story}</p>`;
    byId("gmDifficulty").textContent = mission.difficulty.toUpperCase();
    byId("gmSector").textContent = mission.sector;
    byId("gmMissionCode").textContent = `M-0${mission.id}`;
    renderMissionButtons();
  }

  function updateStats(res) {
    const info = res?.info || {};
    const step = Number(info.step_count || gameState.step || 0);
    const assigned = Number((info.assignments || []).length || 0);
    const conflicts = Number((res?.conflicts || info.conflicts || []).length || 0);
    const threats = document.querySelectorAll("#threatList .threat-item").length;
    gameState.step = step;

    byId("gmStep").textContent = String(step).padStart(4, "0");
    byId("gmAssigned").textContent = String(assigned).padStart(2, "0");
    byId("gmConflicts").textContent = String(conflicts).padStart(2, "0");
    byId("gmThreats").textContent = String(threats).padStart(2, "0");
    byId("gmStepBottom").textContent = String(step).padStart(4, "0");
    byId("gmThreatBottom").textContent = String(threats).padStart(2, "0");
    byId("gmAssignedBottom").textContent = `${assigned}/${(res?.observations?.command?.sensors || []).length || 0}`;
    byId("gmConflictBottom").textContent = conflicts > 0 ? "DETECTED" : "NONE";
    byId("gmEvent").textContent = conflicts > 0 ? "CONFLICT" : "READY";
  }

  function updateAgents(res) {
    const cmdReward = Number((res?.agent_rewards || {}).command || 0);
    byId("gmStatusChakra").textContent = cmdReward !== 0 ? "ACTIVE" : "STANDBY";
    byId("gmActionChakra").textContent = cmdReward !== 0 ? "OVERRIDE INITIATED" : "OVERRIDE";

    const droneSensors = (res?.observations?.command?.sensors || []).filter((s) => s.type === "drone");
    const droneOffline = droneSensors.some((s) => s.available === false);
    byId("gmStatusDhuruv").textContent = droneOffline ? "OFFLINE" : "ACTIVE";
    byId("gmActionDhuruv").textContent = droneOffline ? "RECOVERING" : "TACTICAL HOLD";
  }

  function updateLogs(res, modeText) {
    if (modeText) appendCommentary(modeText, "good");
    const conflicts = Number((res?.conflicts || []).length || 0);
    if (conflicts > 0) appendCommentary("Conflict detected - resolving...", "warn");
    else appendCommentary("ARYA locked high-priority target.", "good");
  }

  function syncMapDataIntoDashboardRenderer(res) {
    const cmdObs = res?.observations?.command || res?.command_observation || null;
    if (!cmdObs) return;
    if (typeof renderSensors === "function") renderSensors(cmdObs.sensors || []);
    if (typeof renderEnvTargets === "function") {
      const conflictTargets = (res?.conflicts || []).filter((c) => c.target_id).map((c) => c.target_id);
      renderEnvTargets(cmdObs.targets || [], conflictTargets);
    }

    const sensorSelect = firstEl(["sensorSelect", "sensorSel"]);
    const targetSelect = firstEl(["targetSelect", "targetSel"]);
    if (sensorSelect) {
      sensorSelect.innerHTML = (cmdObs.sensors || []).filter((s) => s.available).map((s) => `<option value="${s.id}">${s.id} - ${s.type}</option>`).join("");
    }
    if (targetSelect) {
      targetSelect.innerHTML = (cmdObs.targets || []).filter((t) => t.active).sort((a, b) => b.priority - a.priority).map((t) => `<option value="${t.id}">${t.id} (P${t.priority})</option>`).join("");
    }
  }

  function updateMap(res, path, selection) {
    console.log("MAP UPDATE:", res);
    console.log("MAP:", window.map);
    if (!window.map) return;
    syncMapDataIntoDashboardRenderer(res);

    if (window.GameVisuals && typeof window.GameVisuals.updateVisualsFromState === "function") {
      window.GameVisuals.updateVisualsFromState(res, path || "", selection || null);
    }
  }

  function updateUI(res, path, selection, modeText) {
    console.log("UPDATE UI");
    if (!res) return;
    updateMap(res, path, selection);
    updateStats(res);
    updateAgents(res);
    updateLogs(res, modeText);
    fixMap();
  }

  function tickTimer() {
    if (!gameState.running) return;
    const diff = Math.max(0, Date.now() - gameState.startedAtMs);
    const s = Math.floor(diff / 1000);
    const hh = String(Math.floor(s / 3600)).padStart(2, "0");
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    byId("gmTimer").textContent = `${hh}:${mm}:${ss}`;
  }

  async function startMission() {
    console.log("START CLICKED");
    const mission = gameState.activeMission || missions[0];
    gameState.running = true;
    gameState.done = false;
    gameState.step = 0;
    gameState.mode = "manual";
    gameState.startedAtMs = Date.now();
    byId("gmHeaderStatus").textContent = "ACTIVE";
    byId("gmIntroOverlay").style.display = "none";
    if (gameState.autoLoop) { clearInterval(gameState.autoLoop); gameState.autoLoop = null; }
    if (gameState.timerLoop) { clearInterval(gameState.timerLoop); }
    gameState.timerLoop = setInterval(tickTimer, 1000);

    setMode("multi");
    const res = await safeApi("/reset_multi", {
      max_steps: mission.maxSteps,
      seed: mission.seed,
      density_factor: mission.maxSteps === 20 ? 1.5 : mission.maxSteps === 40 ? 2.5 : 4.0,
      failure_prob: mission.maxSteps === 20 ? 0.0 : mission.maxSteps === 40 ? 0.05 : 0.15,
      conflict_injection: mission.maxSteps >= 40
    });
    if (!res) return;
    updateUI(res, "/reset_multi", null, `MISSION ${mission.id}: ${mission.name} started.`);
    alert(`MISSION ${mission.id}: ${mission.name}\n${mission.story}`);
  }

  function setManual() {
    gameState.mode = "manual";
    if (gameState.autoLoop) { clearInterval(gameState.autoLoop); gameState.autoLoop = null; }
    appendCommentary("Manual mode engaged", "good");
  }

  async function setAuto() {
    console.log("AUTO MODE");
    if (!gameState.running || gameState.done) return;
    gameState.mode = "auto";
    if (gameState.autoLoop) clearInterval(gameState.autoLoop);

    gameState.autoLoop = setInterval(async () => {
      if (!gameState.running || gameState.done || gameState.mode !== "auto") return;
      const res = await safeApi("/auto_multi", {});
      if (!res) return;
      gameState.done = !!res.done;
      updateUI(res, "/auto_multi", null, "Auto mode step executed.");
      if (gameState.done) {
        clearInterval(gameState.autoLoop);
        gameState.autoLoop = null;
        await endMission();
      }
    }, 1400);
  }

  async function assignTarget() {
    const sensorEl = firstEl(["sensorSelect", "sensorSel"]);
    const targetEl = firstEl(["targetSelect", "targetSel"]);
    const sensor = sensorEl ? sensorEl.value : "";
    const target = targetEl ? targetEl.value : "";
    console.log("ASSIGN:", sensor, target);
    if (!sensor || !target) {
      alert("Select both");
      return;
    }
    const res = await safeApi("/step_multi", {
      proposals: [{ agent_id: "command", sensor_id: sensor, target_id: target }]
    });
    if (!res) return;
    gameState.done = !!res.done;
    updateUI(res, "/step_multi", { sensorId: sensor, targetId: target }, "Manual assignment executed.");
    if (gameState.done) await endMission();
  }

  async function endMission() {
    if (!gameState.running) return;
    if (gameState.autoLoop) { clearInterval(gameState.autoLoop); gameState.autoLoop = null; }
    if (gameState.timerLoop) { clearInterval(gameState.timerLoop); gameState.timerLoop = null; }
    gameState.running = false;
    gameState.mode = "manual";
    byId("gmHeaderStatus").textContent = "STANDBY";

    const mission = gameState.activeMission || missions[0];
    const res = await safeApi("/grade", { max_steps: mission.maxSteps, seed: mission.seed });
    if (!res) return;
    const score = Number(res.score || 0);
    gameState.score = score;
    const result = score >= 0.6 ? "SUCCESS" : "FAIL";
    appendCommentary(`MISSION COMPLETE: ${result}. Score ${score.toFixed(3)}.`, result === "SUCCESS" ? "good" : "warn");

    if (result === "SUCCESS") {
      const next = missions.find((m) => m.id === mission.id + 1);
      if (next && !next.unlocked) {
        next.unlocked = true;
        saveProgress();
        renderMissionButtons();
        appendCommentary(`MISSION ${next.id} unlocked: ${next.name}.`, "good");
      }
    }

    alert(`MISSION COMPLETE\nResult: ${result}\nScore: ${score.toFixed(3)}\nSteps: ${gameState.step}`);
    fixMap();
  }

  function bindMissionButtons() {
    document.querySelectorAll(".gm-mission-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const m = missions.find((x) => x.id === Number(btn.dataset.mission));
        if (m && m.unlocked) {
          setMission(m);
          appendCommentary(`Mission selected: ${m.name}`, "good");
        }
      });
    });
  }

  function setMission(mission) {
    gameState.activeMission = mission;
    byId("gmMissionTitle").textContent = mission.name.toUpperCase();
    byId("gmMissionObjective").textContent = mission.objective;
    byId("gmMissionStory").innerHTML = `<p>${mission.story}</p>`;
    byId("gmDifficulty").textContent = mission.difficulty.toUpperCase();
    byId("gmSector").textContent = mission.sector;
    byId("gmMissionCode").textContent = `M-0${mission.id}`;
    renderMissionButtons();
  }

  function bindButtons() {
    console.log("Binding buttons...");
    firstEl(["startBtn", "gmStartBtn"])?.addEventListener("click", startMission);
    firstEl(["manualBtn", "gmManualBtn"])?.addEventListener("click", setManual);
    firstEl(["autoBtn", "gmAutoBtn"])?.addEventListener("click", setAuto);
    firstEl(["endBtn", "gmEndBtn"])?.addEventListener("click", endMission);
    firstEl(["assignBtn", "gmAssignBtn"])?.addEventListener("click", assignTarget);
    byId("gmExitBtn")?.addEventListener("click", () => { window.location.href = "/"; });
  }

  window.addEventListener("load", () => {
    debugElements();
    bindButtons();
    loadProgress();
    bindMissionButtons();
    setMission(missions.find((m) => m.unlocked) || missions[0]);
    if (window.GameVisuals && typeof window.GameVisuals.init === "function") {
      window.GameVisuals.init();
    }
    console.log("MAP:", window.map);
    fixMap();
    appendCommentary("Mission Commander online. Select mission and click INITIATE PROTOCOL.");
  });
})();
