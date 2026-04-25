(function () {
  const overlay = document.getElementById("gameModeOverlay");
  const openBtn = document.getElementById("btnGameMode");
  if (!overlay || !openBtn) return;

  const missionBriefs = {
    easy: {
      name: "SENTINEL PATROL",
      story: "Initial surveillance sweep across low-threat sectors.",
      objective: "Stabilize coverage and keep all high-priority threats tracked."
    },
    medium: {
      name: "BORDER SURGE",
      story: "Multiple threats detected across sectors. Drone reliability compromised.",
      objective: "Maintain P3 coverage while minimizing coordination conflicts."
    },
    hard: {
      name: "STORM PROTOCOL",
      story: "Escalating high-threat activity under degraded field conditions.",
      objective: "Cover all P3 threats, preserve sensor efficiency, and limit overrides."
    }
  };

  const commentaryTemplates = {
    aryaLock: "ARYA locked high-priority target.",
    dhuruvMove: "DHURUV repositioning for tactical coverage.",
    rajanTrack: "RAJAN tracking dynamic threat corridor.",
    chakraOverride: "CHAKRA override activated.",
    conflict: "CONFLICT DETECTED - competing assignment proposals.",
    assign: "Agent assigned sensor to threat.",
    optimal: "OPTIMAL COVERAGE - all P3 threats currently covered.",
    storm: "STORM DETECTED - Drone offline."
  };

  const state = {
    visible: false,
    busy: false,
    currentMission: "easy",
    links: [],
    agentCards: {
      satellite: { status: "STANDBY", action: "Awaiting mission", confidence: 0.5 },
      drone: { status: "STANDBY", action: "Awaiting mission", confidence: 0.5 },
      radar: { status: "STANDBY", action: "Awaiting mission", confidence: 0.5 },
      command: { status: "STANDBY", action: "Monitoring", confidence: 0.5 }
    },
    lastDroneDown: false,
    drawInterval: null,
    lastSelection: { sensorId: null, targetId: null }
  };

  function renderOverlay() {
    overlay.innerHTML = `
      <div class="game-ui">
      <div class="game-shell" role="dialog" aria-modal="true" aria-label="Game Mode">
        <div class="game-panel game-header">
          <div class="game-title">OPERATION DHURANDHAR — GAME MODE</div>
          <div class="game-controls">
            <button class="game-btn" id="gmMissionBtn">MISSION MODE</button>
            <button class="game-btn" id="gmManualBtn">MANUAL STEP</button>
            <button class="game-btn" id="gmAutoBtn">AUTO MODE</button>
            <button class="game-btn" id="gmGradeBtn">MISSION DEBRIEF</button>
            <button class="game-btn exit" id="gmExitBtn">EXIT GAME</button>
          </div>
        </div>
        <section class="game-panel briefing-panel">
          <div class="briefing-title">MISSION BRIEFING</div>
          <div class="briefing-mission" id="gmMissionName">MISSION: --</div>
          <div class="briefing-story" id="gmMissionStory">Story will appear when Mission Mode starts.</div>
          <div class="briefing-objective" id="gmMissionObjective">Objective: --</div>
        </section>
        <section class="game-panel status-bar game-stats">
          <div class="status-item">
            <div class="status-lbl">STEP</div>
            <div class="status-val" id="gmStepVal">0</div>
          </div>
          <div class="status-item">
            <div class="status-lbl">ACTIVE THREATS</div>
            <div class="status-val" id="gmThreatVal">0</div>
          </div>
          <div class="status-item">
            <div class="status-lbl">ASSIGNED</div>
            <div class="status-val" id="gmAssignedVal">0</div>
          </div>
          <div class="status-item">
            <div class="status-lbl">CONFLICTS</div>
            <div class="status-val" id="gmConflictVal">0</div>
          </div>
          <div class="status-item">
            <div class="status-lbl">EVENT</div>
            <div class="event-bar" id="gmEventBar">STANDBY</div>
          </div>
        </section>
        <aside class="game-panel agent-panel game-agents">
          <h3 class="agent-title">AGENT CHARACTER CARDS</h3>
          <div class="agent-card satellite">
            <div class="agent-head"><span class="agent-name">ARYA (Satellite)</span><span class="agent-status" id="gmStatusSatellite">STANDBY</span></div>
            <div class="agent-line">Action: <span id="gmActionSatellite">Awaiting mission</span></div>
            <div class="agent-line">Confidence: <span id="gmConfidenceSatellite">50%</span></div>
            <div class="agent-line">Role: Strategic overwatch and P3 acquisition.</div>
          </div>
          <div class="agent-card drone">
            <div class="agent-head"><span class="agent-name">DHURUV (Drone)</span><span class="agent-status" id="gmStatusDrone">STANDBY</span></div>
            <div class="agent-line">Action: <span id="gmActionDrone">Awaiting mission</span></div>
            <div class="agent-line">Confidence: <span id="gmConfidenceDrone">50%</span></div>
            <div class="agent-line">Role: Tactical rapid response in dense zones.</div>
          </div>
          <div class="agent-card radar">
            <div class="agent-head"><span class="agent-name">RAJAN (Radar)</span><span class="agent-status" id="gmStatusRadar">STANDBY</span></div>
            <div class="agent-line">Action: <span id="gmActionRadar">Awaiting mission</span></div>
            <div class="agent-line">Confidence: <span id="gmConfidenceRadar">50%</span></div>
            <div class="agent-line">Role: Persistent airspace and sector monitoring.</div>
          </div>
          <div class="agent-card command">
            <div class="agent-head"><span class="agent-name">CHAKRA (Command)</span><span class="agent-status" id="gmStatusCommand">STANDBY</span></div>
            <div class="agent-line">Action: <span id="gmActionCommand">Monitoring</span></div>
            <div class="agent-line">Confidence: <span id="gmConfidenceCommand">50%</span></div>
            <div class="agent-line">Role: Conflict resolution and strategic override.</div>
          </div>
        </aside>
        <section class="game-panel map-hud game-map-area">
          <div class="map-hud-text">LIVE GAME MAP (reusing original ARYA-X battlefield)</div>
          <svg class="links-layer" id="gmLinksLayer"></svg>
        </section>
        <aside class="game-panel commentary-panel game-commentary">
          <h3 class="commentary-title">FIELD COMMENTARY</h3>
          <div class="commentary-feed" id="gmCommentaryFeed"></div>
        </aside>
        <div class="debrief-modal" id="gmDebriefModal">
          <div class="debrief-card">
            <h4>MISSION DEBRIEF</h4>
            <div class="debrief-result" id="gmDebriefResult">MISSION RESULT: --</div>
            <div class="debrief-row"><span>Score</span><span id="gmDebriefScore">--</span></div>
            <div class="debrief-row"><span>Efficiency</span><span id="gmDebriefEfficiency">--</span></div>
            <div class="debrief-row"><span>Coordination Score</span><span id="gmDebriefCoordination">--</span></div>
            <div class="debrief-row"><span>Breakdown</span><span id="gmDebriefBreakdown">--</span></div>
            <button class="game-btn debrief-close" id="gmCloseDebriefBtn">CLOSE</button>
          </div>
        </div>
      </div>
      </div>
    `;
  }

  function detectMissionFromUI() {
    if (document.getElementById("btn-hard")?.classList.contains("active")) return "hard";
    if (document.getElementById("btn-medium")?.classList.contains("active")) return "medium";
    return "easy";
  }

  function updateBriefing() {
    state.currentMission = detectMissionFromUI();
    const brief = missionBriefs[state.currentMission];
    const name = document.getElementById("gmMissionName");
    const story = document.getElementById("gmMissionStory");
    const objective = document.getElementById("gmMissionObjective");
    if (name) name.textContent = `MISSION: ${brief.name}`;
    if (story) story.textContent = brief.story;
    if (objective) objective.textContent = `Objective: ${brief.objective}`;
  }

  function setEvent(message) {
    const el = document.getElementById("gmEventBar");
    if (el) el.textContent = message;
  }

  function setRunState(text) {
    setEvent(text);
  }

  function addCommentary(text) {
    const feed = document.getElementById("gmCommentaryFeed");
    if (!feed) return;
    const row = document.createElement("div");
    row.className = "commentary-line";
    row.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    feed.prepend(row);
  }

  function updateTopStats() {
    const step = document.getElementById("statStep")?.textContent || "0";
    const assigned = document.getElementById("statAssigned")?.textContent || "0";
    const conflict = document.getElementById("conflictCount")?.textContent || "0";
    const activeThreats = document.querySelectorAll("#threatList .threat-item").length;
    const stepEl = document.getElementById("gmStepVal");
    const assignedEl = document.getElementById("gmAssignedVal");
    const conflictEl = document.getElementById("gmConflictVal");
    const threatsEl = document.getElementById("gmThreatVal");
    if (stepEl) stepEl.textContent = step;
    if (assignedEl) assignedEl.textContent = assigned;
    if (conflictEl) conflictEl.textContent = String(conflict || "0");
    if (threatsEl) threatsEl.textContent = String(activeThreats);
  }

  function updateAgentCard(agent, data) {
    state.agentCards[agent] = { ...state.agentCards[agent], ...data };
    const key = agent.charAt(0).toUpperCase() + agent.slice(1);
    const statusEl = document.getElementById(`gmStatus${key}`);
    const actionEl = document.getElementById(`gmAction${key}`);
    const confidenceEl = document.getElementById(`gmConfidence${key}`);
    if (statusEl) statusEl.textContent = state.agentCards[agent].status;
    if (actionEl) actionEl.textContent = state.agentCards[agent].action;
    if (confidenceEl) confidenceEl.textContent = `${Math.round(state.agentCards[agent].confidence * 100)}%`;
  }

  function markerElementForSensor(sensorId) {
    const allMarkers = Array.from(document.querySelectorAll(".leaflet-marker-icon[title]"));
    return allMarkers.find((el) => el.getAttribute("title").startsWith(`${sensorId} `));
  }

  function markerElementForTarget(targetId) {
    const allMarkers = Array.from(document.querySelectorAll(".leaflet-marker-icon[title]"));
    return allMarkers.find((el) => el.getAttribute("title") === targetId);
  }

  function centerOf(el) {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }

  function drawLinks() {
    const svg = document.getElementById("gmLinksLayer");
    if (!svg || !state.visible) return;
    const overlayRect = overlay.getBoundingClientRect();
    svg.setAttribute("width", String(overlayRect.width));
    svg.setAttribute("height", String(overlayRect.height));
    svg.innerHTML = "";
    const now = Date.now();
    state.links = state.links.filter((link) => now - link.t < 2200);
    state.links.forEach((link) => {
      const s = centerOf(markerElementForSensor(link.sensorId));
      const t = centerOf(markerElementForTarget(link.targetId));
      if (!s || !t) return;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(s.x - overlayRect.left));
      line.setAttribute("y1", String(s.y - overlayRect.top));
      line.setAttribute("x2", String(t.x - overlayRect.left));
      line.setAttribute("y2", String(t.y - overlayRect.top));
      line.setAttribute("stroke", link.color);
      line.setAttribute("stroke-width", "3");
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("stroke-opacity", "0.88");
      if (link.pending) line.setAttribute("stroke-dasharray", "8 6");
      svg.appendChild(line);
    });
  }

  function addLink(sensorId, targetId, color, pending) {
    state.links.push({ sensorId, targetId, color, pending: !!pending, t: Date.now() });
    drawLinks();
  }

  function handleSelectionHighlights() {
    const sensorSel = document.getElementById("sensorSel");
    const targetSel = document.getElementById("targetSel");
    if (!sensorSel || !targetSel) return;
    if (sensorSel.value && targetSel.value) {
      addLink(sensorSel.value, targetSel.value, "#AA8B56", true);
      addCommentary(`Commander selection prepared: ${sensorSel.value} -> ${targetSel.value}.`);
    }
  }

  async function missionMode() {
    if (typeof setMode === "function") setMode("single");
    updateBriefing();
    setRunState("MISSION READY");
    updateAgentCard("satellite", { status: "ACTIVE", action: "Sector scan", confidence: 0.8 });
    updateAgentCard("drone", { status: "ACTIVE", action: "Holding tactical route", confidence: 0.72 });
    updateAgentCard("radar", { status: "ACTIVE", action: "Airspace watch", confidence: 0.75 });
    updateAgentCard("command", { status: "STANDBY", action: "Awaiting conflict", confidence: 0.7 });
    addCommentary("Mission briefing loaded. Commander controls enabled.");
    updateTopStats();
  }

  async function manualStepGame() {
    if (state.busy) return;
    if (typeof manualStep !== "function") return;
    const sensorSel = document.getElementById("sensorSel");
    const targetSel = document.getElementById("targetSel");
    if (sensorSel?.value && targetSel?.value) {
      state.lastSelection = { sensorId: sensorSel.value, targetId: targetSel.value };
      addLink(sensorSel.value, targetSel.value, "#4E6C50", false);
      addCommentary(commentaryTemplates.assign);
    }
    state.busy = true;
    setRunState("MANUAL STEP");
    try {
      await manualStep();
      updateAgentCard("satellite", { status: "ACTIVE", action: "ARYA locked high-priority target", confidence: 0.85 });
      addCommentary(commentaryTemplates.aryaLock);
      updateTopStats();
    } finally {
      state.busy = false;
      setRunState("READY");
    }
  }

  async function autoModeGame() {
    if (state.busy) return;
    if (typeof setMode === "function") setMode("multi");
    if (typeof autoMultiStep !== "function") return;
    state.busy = true;
    setRunState("AUTO RUNNING");
    try {
      await autoMultiStep();
      updateTopStats();
      drawLinks();
      const conflictCount = Number(document.getElementById("conflictCount")?.textContent || 0);
      if (conflictCount > 0) {
        addCommentary(commentaryTemplates.conflict);
        setEvent("CONFLICT DETECTED");
      }
      const cmdReward = parseFloat(document.getElementById("rewardCommand")?.textContent || "0");
      if (cmdReward !== 0) {
        updateAgentCard("command", { status: "ACTIVE", action: "Override in effect", confidence: 0.84 });
        addCommentary(commentaryTemplates.chakraOverride);
      }
      const missed = Number(document.getElementById("statMissed")?.textContent || 0);
      if (missed === 0) {
        setEvent("OPTIMAL COVERAGE");
        addCommentary(commentaryTemplates.optimal);
      }
    } finally {
      state.busy = false;
      setRunState("READY");
    }
  }

  async function showDebrief() {
    setRunState("DEBRIEF");
    const modal = document.getElementById("gmDebriefModal");
    if (!modal) return;

    let scoreVal = "--";
    let efficiencyVal = "--";
    let coordinationVal = "--";

    try {
      if (typeof api === "function") {
        const maxSteps = parseInt(document.getElementById("statStep")?.textContent || "20", 10) || 20;
        const grade = await api("POST", "/grade", { max_steps: maxSteps });
        if (grade && typeof grade.score !== "undefined") scoreVal = grade.score;
        if (grade && typeof grade.efficiency !== "undefined") efficiencyVal = grade.efficiency;
        if (grade && typeof grade.coordination_score !== "undefined") coordinationVal = grade.coordination_score;
      }
    } catch (_) {}

    if (efficiencyVal === "--") {
      efficiencyVal = document.getElementById("scorePct")?.textContent || "--";
    }
    if (coordinationVal === "--") {
      coordinationVal = document.getElementById("coordinationScore")?.textContent || "--";
    }

    const scoreNum = Number(scoreVal);
    const resultText = !Number.isNaN(scoreNum) && scoreNum >= 0.6 ? "MISSION RESULT: SUCCESS" : "MISSION RESULT: FAIL";
    const scoreEl = document.getElementById("gmDebriefScore");
    const efficiencyEl = document.getElementById("gmDebriefEfficiency");
    const coordEl = document.getElementById("gmDebriefCoordination");
    const resultEl = document.getElementById("gmDebriefResult");
    const breakdownEl = document.getElementById("gmDebriefBreakdown");
    if (scoreEl) scoreEl.textContent = String(scoreVal);
    if (efficiencyEl) efficiencyEl.textContent = String(efficiencyVal);
    if (coordEl) coordEl.textContent = String(coordinationVal);
    if (resultEl) resultEl.textContent = resultText;
    if (breakdownEl) {
      const assigned = document.getElementById("statAssigned")?.textContent || "0";
      const conflicts = document.getElementById("conflictCount")?.textContent || "0";
      breakdownEl.textContent = `Assigned: ${assigned} | Conflicts: ${conflicts}`;
    }

    modal.style.display = "flex";
    addCommentary("Mission debrief generated from /grade response.");
    setRunState("READY");
  }

  function closeDebrief() {
    const modal = document.getElementById("gmDebriefModal");
    if (modal) modal.style.display = "none";
  }

  function hydrateFromApi(path, response) {
    if (!state.visible || !response) return;
    if (path === "/auto_multi") {
      const proposals = response.proposals || [];
      const conflicts = response.conflicts || [];
      proposals.forEach((p) => addLink(p.sensor_id, p.target_id, "#AA8B56", true));
      conflicts.forEach((c) => {
        if (c.sensor_id && c.target_id) addLink(c.sensor_id, c.target_id, "#8B2E2E", false);
      });
      if (proposals.length > 0) {
        const p = proposals[0];
        if (p.agent_id === "satellite") addCommentary(commentaryTemplates.aryaLock);
        if (p.agent_id === "drone") addCommentary(commentaryTemplates.dhuruvMove);
        if (p.agent_id === "radar") addCommentary(commentaryTemplates.rajanTrack);
      }
      updateAgentCard("satellite", { status: "ACTIVE", action: "Priority lock", confidence: 0.82 });
      updateAgentCard("drone", { status: "ACTIVE", action: "Tactical reposition", confidence: 0.74 });
      updateAgentCard("radar", { status: "ACTIVE", action: "Coverage hold", confidence: 0.78 });

      const cmdReward = Number((response.agent_rewards || {}).command || 0);
      updateAgentCard("command", {
        status: cmdReward !== 0 ? "ACTIVE" : "STANDBY",
        action: cmdReward !== 0 ? "Override authority" : "Monitoring",
        confidence: cmdReward !== 0 ? 0.84 : 0.68
      });

      const allSensors = response.observations?.command?.sensors || [];
      const droneDown = allSensors.some((s) => s.type === "drone" && s.available === false);
      if (droneDown && !state.lastDroneDown) {
        setEvent("STORM DETECTED - Drone offline");
        addCommentary(commentaryTemplates.storm);
      }
      state.lastDroneDown = droneDown;

      if (conflicts.length > 0) {
        setEvent("CONFLICT DETECTED");
        addCommentary(commentaryTemplates.conflict);
      }

      const missedCount = (response.info?.missed_targets || []).length;
      if (missedCount === 0) {
        setEvent("OPTIMAL COVERAGE");
        addCommentary(commentaryTemplates.optimal);
      }
    }

    if (path === "/step") {
      const sensorId = document.getElementById("sensorSel")?.value;
      const targetId = document.getElementById("targetSel")?.value;
      if (sensorId && targetId) addLink(sensorId, targetId, "#4E6C50", false);
      addCommentary(commentaryTemplates.assign);
    }

    if (path === "/step/auto" && Array.isArray(response.actions)) {
      response.actions.forEach((a) => addLink(a.sensor_id, a.target_id, "#4E6C50", false));
      addCommentary(commentaryTemplates.assign);
    }

    if (window.GameVisuals && typeof window.GameVisuals.updateVisualsFromState === "function") {
      window.GameVisuals.updateVisualsFromState(response, path, state.lastSelection);
    }
    updateTopStats();
  }

  function patchApi() {
    if (window.__gmApiPatched || typeof api !== "function") return;
    window.__gmApiPatched = true;
    const originalApi = api;
    window.api = async function patchedApi(method, path, body) {
      const res = await originalApi(method, path, body);
      hydrateFromApi(path, res);
      return res;
    };
  }

  function openGame() {
    if (state.visible) return;
    state.visible = true;
    renderOverlay();
    document.body.classList.add("game-mode-active");
    overlay.classList.add("game-visible");
    overlay.style.display = "block";
    updateBriefing();
    updateTopStats();
    patchApi();
    if (window.GameVisuals && typeof window.GameVisuals.init === "function") {
      window.GameVisuals.init();
    }
    window.setTimeout(() => {
      if (window.map && typeof window.map.invalidateSize === "function") {
        window.map.invalidateSize();
      }
    }, 200);

    const sensorSel = document.getElementById("sensorSel");
    const targetSel = document.getElementById("targetSel");
    if (sensorSel) sensorSel.addEventListener("change", handleSelectionHighlights);
    if (targetSel) targetSel.addEventListener("change", handleSelectionHighlights);

    addCommentary("Game mode activated. Live battlefield linked.");
    setRunState("MISSION READY");

    document.getElementById("gmMissionBtn")?.addEventListener("click", missionMode);
    document.getElementById("gmManualBtn")?.addEventListener("click", manualStepGame);
    document.getElementById("gmAutoBtn")?.addEventListener("click", autoModeGame);
    document.getElementById("gmGradeBtn")?.addEventListener("click", showDebrief);
    document.getElementById("gmExitBtn")?.addEventListener("click", exitGame);
    document.getElementById("gmCloseDebriefBtn")?.addEventListener("click", closeDebrief);

    state.drawInterval = window.setInterval(() => {
      if (!state.visible) return;
      drawLinks();
      updateTopStats();
    }, 300);
  }

  function exitGame() {
    closeDebrief();
    overlay.classList.remove("game-visible");
    overlay.style.display = "none";
    overlay.innerHTML = "";
    state.visible = false;
    state.busy = false;
    state.links = [];
    state.lastDroneDown = false;
    if (state.drawInterval) {
      window.clearInterval(state.drawInterval);
      state.drawInterval = null;
    }
    if (window.GameVisuals && typeof window.GameVisuals.clearAllLines === "function") {
      window.GameVisuals.clearAllLines();
    }
    document.body.classList.remove("game-mode-active");
    window.setTimeout(() => {
      if (window.map && typeof window.map.invalidateSize === "function") {
        window.map.invalidateSize();
      }
    }, 200);
  }

  openBtn.addEventListener("click", openGame);
})();
