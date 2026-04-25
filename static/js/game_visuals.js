(function () {
  function getMapInstance() {
    if (window.map && typeof window.map.addLayer === "function") return window.map;
    return null;
  }

  function markerElementByTitle(match, startsWith) {
    const markers = Array.from(document.querySelectorAll(".leaflet-marker-icon[title]"));
    if (startsWith) return markers.find((m) => (m.getAttribute("title") || "").startsWith(match));
    return markers.find((m) => (m.getAttribute("title") || "") === match);
  }

  function markerLatLng(map, markerEl) {
    if (!map || !markerEl) return null;
    const mapRect = map.getContainer().getBoundingClientRect();
    const rect = markerEl.getBoundingClientRect();
    const point = L.point((rect.left + rect.width / 2) - mapRect.left, (rect.top + rect.height / 2) - mapRect.top);
    return map.containerPointToLatLng(point);
  }

  const palette = {
    success: "#16a34a",
    conflict: "#dc2626",
    pending: "#d97706"
  };

  const visuals = {
    map: null,
    activeLines: [],
    targetPulseClass: "gm-target-pulse",
    sensorOfflineClass: "gm-sensor-offline",

    init() {
      this.map = getMapInstance();
      return this.map;
    },

    clearAllLines() {
      if (!this.map) this.init();
      this.activeLines.forEach((line) => {
        try { this.map.removeLayer(line); } catch (_) {}
      });
      this.activeLines = [];
    },

    drawAssignments(assignments) {
      if (!this.map) this.init();
      if (!this.map || !Array.isArray(assignments)) return;

      assignments.forEach((a) => {
        const sensorEl = markerElementByTitle(`${a.sensorId} `, true);
        const targetEl = markerElementByTitle(a.targetId, false);
        const sLatLng = markerLatLng(this.map, sensorEl);
        const tLatLng = markerLatLng(this.map, targetEl);
        if (!sLatLng || !tLatLng) return;

        const color = palette[a.status] || palette.pending;
        const line = L.polyline([sLatLng, tLatLng], {
          color,
          weight: 3,
          opacity: 0.82,
          dashArray: a.status === "pending" ? "6, 8" : null
        }).addTo(this.map);
        this.activeLines.push(line);
      });
    },

    applyTargetPriorityVisuals() {
      const threatList = Array.from(document.querySelectorAll("#threatList .threat-item"));
      const priorityById = {};
      threatList.forEach((item) => {
        const strong = item.querySelector("strong");
        if (!strong) return;
        const id = strong.textContent.trim();
        let p = "p1";
        if (item.classList.contains("p3")) p = "p3";
        else if (item.classList.contains("p2")) p = "p2";
        priorityById[id] = p;
      });

      const markers = Array.from(document.querySelectorAll(".leaflet-marker-icon[title]"));
      markers.forEach((m) => {
        const title = m.getAttribute("title") || "";
        const p = priorityById[title];
        if (!p) return;
        m.style.filter = "";
        m.classList.remove(this.targetPulseClass);
        if (p === "p3") {
          m.style.filter = "drop-shadow(0 0 6px rgba(220,38,38,0.9))";
          m.classList.add(this.targetPulseClass);
        } else if (p === "p2") {
          m.style.filter = "drop-shadow(0 0 5px rgba(217,119,6,0.75))";
        } else {
          m.style.filter = "drop-shadow(0 0 4px rgba(234,179,8,0.65))";
        }
      });
    },

    applySensorStateVisuals(response) {
      const sensors = response?.observations?.command?.sensors || response?.observation?.sensors || [];
      if (!Array.isArray(sensors)) return;
      sensors.forEach((s) => {
        const marker = markerElementByTitle(`${s.id} `, true);
        if (!marker) return;
        marker.classList.remove(this.sensorOfflineClass);
        marker.style.filter = "";
        if (s.available === false) {
          marker.classList.add(this.sensorOfflineClass);
          marker.style.filter = "grayscale(1) opacity(0.6)";
        } else {
          marker.style.filter = "drop-shadow(0 0 6px rgba(22,163,74,0.6))";
        }
      });
    },

    updateVisualsFromState(response, path, selected) {
      if (!response) return;
      if (!this.map) this.init();
      if (!this.map) return;

      const assignments = [];
      if (path === "/auto_multi") {
        (response.proposals || []).forEach((p) => {
          assignments.push({ sensorId: p.sensor_id, targetId: p.target_id, status: "pending" });
        });
        (response.conflicts || []).forEach((c) => {
          if (c.sensor_id && c.target_id) {
            assignments.push({ sensorId: c.sensor_id, targetId: c.target_id, status: "conflict" });
          }
        });
      } else if (path === "/step/auto" && Array.isArray(response.actions)) {
        response.actions.forEach((a) => assignments.push({ sensorId: a.sensor_id, targetId: a.target_id, status: "success" }));
      } else if (path === "/step" && selected?.sensorId && selected?.targetId) {
        assignments.push({ sensorId: selected.sensorId, targetId: selected.targetId, status: "success" });
      }

      this.clearAllLines();
      this.drawAssignments(assignments);
      this.applyTargetPriorityVisuals();
      this.applySensorStateVisuals(response);
    }
  };

  window.GameVisuals = visuals;
})();
