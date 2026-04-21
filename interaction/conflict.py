"""
ARYA-X — Conflict Detector
Detects four types of multi-agent proposal conflicts each environment step.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ── Capability Matrix ────────────────────────────────────────────────────────
CAPABILITY_MATRIX: dict[tuple[str, str], float] = {
    ("satellite", "strategic"): 0.95,
    ("satellite", "kinetic"):   0.40,
    ("satellite", "airspace"):  0.60,
    ("drone",     "kinetic"):   0.95,
    ("drone",     "strategic"): 0.30,
    ("drone",     "airspace"):  0.50,
    ("radar",     "airspace"):  0.95,
    ("radar",     "kinetic"):   0.65,
    ("radar",     "strategic"): 0.45,
}

# Target-priority → max sensors allowed
PRIORITY_MAX_SENSORS: dict[int, int] = {1: 1, 2: 2, 3: 2}


# ── Enums ────────────────────────────────────────────────────────────────────
class ConflictType(str, Enum):
    REDUNDANT_COVERAGE  = "REDUNDANT_COVERAGE"
    OVER_ASSIGNMENT     = "OVER_ASSIGNMENT"
    MISSED_PRIORITY_3   = "MISSED_PRIORITY_3"
    FORCED_ARBITRATION  = "FORCED_ARBITRATION"


class Severity(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


# ── ConflictRecord ───────────────────────────────────────────────────────────
@dataclass
class ConflictRecord:
    conflict_type:    ConflictType
    severity:         Severity
    involved_agents:  list[str]
    involved_targets: list[str]
    involved_sensors: list[str]
    description:      str
    step:             int = 0


# ── ConflictDetector ─────────────────────────────────────────────────────────
class ConflictDetector:
    """
    Deterministic detector for four conflict types:
      1. REDUNDANT_COVERAGE  (MEDIUM) — multiple sensors proposed for same target
      2. OVER_ASSIGNMENT     (LOW)    — more sensors than priority allows
      3. MISSED_PRIORITY_3   (HIGH)   — critical target uncovered despite capacity
      4. FORCED_ARBITRATION  (HIGH)   — escalated by ConflictResolver after Pass 2
    """

    # ------------------------------------------------------------------
    def detect(
        self,
        proposals:   list[dict[str, Any]],
        world_state: dict[str, Any],
    ) -> list[ConflictRecord]:
        """
        Analyse agent proposals and return all ConflictRecords for this step.
        Always deterministic: same inputs → same outputs.
        """
        if not proposals:
            return []

        # ── Validate world_state keys ─────────────────────────────────
        required_keys = {"targets", "sensors", "idle_sensors", "step"}
        missing = required_keys - world_state.keys()
        if missing:
            logger.error(
                "world_state missing keys %s — skipping conflict detection.", missing
            )
            return []

        step: int = world_state.get("step", 0)
        targets: list[dict] = world_state.get("targets", [])
        idle_sensors: list[str] = world_state.get("idle_sensors", [])
        sensors: list[dict] = world_state.get("sensors", [])

        conflicts: list[ConflictRecord] = []

        # Build lookup maps used across checks
        target_map: dict[str, dict] = {t["id"]: t for t in targets if t.get("active", True)}
        sensor_map: dict[str, dict] = {s["id"]: s for s in sensors}

        # Proposals grouped by target_id -> list[{proposal}]
        by_target: dict[str, list[dict]] = defaultdict(list)
        for p in proposals:
            tid = p.get("target_id")
            if tid:
                by_target[tid].append(p)

        # ── CHECK 1 — REDUNDANT_COVERAGE ─────────────────────────────
        for target_id, props in by_target.items():
            # Multiple *different* sensors proposed for the same target
            unique_sensors = list({p["sensor_id"] for p in props})
            if len(unique_sensors) > 1:
                agents = sorted({p["agent_id"] for p in props})
                rec = ConflictRecord(
                    conflict_type=ConflictType.REDUNDANT_COVERAGE,
                    severity=Severity.MEDIUM,
                    involved_agents=agents,
                    involved_targets=[target_id],
                    involved_sensors=unique_sensors,
                    description=(
                        f"Multiple sensors {unique_sensors} all proposed for target "
                        f"{target_id} by agents {agents} — one sensor wasted."
                    ),
                    step=step,
                )
                logger.warning("CONFLICT [REDUNDANT_COVERAGE] step=%d %s", step, rec.description)
                conflicts.append(rec)

        # ── CHECK 2 — OVER_ASSIGNMENT ─────────────────────────────────
        for target_id, props in by_target.items():
            unique_sensors = list({p["sensor_id"] for p in props})
            tgt = target_map.get(target_id)
            if tgt is None:
                continue
            priority = tgt.get("priority", 1)
            max_allowed = PRIORITY_MAX_SENSORS.get(priority, 1)
            if len(unique_sensors) > max_allowed:
                agents = sorted({p["agent_id"] for p in props})
                rec = ConflictRecord(
                    conflict_type=ConflictType.OVER_ASSIGNMENT,
                    severity=Severity.LOW,
                    involved_agents=agents,
                    involved_targets=[target_id],
                    involved_sensors=unique_sensors,
                    description=(
                        f"Target {target_id} (priority {priority}) received "
                        f"{len(unique_sensors)} sensor proposals but max allowed is "
                        f"{max_allowed}."
                    ),
                    step=step,
                )
                logger.warning("CONFLICT [OVER_ASSIGNMENT] step=%d %s", step, rec.description)
                conflicts.append(rec)

        # ── CHECK 3 — MISSED_PRIORITY_3 ──────────────────────────────
        # Targets covered by at least one proposal
        covered_targets = set(by_target.keys())

        for tgt in targets:
            if not tgt.get("active", True):
                continue
            if tgt.get("priority", 0) != 3:
                continue
            tid = tgt["id"]
            if tid in covered_targets:
                continue

            # Check whether any idle sensor has capability > 0.5 for this target
            target_type = tgt.get("type", "strategic")
            capable_idle = [
                sid for sid in idle_sensors
                if CAPABILITY_MATRIX.get(
                    (sensor_map.get(sid, {}).get("type", ""), target_type), 0.0
                ) > 0.5
            ]
            if capable_idle:
                rec = ConflictRecord(
                    conflict_type=ConflictType.MISSED_PRIORITY_3,
                    severity=Severity.HIGH,
                    involved_agents=[],
                    involved_targets=[tid],
                    involved_sensors=capable_idle,
                    description=(
                        f"Priority-3 target {tid} (type={target_type}) was ignored despite "
                        f"capable idle sensors {capable_idle}."
                    ),
                    step=step,
                )
                logger.warning("CONFLICT [MISSED_PRIORITY_3] step=%d %s", step, rec.description)
                conflicts.append(rec)

        return conflicts

    # ------------------------------------------------------------------
    def escalate(self, record: ConflictRecord) -> ConflictRecord:
        """
        Upgrade an existing ConflictRecord to FORCED_ARBITRATION / HIGH.
        Called by ConflictResolver when a conflict survives Pass 1 and Pass 2.
        """
        record.conflict_type = ConflictType.FORCED_ARBITRATION
        record.severity       = Severity.HIGH
        record.description    = (
            "[FORCED_ARBITRATION] " + record.description
        )
        logger.warning("CONFLICT [FORCED_ARBITRATION] escalated from %s", record.original_type
                       if hasattr(record, "original_type") else "unknown")
        return record
