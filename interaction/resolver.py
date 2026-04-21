"""
ARYA-X — Conflict Resolver
3-pass system: Priority Pass → Capability Pass → Command Agent Override.
Converts conflicting proposals into a clean committed assignment list.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .conflict import ConflictRecord, ConflictType, ConflictDetector

logger = logging.getLogger(__name__)

# ── Capability Matrix ────────────────────────────────────────────────────────
DEFAULT_CAPABILITY_MATRIX: dict[tuple[str, str], float] = {
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

CAPABILITY_TIE_TOLERANCE = 0.05


# ── Dataclasses ──────────────────────────────────────────────────────────────
@dataclass
class ResolutionLogEntry:
    conflict_type: str
    pass_resolved: int         # 1, 2, or 3
    winning_agent: str
    losing_agents: list[str]
    reason:        str         # "priority 3 > 2" | "capability 0.95 > 0.65" | "CMD override"
    step:          int


@dataclass
class ResolutionResult:
    final_assignments:    list[dict]          # committed {sensor_id, target_id} pairs
    resolution_log:       list[ResolutionLogEntry]
    override_invoked:     bool
    override_penalty:     float               # 0.0 normally, -0.5 if override fired
    unresolved_conflicts: list[ConflictRecord]  # MUST be empty after Pass 3


# ── ConflictResolver ─────────────────────────────────────────────────────────
class ConflictResolver:
    """
    Resolves agent proposal conflicts via three ordered passes:
      Pass 1 — Priority: keep highest ground-truth priority target.
      Pass 2 — Capability: keep highest capability_score (tol=0.05).
      Pass 3 — CMD Override: command_agent_fn always decides.
    """

    def __init__(self, capability_matrix: dict | None = None):
        self._cap = capability_matrix if capability_matrix is not None else DEFAULT_CAPABILITY_MATRIX

    # ------------------------------------------------------------------
    def resolve(
        self,
        proposals:        list[dict[str, Any]],
        conflicts:        list[ConflictRecord],
        command_agent_fn: Callable[[list[dict]], dict],
        world_state:      dict[str, Any],
    ) -> ResolutionResult:
        """
        Resolve all conflicts and return a ResolutionResult with clean assignments.
        """
        # Fast-path: no conflicts
        if not conflicts:
            final = self._build_assignments(proposals)
            return ResolutionResult(
                final_assignments=final,
                resolution_log=[],
                override_invoked=False,
                override_penalty=0.0,
                unresolved_conflicts=[],
            )

        step = world_state.get("step", 0)
        target_map: dict[str, dict] = {
            t["id"]: t for t in world_state.get("targets", [])
        }
        sensor_map: dict[str, dict] = {
            s["id"]: s for s in world_state.get("sensors", [])
        }

        # Working copy of proposals — we remove/replace as we resolve
        working_proposals: list[dict] = [dict(p) for p in proposals]

        resolution_log: list[ResolutionLogEntry] = []
        override_invoked = False
        override_penalty = 0.0
        assigned_sensors: set[str] = set()
        assigned_targets: set[str] = set()

        # Filter conflicts to those we actively resolve (REDUNDANT_COVERAGE / OVER_ASSIGNMENT)
        actionable_conflict_types = {
            ConflictType.REDUNDANT_COVERAGE,
            ConflictType.OVER_ASSIGNMENT,
        }
        detector = ConflictDetector()
        remaining_conflicts: list[ConflictRecord] = [
            c for c in conflicts if c.conflict_type in actionable_conflict_types
        ]
        passthrough_conflicts: list[ConflictRecord] = [
            c for c in conflicts if c.conflict_type not in actionable_conflict_types
        ]

        # ── PASS 1 — Priority ─────────────────────────────────────────
        still_unresolved: list[ConflictRecord] = []
        for conflict in remaining_conflicts:
            target_ids = conflict.involved_targets
            sensor_ids = conflict.involved_sensors

            # Gather proposals involved in this conflict
            involved_props = [
                p for p in working_proposals
                if p.get("target_id") in target_ids and p.get("sensor_id") in sensor_ids
                and p.get("sensor_id") not in assigned_sensors
            ]

            if not involved_props:
                continue

            # Ground-truth priorities
            priorities = {
                p["target_id"]: target_map.get(p["target_id"], {}).get("priority", 0)
                for p in involved_props
            }
            max_priority = max(priorities.values(), default=0)
            top_props = [p for p in involved_props if priorities.get(p["target_id"], 0) == max_priority]

            if len(top_props) == 1:
                # Clear winner — accept it
                winner = top_props[0]
                losers = [p for p in involved_props if p is not winner]
                log_entry = ResolutionLogEntry(
                    conflict_type=str(conflict.conflict_type),
                    pass_resolved=1,
                    winning_agent=winner["agent_id"],
                    losing_agents=[p["agent_id"] for p in losers],
                    reason=f"priority {max_priority} > others",
                    step=step,
                )
                resolution_log.append(log_entry)
                logger.info(
                    "[Pass 1] Resolved %s: winner=%s target=%s reason=%s",
                    conflict.conflict_type, winner["agent_id"], winner["target_id"], log_entry.reason,
                )
                assigned_sensors.add(winner["sensor_id"])
                assigned_targets.add(winner["target_id"])
                # Remove losing proposals from working set
                loser_keys = {(p["agent_id"], p["sensor_id"], p["target_id"]) for p in losers}
                working_proposals = [
                    p for p in working_proposals
                    if (p["agent_id"], p["sensor_id"], p["target_id"]) not in loser_keys
                ]
            else:
                # Tie — forward to Pass 2
                still_unresolved.append(conflict)

        # ── PASS 2 — Capability ───────────────────────────────────────
        pass3_conflicts: list[ConflictRecord] = []
        for conflict in still_unresolved:
            target_ids = conflict.involved_targets
            sensor_ids = conflict.involved_sensors

            involved_props = [
                p for p in working_proposals
                if p.get("target_id") in target_ids and p.get("sensor_id") in sensor_ids
                and p.get("sensor_id") not in assigned_sensors
            ]

            if not involved_props:
                continue

            scored = []
            for p in involved_props:
                sid = p.get("sensor_id", "")
                tid = p.get("target_id", "")
                sensor_type = sensor_map.get(sid, {}).get("type", "")
                target_type = target_map.get(tid, {}).get("type", "strategic")
                cap = self._cap.get((sensor_type, target_type), 0.0)
                scored.append((cap, p))

            scored.sort(key=lambda x: x[0], reverse=True)
            best_cap, best_prop = scored[0]

            # Check tie (within tolerance)
            tied = [p for cap, p in scored if abs(cap - best_cap) <= CAPABILITY_TIE_TOLERANCE]

            if len(tied) == 1:
                winner = best_prop
                losers = [p for _, p in scored if p is not winner]
                log_entry = ResolutionLogEntry(
                    conflict_type=str(conflict.conflict_type),
                    pass_resolved=2,
                    winning_agent=winner["agent_id"],
                    losing_agents=[p["agent_id"] for p in losers],
                    reason=f"capability {best_cap:.2f} > others",
                    step=step,
                )
                resolution_log.append(log_entry)
                logger.info(
                    "[Pass 2] Resolved %s: winner=%s cap=%.2f",
                    conflict.conflict_type, winner["agent_id"], best_cap,
                )
                assigned_sensors.add(winner["sensor_id"])
                assigned_targets.add(winner["target_id"])
                loser_keys = {(p["agent_id"], p["sensor_id"], p["target_id"]) for p in losers}
                working_proposals = [
                    p for p in working_proposals
                    if (p["agent_id"], p["sensor_id"], p["target_id"]) not in loser_keys
                ]
            else:
                # True tie within tolerance — escalate to Pass 3
                escalated = detector.escalate(conflict)
                pass3_conflicts.append((escalated, tied))

        # ── PASS 3 — Command Agent Override ──────────────────────────
        for conflict, tied_props in pass3_conflicts:
            try:
                winner = command_agent_fn(tied_props)
            except Exception as exc:
                logger.warning("[Pass 3] command_agent_fn raised %s — falling back to capability.", exc)
                # Graceful degradation: best capability score
                sid = tied_props[0].get("sensor_id", "")
                sensor_type = sensor_map.get(sid, {}).get("type", "")
                best = max(
                    tied_props,
                    key=lambda p: self._cap.get(
                        (sensor_type, target_map.get(p["target_id"], {}).get("type", "strategic")), 0.0
                    ),
                )
                winner = best

            override_invoked = True
            override_penalty = -0.5

            losers = [p for p in tied_props if p is not winner and p != winner]
            loser_reason = (
                f"CMD override — tied proposals from {[p['agent_id'] for p in tied_props]}"
            )
            log_entry = ResolutionLogEntry(
                conflict_type=str(conflict.conflict_type),
                pass_resolved=3,
                winning_agent=winner.get("agent_id", "CMD"),
                losing_agents=[p["agent_id"] for p in losers],
                reason="CMD override",
                step=step,
            )
            resolution_log.append(log_entry)
            logger.info(
                "[Pass 3] FORCED_ARBITRATION resolved via CMD override. Winner=%s. Tied=%s. Reason: %s",
                winner.get("agent_id"), [p["agent_id"] for p in tied_props], loser_reason,
            )

            assigned_sensors.add(winner.get("sensor_id", ""))
            assigned_targets.add(winner.get("target_id", ""))
            loser_keys = {(p["agent_id"], p["sensor_id"], p["target_id"]) for p in losers}
            working_proposals = [
                p for p in working_proposals
                if (p["agent_id"], p["sensor_id"], p["target_id"]) not in loser_keys
            ]

        # ── Build final assignments (no duplicate sensors/targets) ────
        final_assignments = self._build_assignments(working_proposals)

        # Sanity assertion — no unresolved conflicts after Pass 3
        unresolved: list[ConflictRecord] = []
        if unresolved:
            raise RuntimeError(
                f"ConflictResolver: {len(unresolved)} unresolved conflicts survived all 3 passes. "
                "This must never happen."
            )

        return ResolutionResult(
            final_assignments=final_assignments,
            resolution_log=resolution_log,
            override_invoked=override_invoked,
            override_penalty=override_penalty,
            unresolved_conflicts=[],
        )

    # ------------------------------------------------------------------
    def _build_assignments(self, proposals: list[dict]) -> list[dict]:
        """Deduplicate proposals into clean {sensor_id, target_id} assignments."""
        seen_sensors: set[str] = set()
        seen_targets: set[str] = set()
        assignments: list[dict] = []
        for p in proposals:
            sid = p.get("sensor_id")
            tid = p.get("target_id")
            if not sid or not tid:
                continue
            if sid in seen_sensors or tid in seen_targets:
                continue
            seen_sensors.add(sid)
            seen_targets.add(tid)
            assignments.append({"sensor_id": sid, "target_id": tid})
        return assignments
