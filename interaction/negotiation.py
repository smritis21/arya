"""
ARYA-X — Negotiation Layer
Top-level orchestrator of the proposal → detect → resolve → commit pipeline.
Called once per environment step.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .conflict import ConflictDetector, ConflictRecord
from .resolver import ConflictResolver, ResolutionResult

logger = logging.getLogger(__name__)

_HISTORY_CAP = 1000


# ── NegotiationResult ────────────────────────────────────────────────────────
@dataclass
class NegotiationResult:
    final_assignments:  list[dict]
    conflicts_detected: list[ConflictRecord]
    resolution_result:  ResolutionResult
    override_invoked:   bool
    override_penalty:   float
    step_metrics:       dict


# ── NegotiationLayer ─────────────────────────────────────────────────────────
class NegotiationLayer:
    """
    Full pipeline per env step:
      1. Validate proposals
      2. ConflictDetector.detect()
      3. ConflictResolver.resolve()
      4. Post-resolution validation
      5. Return NegotiationResult + update history
    """

    def __init__(self) -> None:
        self.detector  = ConflictDetector()
        self.resolver  = ConflictResolver()
        self._history: list[NegotiationResult] = []

    # ── Main entry point ─────────────────────────────────────────────
    def negotiate(
        self,
        proposals:        list[dict[str, Any]],
        world_state:      dict[str, Any],
        command_agent_fn: Callable[[list[dict]], dict],
    ) -> NegotiationResult:
        """
        Run the full negotiation pipeline for one environment step.
        Never raises — degrades gracefully on validation or resolver errors.
        """
        step = world_state.get("step", 0)

        # ── 1. Validate proposals ──────────────────────────────────
        valid_proposals = self._validate_proposals(proposals, world_state)

        # ── 2. Detect conflicts ────────────────────────────────────
        conflicts: list[ConflictRecord] = []
        try:
            conflicts = self.detector.detect(valid_proposals, world_state)
        except Exception as exc:
            logger.error("ConflictDetector.detect() raised unexpectedly: %s", exc)

        # ── 3. Resolve conflicts ───────────────────────────────────
        resolution_result: ResolutionResult | None = None
        try:
            resolution_result = self.resolver.resolve(
                valid_proposals, conflicts, command_agent_fn, world_state
            )
        except RuntimeError as exc:
            logger.error("ConflictResolver.resolve() raised RuntimeError: %s", exc)
            # Fallback: return empty assignments rather than corrupt state
            from .resolver import ResolutionResult as RR
            resolution_result = RR(
                final_assignments=[],
                resolution_log=[],
                override_invoked=False,
                override_penalty=0.0,
                unresolved_conflicts=list(conflicts),
            )
        except Exception as exc:
            logger.error("ConflictResolver.resolve() raised unexpected error: %s", exc)
            from .resolver import ResolutionResult as RR
            resolution_result = RR(
                final_assignments=[],
                resolution_log=[],
                override_invoked=False,
                override_penalty=0.0,
                unresolved_conflicts=[],
            )

        # ── 4. Post-resolution validation ─────────────────────────
        final_assignments = self._post_validate(resolution_result.final_assignments)

        # ── 5. Assemble metrics ────────────────────────────────────
        passes_used = [entry.pass_resolved for entry in resolution_result.resolution_log]
        conflict_types = list({str(c.conflict_type) for c in conflicts})

        step_metrics = {
            "step":                   step,
            "num_proposals":          len(proposals),
            "num_conflicts":          len(conflicts),
            "conflict_types":         conflict_types,
            "resolution_passes_used": passes_used,
            "override_invoked":       resolution_result.override_invoked,
            "override_penalty":       resolution_result.override_penalty,
            "final_assignment_count": len(final_assignments),
        }

        result = NegotiationResult(
            final_assignments=final_assignments,
            conflicts_detected=conflicts,
            resolution_result=resolution_result,
            override_invoked=resolution_result.override_invoked,
            override_penalty=resolution_result.override_penalty,
            step_metrics=step_metrics,
        )

        # ── 6. Update history (capped) ────────────────────────────
        if len(self._history) >= _HISTORY_CAP:
            self._history.pop(0)
        self._history.append(result)

        return result

    # ── Key training metric ──────────────────────────────────────────
    def get_conflict_rate(self, window: int = 50) -> float:
        """
        Fraction of steps that had at least one conflict over the last `window` steps.
        """
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        steps_with_conflict = sum(1 for r in recent if r.step_metrics["num_conflicts"] > 0)
        return min(1.0, steps_with_conflict / len(recent))

    # ── Accessors ────────────────────────────────────────────────────
    def get_step_metrics_history(self) -> list[dict]:
        return [r.step_metrics for r in self._history]

    def reset(self) -> None:
        """Clear history — call at the start of each episode."""
        self._history.clear()

    # ── Internal helpers ─────────────────────────────────────────────
    def _validate_proposals(
        self,
        proposals: list[dict],
        world_state: dict,
    ) -> list[dict]:
        """
        Remove proposals whose sensor_id or target_id don't exist / aren't active.
        Validation errors are logged at WARNING level but never raised.
        """
        valid_sensor_ids = {
            s["id"] for s in world_state.get("sensors", [])
            if s.get("available", True)
        }
        valid_target_ids = {
            t["id"] for t in world_state.get("targets", [])
            if t.get("active", True)
        }

        valid: list[dict] = []
        for p in proposals:
            sid = p.get("sensor_id")
            tid = p.get("target_id")
            if not sid or not tid:
                logger.warning(
                    "Proposal from agent %s missing sensor_id or target_id — dropped.",
                    p.get("agent_id", "?"),
                )
                continue
            if sid not in valid_sensor_ids:
                logger.warning(
                    "Proposal from %s: sensor_id=%s is unavailable/unknown — dropped.",
                    p.get("agent_id", "?"), sid,
                )
                continue
            if tid not in valid_target_ids:
                logger.warning(
                    "Proposal from %s: target_id=%s is inactive/unknown — dropped.",
                    p.get("agent_id", "?"), tid,
                )
                continue
            valid.append(p)

        return valid

    def _post_validate(self, assignments: list[dict]) -> list[dict]:
        """Assert no duplicate sensor_ids or target_ids in final assignments."""
        seen_sensors: set[str] = set()
        seen_targets: set[str] = set()
        clean: list[dict] = []
        for a in assignments:
            sid = a.get("sensor_id")
            tid = a.get("target_id")
            if sid in seen_sensors:
                logger.warning("Post-validation: duplicate sensor_id=%s detected — skipped.", sid)
                continue
            if tid in seen_targets:
                logger.warning("Post-validation: duplicate target_id=%s detected — skipped.", tid)
                continue
            seen_sensors.add(sid)
            seen_targets.add(tid)
            clean.append(a)
        return clean
