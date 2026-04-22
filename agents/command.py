from typing import List, Optional
from agents.base_agent import BaseAgent
from env.models import AgentObservation, Proposal, ConflictRecord


class CommandAgent(BaseAgent):
    """
    Command agent specialization:
    - Does NOT claim sensors proactively in normal operation
    - Activates ONLY when:
        (a) uncovered P3 targets exist AND no other agent has proposed for them, OR
        (b) conflict_rate in recent history exceeds a threshold (system-level override)
    - Uses ANY sensor type (all = "command")
    - When overriding: picks the highest-priority-to-system-reward assignment
      (priority × sensor_range as a tie-breaker for coordination value)
    - Deliberately withholds proposals late in episode to keep reserve capacity
    - Confidence always 1.0 (authoritative)
    """

    def __init__(self, max_steps: int = 20):
        super().__init__("command", "command", "all")
        self.max_steps = max_steps
        self._all_proposals: List[Proposal] = []  # set externally by orchestrator
        self.episode_step: int = 0

    def observe(self, obs: AgentObservation, proposals: Optional[List[Proposal]] = None) -> None:
        super().observe(obs)
        self._all_proposals = proposals or []
        self.episode_step = obs.timestep

    def propose(self) -> List[Proposal]:
        """
        Command override logic:
        1. Compute which P3 targets are currently uncovered by ANY other agent.
        2. Check recent conflict rate — if high, also cover P2 conflicts.
        3. Use idle sensors to fill gaps. Reserve the last sensor at end-of-episode.
        4. If no gaps, return nothing (minimal footprint).
        """
        covered_targets = {p.target_id for p in self._all_proposals}
        used_sensors = {p.sensor_id for p in self._all_proposals}

        active_targets = self._active_targets()
        available_sensors = self._available_sensors()

        # ── Gap detection ──────────────────────────────────────────────────────
        uncovered_p3 = [
            t for t in active_targets
            if t.priority == 3 and t.id not in covered_targets
        ]

        # High conflict rate: also cover uncovered P2
        recent_conflict_rate = self._estimate_conflict_rate()
        uncovered_p2 = []
        if recent_conflict_rate > 0.4:
            uncovered_p2 = [
                t for t in active_targets
                if t.priority == 2 and t.id not in covered_targets
            ]

        # Targets command will act on (P3 first, then P2 if conflict is high)
        gap_targets = uncovered_p3 + uncovered_p2
        if not gap_targets:
            # No gaps — command stays silent (minimal footprint)
            self.recent_assignments.append({
                "step": self.episode_step, "proposals": 0, "target_ids": []
            })
            return []

        # ── Sensor selection ───────────────────────────────────────────────────
        idle_sensors = [s for s in available_sensors if s.id not in used_sensors]

        # Late-episode reserve: hold back the last idle sensor
        late_episode = self.episode_step > self.max_steps * 0.65
        if late_episode and len(idle_sensors) > 1:
            idle_sensors = idle_sensors[:-1]

        if not idle_sensors:
            # No idle sensors — nothing command can do without conflicting
            self.recent_assignments.append({
                "step": self.episode_step, "proposals": 0, "target_ids": []
            })
            return []

        # ── Build override proposals ───────────────────────────────────────────
        # Sort gap targets by system-level reward value: priority descending,
        # then highest-range available sensor as tie-breaker
        gap_sorted = sorted(gap_targets, key=lambda t: -t.priority)

        proposals = []
        used_targets: set = set()
        for sensor in idle_sensors:
            for target in gap_sorted:
                if target.id not in used_targets:
                    proposals.append(Proposal(
                        sensor_id=sensor.id,
                        target_id=target.id,
                        agent_id=self.agent_id,
                        priority_estimate=target.priority,
                        confidence=1.0,  # authoritative override
                    ))
                    used_targets.add(target.id)
                    break

        self.recent_assignments.append({
            "step": self.episode_step,
            "proposals": len(proposals),
            "target_ids": [p.target_id for p in proposals],
        })
        return proposals

    def issue_override(self, conflict) -> Optional[Proposal]:
        """Called by resolver when conflict persists. Returns a binding override proposal.
        Works with interaction.conflict.ConflictRecord (involved_targets/involved_agents)
        and env.models.ConflictRecord (target_id/agents_involved).
        """
        # Resolve target_id from whichever schema is present
        involved_targets = getattr(conflict, "involved_targets", None)
        target_id = (
            involved_targets[0]
            if involved_targets
            else getattr(conflict, "target_id", None)
        )
        if target_id is None:
            return None

        target = next((t for t in self._active_targets() if t.id == target_id), None)
        if target is None:
            return None

        # Resolve involved agents from whichever schema is present
        involved = (
            getattr(conflict, "involved_agents", None)
            or getattr(conflict, "agents_involved", [])
        )

        used_sensors = {p.sensor_id for p in self._all_proposals}
        free_sensor = next(
            (s for s in self._available_sensors() if s.id not in used_sensors), None
        )

        if free_sensor is None:
            override_prop = next(
                (p for p in self._all_proposals
                 if p.agent_id in involved and p.target_id == target_id),
                None
            )
            if override_prop:
                return Proposal(
                    sensor_id=override_prop.sensor_id,
                    target_id=target_id,
                    agent_id=self.agent_id,
                    priority_estimate=target.priority,
                    confidence=1.0,
                )
            return None

        return Proposal(
            sensor_id=free_sensor.id,
            target_id=target_id,
            agent_id=self.agent_id,
            priority_estimate=target.priority,
            confidence=1.0,
        )

    def _estimate_conflict_rate(self) -> float:
        """
        Estimate recent conflict rate from conflict_history.
        Returns fraction of recent steps that had any conflict.
        """
        recent = self.conflict_history[-6:]  # last 6 recorded conflicts
        if not recent:
            return 0.0
        # Each ConflictRecord is one conflict event — if many recent, rate is high
        return min(1.0, len(recent) / 6.0)
