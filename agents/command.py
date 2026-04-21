from typing import List, Optional
from agents.base_agent import BaseAgent
from env.models import AgentObservation, Proposal, ConflictRecord


class CommandAgent(BaseAgent):
    def __init__(self, max_steps: int = 20):
        super().__init__("command", "command", "all")
        self.max_steps = max_steps
        self._all_proposals: List[Proposal] = []
        self.episode_step: int = 0

    def observe(self, obs: AgentObservation, proposals: Optional[List[Proposal]] = None) -> None:
        super().observe(obs)
        self._all_proposals = proposals or []
        self.episode_step = obs.timestep

    def propose(self) -> List[Proposal]:
        """Fill uncovered priority-3 targets with idle sensors."""
        covered_targets = {p.target_id for p in self._all_proposals}
        used_sensors = {p.sensor_id for p in self._all_proposals}

        uncovered_p3 = [t for t in self._active_targets()
                        if t.priority == 3 and t.id not in covered_targets]
        idle_sensors = [s for s in self._available_sensors() if s.id not in used_sensors]

        # Long-horizon: late in episode, hold one sensor in reserve
        late_episode = self.episode_step > self.max_steps * 0.6
        if late_episode and len(idle_sensors) > 1:
            idle_sensors = idle_sensors[:-1]

        proposals = []
        for sensor, target in zip(idle_sensors, uncovered_p3):
            proposals.append(Proposal(
                sensor_id=sensor.id,
                target_id=target.id,
                agent_id=self.agent_id,
                priority_estimate=target.priority,
                confidence=1.0,
            ))
        self.recent_assignments.append({"step": self.episode_step, "proposals": len(proposals)})
        return proposals

    def issue_override(self, conflict: ConflictRecord) -> Optional[Proposal]:
        """Called by resolver when conflict persists. Returns binding proposal."""
        # Find the target from conflict
        target_id = conflict.target_id
        target = next((t for t in self._active_targets() if t.id == target_id), None)
        if target is None:
            return None

        # Find any available sensor not already committed
        used_sensors = {p.sensor_id for p in self._all_proposals}
        free_sensor = next((s for s in self._available_sensors() if s.id not in used_sensors), None)
        if free_sensor is None:
            # Use the sensor from the first involved agent's proposal
            involved = conflict.agents_involved
            override_proposal = next(
                (p for p in self._all_proposals if p.agent_id in involved and p.target_id == target_id),
                None
            )
            if override_proposal:
                return Proposal(
                    sensor_id=override_proposal.sensor_id,
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
