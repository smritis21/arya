from typing import List
from agents.base_agent import BaseAgent
from env.models import AgentObservation, Proposal


class SatelliteAgent(BaseAgent):
    def __init__(self):
        super().__init__("satellite", "satellite", "satellite")

    def propose(self) -> List[Proposal]:
        sensors = [s for s in self._available_sensors() if s.type == "satellite"]
        # Satellite ignores priority-1 targets — only proposes for priority-2 and priority-3
        targets = sorted(
            [t for t in self._active_targets() if t.priority >= 2],
            key=lambda t: t.priority, reverse=True
        )
        proposals = []
        for sensor, target in zip(sensors, targets):
            confidence = 0.9 if target.priority == 3 else 0.6
            proposals.append(Proposal(
                sensor_id=sensor.id,
                target_id=target.id,
                agent_id=self.agent_id,
                priority_estimate=target.priority,
                confidence=confidence,
            ))
        self.recent_assignments.append({"step": getattr(self._obs, "timestep", -1), "proposals": len(proposals)})
        return proposals
