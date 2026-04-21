from typing import List
from agents.base_agent import BaseAgent
from env.models import AgentObservation, Proposal


class DroneAgent(BaseAgent):
    def __init__(self):
        super().__init__("drone", "drone", "drone")

    def propose(self) -> List[Proposal]:
        sensors = [s for s in self._available_sensors() if s.type == "drone"]
        targets = sorted(self._active_targets(), key=lambda t: t.priority, reverse=True)

        # Theory-of-mind: if Radar won last 3 boundary-zone disputes, reduce confidence on boundary targets
        radar_wins = self.losses.get("redundant_coverage", 0)
        boundary_confidence_penalty = 0.5 if radar_wins >= 3 else 0.0

        proposals = []
        for sensor, target in zip(sensors, targets):
            is_boundary = self._is_boundary_target(target.id)
            base_confidence = 0.8
            if is_boundary:
                base_confidence = max(0.3, base_confidence - boundary_confidence_penalty)
            proposals.append(Proposal(
                sensor_id=sensor.id,
                target_id=target.id,
                agent_id=self.agent_id,
                priority_estimate=target.priority,
                confidence=base_confidence,
            ))
        self.recent_assignments.append({"step": getattr(self._obs, "timestep", -1), "proposals": len(proposals)})
        return proposals

    def _is_boundary_target(self, target_id: str) -> bool:
        """Boundary zone: targets with even sequential index (overlap with radar arc)."""
        try:
            idx = int(target_id.split("_")[1])
            return idx % 2 == 0
        except (IndexError, ValueError):
            return False
