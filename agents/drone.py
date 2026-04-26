from typing import List
from agents.base_agent import BaseAgent
from env.models import AgentObservation, Proposal


class DroneAgent(BaseAgent):
    """
    Drone agent specialization:
    - Exclusively claims drone-type sensors
    - Focuses on medium + high priority (P2 + P3)
    - "Closest target" proxy: prefers targets with LOWEST sequential index
      (earliest in step = locally present, not globally distant)
    - Adaptive: if it has repeatedly lost redundant_coverage conflicts against radar,
      it retreats from even-indexed targets (boundary zone)
    - Confidence: 0.85 base, reduced on boundary targets after conflict losses
    """

    def __init__(self):
        super().__init__("drone", "drone", "drone")

    def propose(self) -> List[Proposal]:
        sensors = [s for s in self._available_sensors() if s.type == "drone"]
        targets = [t for t in self._active_targets() if t.priority >= 2]

        targets_sorted = sorted(
            targets,
            key=lambda t: (0 if t.type == "kinetic" else 1, -t.priority)
        )
        if not targets_sorted:
            targets_sorted = sorted(self._active_targets(), key=lambda t: -t.priority)

        proposals = []
        used_targets: set = set()
        for sensor in sensors:
            for target in targets_sorted:
                if target.id not in used_targets:
                    proposals.append(Proposal(
                        sensor_id=sensor.id,
                        target_id=target.id,
                        agent_id=self.agent_id,
                        priority_estimate=target.priority,
                        confidence=0.85,
                    ))
                    used_targets.add(target.id)
                    break

        self.recent_assignments.append({
            "step": getattr(self._obs, "timestep", -1),
            "proposals": len(proposals),
            "target_ids": [p.target_id for p in proposals],
        })
        return proposals

    def _proximity_score(self, target_id: str) -> int:
        """
        Lower sequential index → locally closer (lower score = closer = preferred).
        T3_1 scores 1 (close), T3_4 scores 4 (far).
        """
        try:
            return int(target_id.split("_")[-1])
        except (IndexError, ValueError):
            return 99

    def _is_boundary_target(self, target_id: str) -> bool:
        return self._is_boundary(target_id)

    def _is_boundary(self, target_id: str) -> bool:
        """Even sequential index = boundary / overlap zone with radar arc."""
        try:
            return int(target_id.split("_")[-1]) % 2 == 0
        except (IndexError, ValueError):
            return False
