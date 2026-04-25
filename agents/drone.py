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

        # Theory-of-mind: back off boundary targets after losing 3+ redundant_coverage
        radar_wins = self.losses.get("redundant_coverage", 0)
        retreat_boundary = radar_wins >= 3

        # Sort: closest (lowest index) first, then by priority descending
        targets_sorted = sorted(
            targets,
            key=lambda t: (self._proximity_score(t.id), -t.priority)
        )

        proposals = []
        used_targets: set = set()
        for sensor in sensors:
            for target in targets_sorted:
                if target.id in used_targets:
                    continue
                is_boundary = self._is_boundary(target.id)
                if retreat_boundary and is_boundary:
                    continue  # skip boundary targets when radar dominates
                confidence = 0.85
                if is_boundary:
                    confidence -= 0.2  # lower confidence on boundary anyway
                proposals.append(Proposal(
                    sensor_id=sensor.id,
                    target_id=target.id,
                    agent_id=self.agent_id,
                    priority_estimate=target.priority,
                    confidence=round(confidence, 3),
                ))
                used_targets.add(target.id)
                break

        # Fallback: if no P2+ targets available, cover the closest P1
        if not proposals and sensors:
            p1_targets = sorted(
                [t for t in self._active_targets() if t.priority == 1],
                key=lambda t: self._proximity_score(t.id)
            )
            for sensor in sensors:
                if p1_targets:
                    target = p1_targets[0]
                    proposals.append(Proposal(
                        sensor_id=sensor.id,
                        target_id=target.id,
                        agent_id=self.agent_id,
                        priority_estimate=target.priority,
                        confidence=0.50,
                    ))
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
