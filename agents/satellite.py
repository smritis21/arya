from typing import List
from agents.base_agent import BaseAgent
from env.models import AgentObservation, Proposal


class SatelliteAgent(BaseAgent):
    """
    Satellite agent specialization:
    - Exclusively claims satellite-type sensors
    - Strongly prefers priority-3 (HIGH) targets
    - Among equal-priority targets, prefers those with higher sequential index
      (proxy for "further away / global coverage" since targets lack geo-coords)
    - Ignores priority-1 (LOW) targets entirely — not worth satellite bandwidth
    - Confidence scales with priority: P3=0.95, P2=0.70
    """

    def __init__(self):
        super().__init__("satellite", "satellite", "satellite")

    def propose(self) -> List[Proposal]:
        sensors = [s for s in self._available_sensors() if s.type == "satellite"]

        # Priority ordering: P3 first, then P2. Skip P1.
        targets_p3 = sorted(
            [t for t in self._active_targets() if t.priority == 3],
            key=lambda t: self._global_index(t.id), reverse=True  # prefer high-step / distant
        )
        targets_p2 = sorted(
            [t for t in self._active_targets() if t.priority == 2],
            key=lambda t: self._global_index(t.id), reverse=True
        )
        priority_queue = targets_p3 + targets_p2

        # Avoid re-covering targets we already covered this episode (if any idle)
        recently_covered = {
            a["target_id"]
            for a in self.recent_assignments[-3:]
            if isinstance(a, dict) and "target_id" in a
        }
        # Prefer uncovered; fall back to covered if nothing else available
        uncovered = [t for t in priority_queue if t.id not in recently_covered]
        final_queue = uncovered if uncovered else priority_queue

        proposals = []
        used_targets: set = set()
        for sensor in sensors:
            for target in final_queue:
                if target.id not in used_targets:
                    confidence = 0.95 if target.priority == 3 else 0.70
                    proposals.append(Proposal(
                        sensor_id=sensor.id,
                        target_id=target.id,
                        agent_id=self.agent_id,
                        priority_estimate=target.priority,
                        confidence=confidence,
                    ))
                    used_targets.add(target.id)
                    break

        self.recent_assignments.append({
            "step": getattr(self._obs, "timestep", -1),
            "proposals": len(proposals),
            "target_ids": [p.target_id for p in proposals],
        })
        return proposals

    def _global_index(self, target_id: str) -> int:
        """
        Extract the sequential index from target_id (e.g. T3_2 → 2).
        Higher index = later in step sequence = proxy for more global/distant target.
        """
        try:
            return int(target_id.split("_")[-1])
        except (IndexError, ValueError):
            return 0
