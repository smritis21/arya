from typing import List, Set
from agents.base_agent import BaseAgent
from env.models import AgentObservation, Proposal


class RadarAgent(BaseAgent):
    def __init__(self):
        super().__init__("radar", "radar", "radar")
        self._persistent_targets: dict = {}  # target_id -> consecutive steps covered

    def propose(self) -> List[Proposal]:
        sensors = [s for s in self._available_sensors() if s.type == "radar"]
        targets = self._active_targets()
        timestep = getattr(self._obs, "timestep", 0)

        # Update persistence tracking — drop targets no longer active
        active_ids = {t.id for t in targets}
        self._persistent_targets = {k: v for k, v in self._persistent_targets.items() if k in active_ids}

        airspace = sorted([t for t in targets if self._is_airspace(t.id)], key=lambda t: t.priority, reverse=True)
        non_airspace = sorted([t for t in targets if not self._is_airspace(t.id)], key=lambda t: t.priority, reverse=True)

        # Grid persistence: always keep a sensor on airspace targets covered 2+ consecutive steps
        persistent = [t for t in airspace if self._persistent_targets.get(t.id, 0) >= 2]
        priority_queue = persistent + [t for t in airspace if t not in persistent] + non_airspace

        proposals = []
        used_targets: Set[str] = set()
        for sensor, target in zip(sensors, priority_queue):
            if target.id in used_targets:
                continue
            is_airspace = self._is_airspace(target.id)
            confidence = 0.85 if is_airspace else 0.5
            proposals.append(Proposal(
                sensor_id=sensor.id,
                target_id=target.id,
                agent_id=self.agent_id,
                priority_estimate=target.priority,
                confidence=confidence,
            ))
            used_targets.add(target.id)
            self._persistent_targets[target.id] = self._persistent_targets.get(target.id, 0) + 1

        self.recent_assignments.append({"step": timestep, "proposals": len(proposals)})
        return proposals

    def _is_airspace(self, target_id: str) -> bool:
        """Airspace targets: odd sequential index."""
        try:
            idx = int(target_id.split("_")[1])
            return idx % 2 == 1
        except (IndexError, ValueError):
            return True
