from typing import List, Set
from agents.base_agent import BaseAgent
from env.models import AgentObservation, Proposal


class RadarAgent(BaseAgent):
    """
    Radar agent specialization:
    - Exclusively claims radar-type sensors
    - Prefers SPREAD coverage: avoids duplicate targeting, covers different targets
      than its own recent assignments (continuous / persistent airspace watch)
    - Priority ordering: P3 > P2 > P1, with airspace (odd index) preferred
    - Persistent targets: if a target has been covered 2+ consecutive steps, keep
      a sensor locked on it (continuous coverage logic)
    - Never proposes the same target twice in the same step
    - Confidence: 0.90 for airspace/persistent; 0.60 otherwise
    """

    def __init__(self):
        super().__init__("radar", "radar", "radar")
        # target_id → count of consecutive steps covered
        self._persistent_targets: dict = {}

    def propose(self) -> List[Proposal]:
        sensors = [s for s in self._available_sensors() if s.type == "radar"]
        targets = self._active_targets()
        timestep = getattr(self._obs, "timestep", 0)

        # Refresh persistence tracking — drop targets no longer active
        active_ids = {t.id for t in targets}
        self._persistent_targets = {
            k: v for k, v in self._persistent_targets.items() if k in active_ids
        }

        # Targets covered 2+ consecutive steps by radar → lock on them first
        persistent = [t for t in targets if self._persistent_targets.get(t.id, 0) >= 2]

        # Airspace (odd index) non-persistent
        airspace_fresh = [
            t for t in targets
            if self._is_airspace(t.id) and t not in persistent
        ]
        airspace_fresh.sort(key=lambda t: -t.priority)

        # Non-airspace
        non_airspace = [
            t for t in targets
            if not self._is_airspace(t.id) and t not in persistent
        ]
        non_airspace.sort(key=lambda t: -t.priority)

        # Priority queue: persistent → airspace fresh → non-airspace
        priority_queue = persistent + airspace_fresh + non_airspace

        # Exclude targets covered in the very last step (spread coverage)
        last_covered = set()
        if self.recent_assignments:
            last = self.recent_assignments[-1]
            if isinstance(last, dict):
                last_covered = set(last.get("target_ids", []))

        # Build proposals — never duplicate targets
        proposals = []
        used_targets: Set[str] = set()

        for sensor in sensors:
            for target in priority_queue:
                if target.id in used_targets:
                    continue
                # Prefer not repeating last step's targets (spread), but
                # fall back to them if no other option
                if target.id in last_covered and len(priority_queue) > len(sensors):
                    continue
                is_persistent = self._persistent_targets.get(target.id, 0) >= 2
                is_airspace = self._is_airspace(target.id)
                confidence = 0.90 if (is_persistent or is_airspace) else 0.60
                proposals.append(Proposal(
                    sensor_id=sensor.id,
                    target_id=target.id,
                    agent_id=self.agent_id,
                    priority_estimate=target.priority,
                    confidence=confidence,
                ))
                used_targets.add(target.id)
                self._persistent_targets[target.id] = (
                    self._persistent_targets.get(target.id, 0) + 1
                )
                break
            else:
                # Fallback: accept repeated target if queue is exhausted
                for target in priority_queue:
                    if target.id not in used_targets:
                        proposals.append(Proposal(
                            sensor_id=sensor.id,
                            target_id=target.id,
                            agent_id=self.agent_id,
                            priority_estimate=target.priority,
                            confidence=0.50,
                        ))
                        used_targets.add(target.id)
                        self._persistent_targets[target.id] = (
                            self._persistent_targets.get(target.id, 0) + 1
                        )
                        break

        self.recent_assignments.append({
            "step": timestep,
            "proposals": len(proposals),
            "target_ids": [p.target_id for p in proposals],
        })
        return proposals

    def _is_airspace(self, target_id: str) -> bool:
        """Odd sequential index = airspace zone (radar's natural domain)."""
        try:
            return int(target_id.split("_")[-1]) % 2 == 1
        except (IndexError, ValueError):
            return True
