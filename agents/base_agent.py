from abc import ABC, abstractmethod
from typing import List
from env.models import AgentObservation, Proposal, ConflictRecord


class BaseAgent(ABC):
    def __init__(self, agent_id: str, agent_type: str, sensor_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.sensor_type = sensor_type
        self.conflict_history: List[ConflictRecord] = []  # last 10
        self.episode_reward: float = 0.0
        self.recent_assignments: List[dict] = []  # last 5 steps
        self._obs: AgentObservation = None
        # Theory-of-mind tracking
        self.wins: dict = {}   # conflict_type -> win count
        self.losses: dict = {} # conflict_type -> loss count

    def observe(self, obs: AgentObservation) -> None:
        self._obs = obs
        # Sync conflict history from observation (last 5 conflicts)
        for c in obs.conflict_history[-5:]:
            record = ConflictRecord(**c) if isinstance(c, dict) else c
            if record not in self.conflict_history:
                self.conflict_history.append(record)
        self.conflict_history = self.conflict_history[-10:]
        # Track recent assignments
        self.recent_assignments = self.recent_assignments[-4:]

    @abstractmethod
    def propose(self) -> List[Proposal]:
        """Return one Proposal per available sensor this agent controls."""

    def update(self, reward: float, conflict_history: List) -> None:
        """Accept ConflictRecord objects from interaction/conflict.py.
        Uses involved_agents (not agents_involved) and has no resolution field.
        """
        self.episode_reward += reward
        for record in conflict_history:
            # Support both interaction.conflict.ConflictRecord (dataclass) and
            # env.models.ConflictRecord (Pydantic) by reading whichever field exists.
            involved = (
                getattr(record, "involved_agents", None)
                or getattr(record, "agents_involved", [])
            )
            ctype = str(getattr(record, "conflict_type", "unknown"))
            if self.agent_id in involved:
                # Heuristic: first listed agent is the "winner" of the conflict
                if involved and involved[0] == self.agent_id:
                    self.wins[ctype] = self.wins.get(ctype, 0) + 1
                else:
                    self.losses[ctype] = self.losses.get(ctype, 0) + 1
        self.conflict_history.extend(conflict_history)
        self.conflict_history = self.conflict_history[-10:]

    def reset_episode(self) -> None:
        self.episode_reward = 0.0
        self.recent_assignments = []

    def _available_sensors(self):
        if self._obs is None:
            return []
        from env.models import Sensor
        return [Sensor(**s) if isinstance(s, dict) else s for s in self._obs.sensors if (s.get("available") if isinstance(s, dict) else s.available)]

    def _active_targets(self):
        if self._obs is None:
            return []
        from env.models import Target
        return [Target(**t) if isinstance(t, dict) else t for t in self._obs.targets if (t.get("active") if isinstance(t, dict) else t.active)]
