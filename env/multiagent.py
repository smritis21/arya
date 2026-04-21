"""
Arya-X Multi-Agent Environment
Agents: satellite, drone, radar, command
Each agent proposes assignments; ConflictResolver arbitrates; NegotiationLayer coordinates.
"""
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from env.models import Sensor, Target, Action
from env.dynamics import initialize_sensors, spawn_targets

# ── Agent types ───────────────────────────────────────────────────────────────
AGENT_TYPES = ["satellite", "drone", "radar", "command"]

PRIORITY_REWARD = {3: 2.0, 2: 1.0, 1: 0.5}
IDLE_PENALTY    = -2.0
CONFLICT_PENALTY = -0.5


@dataclass
class AgentObservation:
    agent_id: str
    sensors: List[dict]
    targets: List[dict]
    timestep: int
    agent_type: str

    def to_dict(self) -> dict:
        return {
            "agent_id":   self.agent_id,
            "agent_type": self.agent_type,
            "sensors":    self.sensors,
            "targets":    self.targets,
            "timestep":   self.timestep,
        }


@dataclass
class Proposal:
    agent_id: str
    sensor_id: str
    target_id: str


# ── Conflict types ────────────────────────────────────────────────────────────
REDUNDANT_COVERAGE  = "REDUNDANT_COVERAGE"   # two agents assign same target
OVER_ASSIGNMENT     = "OVER_ASSIGNMENT"       # same sensor assigned by two agents
MISSED_PRIORITY_3   = "MISSED_PRIORITY_3"    # high-priority target left uncovered
FORCED_ARBITRATION  = "FORCED_ARBITRATION"   # command agent overrides conflict


@dataclass
class Conflict:
    type: str
    agents: List[str]
    sensor_id: Optional[str] = None
    target_id: Optional[str] = None


class ConflictDetector:
    def detect(self, proposals: List[Proposal], active_targets: List[Target]) -> List[Conflict]:
        conflicts: List[Conflict] = []
        sensor_map: Dict[str, List[str]] = {}   # sensor_id → [agent_ids]
        target_map: Dict[str, List[str]] = {}   # target_id → [agent_ids]

        for p in proposals:
            sensor_map.setdefault(p.sensor_id, []).append(p.agent_id)
            target_map.setdefault(p.target_id, []).append(p.agent_id)

        for sid, agents in sensor_map.items():
            if len(agents) > 1:
                conflicts.append(Conflict(OVER_ASSIGNMENT, agents, sensor_id=sid))

        for tid, agents in target_map.items():
            if len(agents) > 1:
                conflicts.append(Conflict(REDUNDANT_COVERAGE, agents, target_id=tid))

        covered = {p.target_id for p in proposals}
        for t in active_targets:
            if t.priority == 3 and t.id not in covered:
                conflicts.append(Conflict(MISSED_PRIORITY_3, [], target_id=t.id))

        return conflicts


class ConflictResolver:
    """3-pass resolution: Priority → Capability → Command Override."""

    def resolve(
        self,
        proposals: List[Proposal],
        conflicts: List[Conflict],
        sensors: List[Sensor],
        targets: List[Target],
    ) -> List[Proposal]:
        if not conflicts:
            return proposals

        # Pass 1 — Priority: keep proposal targeting highest-priority target
        target_priority = {t.id: t.priority for t in targets}
        resolved: Dict[str, Proposal] = {}   # sensor_id → winning proposal

        for p in proposals:
            if p.sensor_id not in resolved:
                resolved[p.sensor_id] = p
            else:
                existing = resolved[p.sensor_id]
                if target_priority.get(p.target_id, 0) > target_priority.get(existing.target_id, 0):
                    resolved[p.sensor_id] = p

        # Pass 2 — Capability: prefer sensor type matching target priority
        sensor_type = {s.id: s.type for s in sensors}
        capability_order = {"satellite": 3, "drone": 2, "radar": 1}
        used_targets: set = set()
        final: List[Proposal] = []

        for p in sorted(resolved.values(),
                        key=lambda x: (
                            -target_priority.get(x.target_id, 0),
                            -capability_order.get(sensor_type.get(x.sensor_id, "radar"), 1)
                        )):
            if p.target_id not in used_targets:
                final.append(p)
                used_targets.add(p.target_id)

        # Pass 3 — Command Override: if command agent proposed something, honour it
        command_proposals = [p for p in proposals if p.agent_id == "command"]
        for cp in command_proposals:
            already_covered = any(p.target_id == cp.target_id for p in final)
            if not already_covered:
                # Replace any conflicting sensor assignment
                final = [p for p in final if p.sensor_id != cp.sensor_id]
                final.append(cp)

        return final


class NegotiationLayer:
    """Tracks conflict_rate and coordinates multi-agent proposals."""

    def __init__(self):
        self.conflict_history: List[List[Conflict]] = []
        self.conflict_rate: float = 0.0

    def negotiate(
        self,
        proposals: List[Proposal],
        sensors: List[Sensor],
        targets: List[Target],
    ) -> Tuple[List[Proposal], List[Conflict]]:
        detector = ConflictDetector()
        resolver = ConflictResolver()

        active_targets = [t for t in targets if t.active]
        conflicts = detector.detect(proposals, active_targets)
        resolved  = resolver.resolve(proposals, conflicts, sensors, active_targets)

        self.conflict_history.append(conflicts)
        total_steps = len(self.conflict_history)
        steps_with_conflict = sum(1 for c in self.conflict_history if c)
        self.conflict_rate = steps_with_conflict / total_steps if total_steps else 0.0

        return resolved, conflicts

    def reset(self):
        self.conflict_history = []
        self.conflict_rate = 0.0


# ── Multi-agent environment ───────────────────────────────────────────────────
class AryaXEnv:
    def __init__(self, max_steps: int = 10, seed: int = 42):
        self.max_steps   = max_steps
        self.seed        = seed
        self.sensors:    List[Sensor] = []
        self.targets:    List[Target] = []
        self.current_step: int = 0
        self._assignments: List[dict] = []
        self._missed:      List[str]  = []
        self.initial_sensor_count: int = 0
        self.negotiation = NegotiationLayer()
        self._agent_rewards: Dict[str, float] = {a: 0.0 for a in AGENT_TYPES}

    def reset(self) -> Dict[str, AgentObservation]:
        self.sensors = initialize_sensors(self.seed)
        self.targets = spawn_targets(step=0, seed=self.seed)
        self.current_step = 0
        self._assignments = []
        self._missed = []
        self.initial_sensor_count = len(self.sensors)
        self._agent_rewards = {a: 0.0 for a in AGENT_TYPES}
        self.negotiation.reset()
        return self._build_observations()

    def state(self) -> Dict[str, AgentObservation]:
        return self._build_observations()

    def _build_observations(self) -> Dict[str, AgentObservation]:
        sensors_dict = [s.model_dump() for s in self.sensors]
        targets_dict = [t.model_dump() for t in self.targets]
        return {
            agent: AgentObservation(
                agent_id=agent,
                agent_type=agent,
                sensors=sensors_dict,
                targets=targets_dict,
                timestep=self.current_step,
            )
            for agent in AGENT_TYPES
        }

    def step_multiagent(
        self,
        proposals: List[Proposal],
    ) -> Tuple[Dict[str, AgentObservation], Dict[str, float], bool, dict]:
        resolved, conflicts = self.negotiation.negotiate(
            proposals, self.sensors, self.targets
        )

        handled_ids:      set = set()
        used_sensor_ids:  set = set()
        agent_assignments: Dict[str, List[dict]] = {a: [] for a in AGENT_TYPES}

        for p in resolved:
            if p.sensor_id in used_sensor_ids or p.target_id in handled_ids:
                continue
            sensor_ok = any(s.id == p.sensor_id and s.available for s in self.sensors)
            target_ok = any(t.id == p.target_id and t.active for t in self.targets)
            if not sensor_ok or not target_ok:
                continue
            for s in self.sensors:
                if s.id == p.sensor_id:
                    s.available = False
            for t in self.targets:
                if t.id == p.target_id:
                    t.active = False
            handled_ids.add(p.target_id)
            used_sensor_ids.add(p.sensor_id)
            self._assignments.append({"sensor": p.sensor_id, "target": p.target_id, "agent": p.agent_id})
            agent_assignments[p.agent_id].append({"sensor_id": p.sensor_id, "target_id": p.target_id})

        # Per-agent rewards
        step_rewards: Dict[str, float] = {a: 0.0 for a in AGENT_TYPES}
        for p in resolved:
            if p.target_id in handled_ids:
                t_obj = next((t for t in self.targets if t.id == p.target_id), None)
                if t_obj:
                    step_rewards[p.agent_id] += PRIORITY_REWARD.get(t_obj.priority, 0.0)

        # Idle penalty for agents that proposed nothing useful
        for agent in AGENT_TYPES:
            if not agent_assignments[agent]:
                step_rewards[agent] += IDLE_PENALTY

        # Conflict penalty
        for c in conflicts:
            if c.type in (OVER_ASSIGNMENT, REDUNDANT_COVERAGE):
                for aid in c.agents:
                    step_rewards[aid] += CONFLICT_PENALTY

        for a in AGENT_TYPES:
            self._agent_rewards[a] += step_rewards[a]

        self._missed.extend(
            t.id for t in self.targets
            if t.active and t.priority == 3 and t.id not in handled_ids
        )

        self.current_step += 1
        self.targets = spawn_targets(step=self.current_step, seed=self.seed)
        for s in self.sensors:
            s.available = True

        done = self.current_step >= self.max_steps
        info = {
            "assignments":     list(self._assignments),
            "missed_targets":  list(self._missed),
            "step_count":      self.current_step,
            "conflicts":       [{"type": c.type, "agents": c.agents,
                                 "sensor_id": c.sensor_id, "target_id": c.target_id}
                                for c in conflicts],
            "conflict_rate":   round(self.negotiation.conflict_rate, 4),
            "agent_rewards":   dict(self._agent_rewards),
            "step_rewards":    dict(step_rewards),
        }
        return self._build_observations(), step_rewards, done, info
