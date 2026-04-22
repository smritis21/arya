"""
Arya-X Multi-Agent Environment
Agents: satellite, drone, radar, command
Each agent proposes assignments; ConflictResolver arbitrates; NegotiationLayer coordinates.
Uses canonical interaction/ package for conflict detection, resolution, and negotiation.
"""
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from env.models import Sensor, Target, Action
from env.dynamics import initialize_sensors, spawn_targets
from interaction import NegotiationLayer
from interaction.reward import RewardEngine

# ── Agent types ───────────────────────────────────────────────────────────────
AGENT_TYPES = ["satellite", "drone", "radar", "command"]

PRIORITY_REWARD  = {3: 2.0, 2: 1.0, 1: 0.5}
IDLE_PENALTY     = -2.0
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


def _cmd_override(tied: list[dict]) -> dict:
    """Command agent override: pick proposal with highest capability_score."""
    return max(tied, key=lambda p: p.get("capability_score", 0.0))


# ── Multi-agent environment ───────────────────────────────────────────────────
class AryaXEnv:
    def __init__(self, max_steps: int = 10, seed: int = 42,
                 density_factor: float = 1.0, failure_prob: float = 0.0,
                 conflict_injection: bool = False):
        self.max_steps        = max_steps
        self.seed             = seed
        self.density_factor   = density_factor
        self.failure_prob     = failure_prob
        self.conflict_injection = conflict_injection
        self.sensors:    List[Sensor] = []
        self.targets:    List[Target] = []
        self.current_step: int = 0
        self._assignments: List[dict] = []
        self._missed:      List[str]  = []
        self.initial_sensor_count: int = 0
        self.negotiation    = NegotiationLayer()
        self.reward_engine  = RewardEngine()
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
        self.reward_engine.reset()
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
        # Build world_state for interaction/ pipeline
        sensors_dict  = [s.model_dump() for s in self.sensors]
        targets_dict  = [t.model_dump() for t in self.targets]
        proposals_raw = [
            {"agent_id": p.agent_id, "sensor_id": p.sensor_id,
             "target_id": p.target_id, "capability_score": 0.5}
            for p in proposals
        ]
        assigned_sids = {p.sensor_id for p in proposals}
        idle_sensors  = [s["id"] for s in sensors_dict if s["id"] not in assigned_sids]
        world_state = {
            "sensors":      sensors_dict,
            "targets":      targets_dict,
            "idle_sensors": idle_sensors,
            "step":         self.current_step,
            "proposals":    proposals_raw,
        }

        # Run full negotiation pipeline (interaction/)
        neg_result = self.negotiation.negotiate(proposals_raw, world_state, _cmd_override)
        final_assignments = neg_result.final_assignments  # list[{sensor_id, target_id}]
        conflicts = neg_result.conflicts_detected

        # Build agent_id lookup from proposals
        sensor_to_agent = {p.sensor_id: p.agent_id for p in proposals}

        handled_ids:     set = set()
        used_sensor_ids: set = set()
        agent_assignments: Dict[str, List[dict]] = {a: [] for a in AGENT_TYPES}

        for a in final_assignments:
            sid, tid = a["sensor_id"], a["target_id"]
            if sid in used_sensor_ids or tid in handled_ids:
                continue
            sensor_ok = any(s.id == sid and s.available for s in self.sensors)
            target_ok = any(t.id == tid and t.active for t in self.targets)
            if not sensor_ok or not target_ok:
                continue
            for s in self.sensors:
                if s.id == sid:
                    s.available = False
            for t in self.targets:
                if t.id == tid:
                    t.active = False
            handled_ids.add(tid)
            used_sensor_ids.add(sid)
            agent_id = sensor_to_agent.get(sid, "command")
            self._assignments.append({"sensor": sid, "target": tid, "agent": agent_id})
            agent_assignments[agent_id].append({"sensor_id": sid, "target_id": tid})

        # Per-agent rewards via RewardEngine
        committed = [{"sensor_id": sid, "target_id": tid}
                     for sid, tid in ((a["sensor_id"], a["target_id"]) for a in final_assignments)
                     if tid in handled_ids]
        # Rebuild world_state with committed proposals for RewardEngine
        world_state["proposals"] = [
            {"agent_id": sensor_to_agent.get(p["sensor_id"], "command"),
             "sensor_id": p["sensor_id"], "target_id": p["target_id"]}
            for p in proposals_raw
        ]
        step_rewards = self.reward_engine.compute_step_reward(
            committed, neg_result, world_state, AGENT_TYPES
        )

        for a in AGENT_TYPES:
            self._agent_rewards[a] += step_rewards.get(a, 0.0)

        self._missed.extend(
            t.id for t in self.targets
            if t.active and t.priority == 3 and t.id not in handled_ids
        )

        self.current_step += 1
        self.targets = spawn_targets(step=self.current_step, seed=self.seed,
                                      conflict_injection=self.conflict_injection)
        for s in self.sensors:
            s.available = True

        done = self.current_step >= self.max_steps
        conflict_rate = self.negotiation.get_conflict_rate()
        info = {
            "assignments":    list(self._assignments),
            "missed_targets": list(self._missed),
            "step_count":     self.current_step,
            "conflicts": [
                {
                    "type":      str(c.conflict_type),
                    "agents":    c.involved_agents,
                    "sensor_id": c.involved_sensors[0] if c.involved_sensors else None,
                    "target_id": c.involved_targets[0] if c.involved_targets else None,
                }
                for c in conflicts
            ],
            "conflict_rate":  round(conflict_rate, 4),
            "agent_rewards":  dict(self._agent_rewards),
            "step_rewards":   dict(step_rewards),
        }
        return self._build_observations(), step_rewards, done, info
