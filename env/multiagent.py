"""
Arya-X Multi-Agent Environment — unified serving + training env.
mode='multi': partial obs + noise (training). mode='single': full obs (server/inference).
"""
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from env.models import Sensor, Target, Action
from env.dynamics import initialize_sensors, spawn_targets, spawn_targets_stochastic, apply_correlated_failures
from env.world_model import add_observation_noise, apply_mask, get_priority_mapping
from interaction import NegotiationLayer
from interaction.reward import RewardEngine

# ── Agent types ───────────────────────────────────────────────────────────────
AGENT_TYPES = ["satellite", "drone", "radar", "command"]

PRIORITY_REWARD  = {3: 2.0, 2: 1.0, 1: 0.5}
IDLE_PENALTY     = -2.0
CONFLICT_PENALTY = -0.5

# Stable per-agent salts for partial obs noise
_AGENT_SALT = {"satellite": 1001, "drone": 2002, "radar": 3003, "command": 4004}


@dataclass
class AgentObservation:
    agent_id: str
    sensors: List[dict]
    targets: List[dict]
    timestep: int
    agent_type: str
    conflict_history: List[dict] = None

    def __post_init__(self):
        if self.conflict_history is None:
            self.conflict_history = []

    def to_dict(self) -> dict:
        return {
            "agent_id":   self.agent_id,
            "agent_type": self.agent_type,
            "sensors":    self.sensors,
            "targets":    self.targets,
            "timestep":   self.timestep,
            "conflict_history": self.conflict_history,
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
                 conflict_injection: bool = False,
                 min_targets: int = 0, max_targets: int = 0,
                 mode: str = 'multi'):
        self.max_steps        = max_steps
        self.seed             = seed
        self.density_factor   = density_factor
        self.failure_prob     = failure_prob
        self.conflict_injection = conflict_injection
        self.min_targets      = min_targets
        self.max_targets      = max_targets
        # mode='multi': partial obs + noise (training)
        # mode='single': full obs, no noise (server/inference backwards compat)
        self.mode             = mode
        self.episode_number   = 0
        self._rng             = random.Random(seed)
        self.sensors:    List[Sensor] = []
        self.targets:    List[Target] = []
        self.current_step: int = 0
        self._assignments: List[dict] = []
        self._missed:      List[str]  = []
        self.initial_sensor_count: int = 0
        self.negotiation    = NegotiationLayer()
        self.reward_engine  = RewardEngine()
        self._agent_rewards: Dict[str, float] = {a: 0.0 for a in AGENT_TYPES}

    def _clamp_targets(self, targets: List[Target]) -> List[Target]:
        """Enforce min/max target count from curriculum targets_per_step_range."""
        if self.max_targets > 0 and len(targets) > self.max_targets:
            targets = targets[:self.max_targets]
        if self.min_targets > 0 and len(targets) < self.min_targets:
            # Pad with copies of the highest-priority target
            base = targets[-1] if targets else Target(id="T_pad_1", priority=1, active=True)
            while len(targets) < self.min_targets:
                pad = Target(
                    id=f"{base.id}_pad{len(targets)}",
                    priority=base.priority,
                    active=True,
                    type=base.type,
                )
                targets.append(pad)
        return targets

    def reset(self) -> Dict[str, AgentObservation]:
        self._rng = random.Random(self.seed)
        # Scale sensor count with density: easy=3, medium=4, hard=5
        num_sensors = max(3, min(5, int(self.density_factor * 0.9) + 2))
        self.sensors = initialize_sensors(self.seed, num_sensors=num_sensors)
        if self.failure_prob > 0.0:
            self.sensors = apply_correlated_failures(self.sensors, self._rng.randint(0, 9999), self.failure_prob)
        self.targets = self._clamp_targets(
            spawn_targets_stochastic(step=0, seed=self.seed, density_factor=self.density_factor)
        )
        self.current_step = 0
        self._assignments = []
        self._missed = []
        self.initial_sensor_count = len(self.sensors)
        self._agent_rewards = {a: 0.0 for a in AGENT_TYPES}
        self.episode_number += 1
        self.negotiation.reset()
        self.reward_engine.reset()
        return self._build_observations()

    def state(self) -> Dict[str, AgentObservation]:
        return self._build_observations()

    def _build_observations(self) -> Dict[str, AgentObservation]:
        if self.mode == 'single':
            # Full obs — no masking or noise (server/inference backwards compat)
            sensors_dict = [s.model_dump() for s in self.sensors]
            targets_dict = [t.model_dump() for t in self.targets]
            return {
                agent: AgentObservation(
                    agent_id=agent, agent_type=agent,
                    sensors=sensors_dict, targets=targets_dict,
                    timestep=self.current_step,
                )
                for agent in AGENT_TYPES
            }
        # mode='multi': partial obs + noise per agent
        priority_map = get_priority_mapping(self.episode_number)
        obs = {}
        for agent in AGENT_TYPES:
            agent_sensors = [s for s in self.sensors if s.type == agent or agent == "command"]
            masked_targets = apply_mask(self.targets, agent, agent_sensors)
            rng = random.Random(self.seed + self.current_step + _AGENT_SALT.get(agent, 0))
            noisy_targets = add_observation_noise(masked_targets, agent, rng)
            drifted = [Target(id=t.id, priority=priority_map.get(t.priority, t.priority),
                              active=t.active, type=t.type) for t in noisy_targets]
            obs[agent] = AgentObservation(
                agent_id=agent, agent_type=agent,
                sensors=[s.model_dump() for s in agent_sensors],
                targets=[t.model_dump() for t in drifted],
                timestep=self.current_step,
            )
        return obs

    def step_multiagent(
        self,
        proposals: List[Proposal],
    ) -> Tuple[Dict[str, AgentObservation], Dict[str, float], bool, dict]:
        # Build world_state for interaction/ pipeline
        sensors_dict  = [s.model_dump() for s in self.sensors]
        targets_dict  = [t.model_dump() for t in self.targets]
        CAPABILITY_MATRIX = {
            ("satellite","strategic"):0.95, ("satellite","kinetic"):0.40, ("satellite","airspace"):0.60,
            ("drone","kinetic"):0.95, ("drone","strategic"):0.30, ("drone","airspace"):0.50,
            ("radar","airspace"):0.95, ("radar","kinetic"):0.65, ("radar","strategic"):0.45,
        }
        
        proposals_raw = []
        for p in proposals:
            sensor_type = next((getattr(s, "type", "unknown") for s in self.sensors if s.id == p.sensor_id), "unknown")
            target_type = next((getattr(t, "type", "strategic") for t in self.targets if t.id == p.target_id), "strategic")
            cap_score = CAPABILITY_MATRIX.get((sensor_type, target_type), 0.5)
            proposals_raw.append({
                "agent_id": p.agent_id, 
                "sensor_id": p.sensor_id,
                "target_id": p.target_id, 
                "capability_score": cap_score
            })
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
        self.targets = self._clamp_targets(spawn_targets_stochastic(
            step=self.current_step, seed=self.seed, density_factor=self.density_factor
        ))
        if self.conflict_injection:
            # Guarantee at least one P3 target to exercise the conflict pipeline
            from env.dynamics import spawn_targets as _spawn_det
            injected = _spawn_det(step=self.current_step, seed=self.seed, conflict_injection=True)
            p3_ids = {t.id for t in self.targets if t.priority == 3}
            for t in injected:
                if t.priority == 3 and t.id not in p3_ids:
                    self.targets.append(t)
                    break
        for s in self.sensors:
            s.available = True
        if self.failure_prob > 0.0:
            rng = random.Random(self.seed + self.current_step)
            self.sensors = apply_correlated_failures(self.sensors, rng.randint(0, 9999), self.failure_prob)

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
