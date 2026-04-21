import random
from typing import List, Optional, Tuple, Dict, Any, Union
from env.models import Sensor, Target, Observation, Action, AgentObservation
from env.dynamics import initialize_sensors, spawn_targets, spawn_targets_stochastic, apply_correlated_failures
from env.world_model import add_observation_noise, apply_mask, get_priority_mapping
from pydantic import ValidationError

PRIORITY_REWARD = {3: 2.0, 2: 1.0, 1: 0.5}
MISSED_HIGH_PENALTY = -2.0
IDLE_PENALTY = -2.0


class SentinelEnv:
    def __init__(self, max_steps: int = 10, seed: int = 42, config: Optional[Dict] = None):
        self.max_steps = max_steps
        self.seed = seed
        self.config = config
        self.sensors: List[Sensor] = []
        self.targets: List[Target] = []
        self.current_step: int = 0
        self._assignments: List[Dict] = []
        self._missed: List[str] = []
        self.initial_sensor_count: int = 0
        self._sensor_failure_prob: float = float(config.get("sensor_failure_prob", 0.0)) if config else 0.0

    def reset(self) -> Observation:
        if self.config and "sensors" in self.config:
            self.sensors = [
                Sensor(id=f"S{s['id']+1}", type=s.get("type", "radar"),
                       range=float(s.get("range", 200)), available=s.get("available", True))
                for s in self.config["sensors"]
            ]
        else:
            self.sensors = initialize_sensors(self.seed)
        if self.config and "targets" in self.config:
            self.targets = [
                Target(id=f"T0_{t['id']+1}", priority=min(t["priority"], 3), active=True)
                for t in self.config["targets"]
            ]
        else:
            self.targets = spawn_targets(step=0, seed=self.seed)
        self.current_step = 0
        self._assignments = []
        self._missed = []
        self.initial_sensor_count = len(self.sensors)
        return self.state()

    def state(self) -> Observation:
        return Observation(sensors=self.sensors, targets=self.targets, timestep=self.current_step)

    def step(self, action: Union[Action, dict, None]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        return self.step_batch([action] if action is not None else [])

    def step_batch(self, actions: List[Union[Action, dict, None]]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        coerced = [self._coerce_action(a) for a in actions]
        valid_actions = [a for a in (self._validate_action(a) for a in coerced) if a is not None]

        handled_ids = set()
        used_sensor_ids = set()
        for action in valid_actions:
            if action.sensor_id in used_sensor_ids or action.target_id in handled_ids:
                continue
            for s in self.sensors:
                if s.id == action.sensor_id:
                    s.available = False
            for t in self.targets:
                if t.id == action.target_id:
                    t.active = False
            handled_ids.add(action.target_id)
            used_sensor_ids.add(action.sensor_id)
            self._assignments.append({"sensor": action.sensor_id, "target": action.target_id})

        if not handled_ids:
            reward = IDLE_PENALTY
        else:
            reward = sum(PRIORITY_REWARD.get(t.priority, 0.0) for t in self.targets if t.id in handled_ids)
            unhandled_high = [t for t in self.targets if t.active and t.priority == 3]
            idle_sensors = len(self.sensors) - len(handled_ids)
            reward += min(len(unhandled_high), idle_sensors) * MISSED_HIGH_PENALTY

        self._missed.extend(t.id for t in self.targets if t.active and t.priority == 3 and t.id not in handled_ids)

        self.current_step += 1
        if self.config and "targets" in self.config:
            self.targets = [
                Target(id=f"T{self.current_step}_{t['id']+1}", priority=min(t["priority"], 3), active=True)
                for t in self.config["targets"]
            ]
        else:
            self.targets = spawn_targets(step=self.current_step, seed=self.seed)
        for s in self.sensors:
            s.available = True
        if self._sensor_failure_prob > 0.0:
            for s in self.sensors:
                if random.random() < self._sensor_failure_prob:
                    s.available = False

        done = self.current_step >= self.max_steps
        info = {"assignments": list(self._assignments), "missed_targets": list(self._missed), "step_count": self.current_step}
        return self.state(), reward, done, info

    def _coerce_action(self, action) -> Optional[Action]:
        if action is None:
            return None
        if isinstance(action, Action):
            return action
        if isinstance(action, dict):
            try:
                return Action(**action)
            except (ValidationError, TypeError):
                return None
        return None

    def _validate_action(self, action: Optional[Action]) -> Optional[Action]:
        if action is None:
            return None
        sensor_ids = {s.id for s in self.sensors if s.available}
        target_ids = {t.id for t in self.targets if t.active}
        if action.sensor_id in sensor_ids and action.target_id in target_ids:
            return action
        return None


AGENT_TYPES = ["satellite", "drone", "radar", "command"]
PRIORITY_REWARD = {3: 2.0, 2: 1.0, 1: 0.5}
MISSED_HIGH_PENALTY = -2.0
IDLE_PENALTY = -2.0


class AryaXEnv:
    def __init__(self, max_steps: int = 20, seed: int = 42, density_factor: float = 2.5,
                 failure_prob: float = 0.1, num_agents: int = 4):
        self.max_steps = max_steps
        self.seed = seed
        self.density_factor = density_factor
        self.failure_prob = failure_prob
        self.episode_number: int = 0
        self.current_step: int = 0
        self.sensors: List[Sensor] = []
        self.targets: List[Target] = []
        self._step_history: List[dict] = []
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> Dict[str, AgentObservation]:
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self.sensors = initialize_sensors(self.seed)
        self.sensors = apply_correlated_failures(self.sensors, self._rng.randint(0, 9999), self.failure_prob)
        self.targets = spawn_targets_stochastic(0, self.seed, self.density_factor)
        self.current_step = 0
        self._step_history = []
        self.episode_number += 1
        return self._make_agent_observations()

    def state(self) -> Dict[str, AgentObservation]:
        return self._make_agent_observations()

    def step_multiagent(self, final_assignments: List[Action]) -> Tuple[Dict[str, AgentObservation], Dict[str, float], bool, dict]:
        """Takes resolved assignments from negotiation layer, returns per-agent observations and rewards."""
        coerced = [a if isinstance(a, Action) else Action(**a) for a in final_assignments if a is not None]
        valid = [a for a in (self._validate_action_xa(a) for a in coerced) if a is not None]

        handled_ids: set = set()
        used_sensor_ids: set = set()
        sensor_to_agent: Dict[str, str] = self._map_sensors_to_agents()

        per_agent_reward: Dict[str, float] = {a: 0.0 for a in AGENT_TYPES}
        assignments_log = []

        for action in valid:
            if action.sensor_id in used_sensor_ids or action.target_id in handled_ids:
                continue
            for s in self.sensors:
                if s.id == action.sensor_id:
                    s.available = False
            for t in self.targets:
                if t.id == action.target_id:
                    t.active = False
            handled_ids.add(action.target_id)
            used_sensor_ids.add(action.sensor_id)
            assignments_log.append({"sensor": action.sensor_id, "target": action.target_id})
            agent = sensor_to_agent.get(action.sensor_id, "command")
            target_priority = next((t.priority for t in self.targets if t.id == action.target_id), 1)
            per_agent_reward[agent] += PRIORITY_REWARD.get(target_priority, 0.0)

        if not handled_ids:
            for a in AGENT_TYPES:
                per_agent_reward[a] = IDLE_PENALTY / len(AGENT_TYPES)
        else:
            unhandled_high = [t for t in self.targets if t.active and t.priority == 3]
            idle_sensors = len([s for s in self.sensors if s.available])
            penalty = min(len(unhandled_high), idle_sensors) * MISSED_HIGH_PENALTY
            if penalty < 0:
                per_agent_reward["command"] += penalty

        step_entry = {
            "step": self.current_step,
            "assignments": assignments_log,
            "handled": list(handled_ids),
            "targets": [t.dict() for t in self.targets],
            "per_agent_reward": dict(per_agent_reward),
        }
        self._step_history.append(step_entry)

        self.current_step += 1
        self.targets = spawn_targets_stochastic(self.current_step, self.seed, self.density_factor)
        for s in self.sensors:
            s.available = True
        self.sensors = apply_correlated_failures(self.sensors, self._rng.randint(0, 9999), self.failure_prob)

        done = self.current_step >= self.max_steps
        info = {"step_history_entry": step_entry, "assignments": assignments_log}
        return self._make_agent_observations(), per_agent_reward, done, info

    def get_step_history(self) -> List[dict]:
        return list(self._step_history)

    def _make_agent_observations(self) -> Dict[str, AgentObservation]:
        priority_map = get_priority_mapping(self.episode_number)
        obs = {}
        for agent_type in AGENT_TYPES:
            agent_sensors = [s for s in self.sensors if s.type == agent_type or agent_type == "command"]
            masked_targets = apply_mask(self.targets, agent_type, agent_sensors)
            rng = random.Random(self.seed + self.current_step + hash(agent_type))
            noisy_targets = add_observation_noise(masked_targets, agent_type, rng)
            # Apply schema drift to noisy targets
            drifted = [Target(id=t.id, priority=priority_map.get(t.priority, t.priority), active=t.active)
                       for t in noisy_targets]
            obs[agent_type] = AgentObservation(
                agent_id=agent_type,
                agent_type=agent_type,
                sensors=agent_sensors,
                targets=drifted,
                timestep=self.current_step,
            )
        return obs

    def _map_sensors_to_agents(self) -> Dict[str, str]:
        return {s.id: s.type for s in self.sensors}

    def _validate_action_xa(self, action: Action) -> Optional[Action]:
        sensor_ids = {s.id for s in self.sensors if s.available}
        target_ids = {t.id for t in self.targets if t.active}
        if action.sensor_id in sensor_ids and action.target_id in target_ids:
            return action
        return None
