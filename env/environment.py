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

