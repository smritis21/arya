from typing import List, Optional, Tuple, Dict, Any, Union
from env.models import Sensor, Target, Observation, Action
from env.dynamics import initialize_sensors, spawn_targets
from pydantic import ValidationError

PRIORITY_REWARD = {3: 2.0, 2: 1.0, 1: 0.5}
MISSED_HIGH_PENALTY = -2.0
IDLE_PENALTY = -2.0


class SentinelEnv:
    def __init__(self, max_steps: int = 10, seed: int = 42):
        self.max_steps = max_steps
        self.seed = seed
        self.sensors: List[Sensor] = []
        self.targets: List[Target] = []
        self.current_step: int = 0
        self._assignments: List[Dict] = []
        self._missed: List[str] = []

    def reset(self) -> Observation:
        self.sensors = initialize_sensors(self.seed)
        self.targets = spawn_targets(step=0, seed=self.seed)
        self.current_step = 0
        self._assignments = []
        self._missed = []
        return self.state()

    def state(self) -> Observation:
        return Observation(sensors=self.sensors, targets=self.targets, timestep=self.current_step)

    def step(self, action: Union[Action, dict, None]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Single-action step — wraps step_batch for backward compatibility."""
        return self.step_batch([action] if action is not None else [])

    def step_batch(self, actions: List[Union[Action, dict, None]]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Process all sensor assignments for one timestep at once.
        - Each action assigns one sensor to one target.
        - Reward computed once across all assignments.
        - Timestep advances exactly once.
        - Targets that weren't handled this step expire (no accumulation).
        """
        coerced = [self._coerce_action(a) for a in actions]
        valid_actions = [self._validate_action(a) for a in coerced]
        valid_actions = [a for a in valid_actions if a is not None]

        # Apply assignments — mark sensors busy and targets handled
        handled_ids = set()
        used_sensor_ids = set()
        for action in valid_actions:
            if action.sensor_id in used_sensor_ids or action.target_id in handled_ids:
                continue  # skip duplicate sensor/target in same batch
            for s in self.sensors:
                if s.id == action.sensor_id:
                    s.available = False
            for t in self.targets:
                if t.id == action.target_id:
                    t.active = False
            handled_ids.add(action.target_id)
            used_sensor_ids.add(action.sensor_id)
            self._assignments.append({"sensor": action.sensor_id, "target": action.target_id})

        # Compute reward for this timestep
        if not handled_ids:
            reward = IDLE_PENALTY
        else:
            reward = 0.0
            # Reward for each handled target
            for t in self.targets:
                if t.id in handled_ids:
                    reward += PRIORITY_REWARD.get(t.priority, 0.0)

            # Penalty only for HIGH threats that were skipped despite sensors being available
            # i.e. sensors were assigned to lower-priority targets while HIGH ones were ignored
            handled_priorities = [t.priority for t in self.targets if t.id in handled_ids]
            unhandled_high = [t for t in self.targets if t.active and t.priority == 3]
            sensors_used = len(handled_ids)
            total_sensors = len([s for s in self.sensors])
            # How many sensors were left idle (not assigned to anything)
            idle_sensors = total_sensors - sensors_used
            # Penalize only for unhandled HIGH threats that idle sensors could have covered
            avoidable_misses = min(len(unhandled_high), idle_sensors)
            reward += avoidable_misses * MISSED_HIGH_PENALTY

        # Track truly missed HIGH threats — those that expired unhandled
        missed_this_step = [
            t.id for t in self.targets
            if t.active and t.priority == 3 and t.id not in handled_ids
        ]
        self._missed.extend(missed_this_step)

        # Advance timestep — all remaining active targets expire
        self.current_step += 1
        new_targets = spawn_targets(step=self.current_step, seed=self.seed)
        self.targets = new_targets  # fresh targets only, no carryover
        for s in self.sensors:
            s.available = True

        done = self.current_step >= self.max_steps

        info = {
            "assignments": list(self._assignments),
            "missed_targets": list(self._missed),
            "step_count": self.current_step
        }
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
