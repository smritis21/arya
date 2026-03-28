from typing import List, Optional, Tuple, Dict, Any, Union
from env.models import Sensor, Target, Observation, Action
from env.dynamics import initialize_sensors, spawn_targets, update_targets
from env.reward import compute_reward
from pydantic import ValidationError


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
        return Observation(
            sensors=self.sensors,
            targets=self.targets,
            timestep=self.current_step
        )

    def step(self, action: Union[Action, dict, None]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        # --- FIXED: Accept dict or Action; convert and validate before processing ---
        action = self._coerce_action(action)
        # Validate Action object against current env state
        valid_action = self._validate_action(action)

        # Compute reward BEFORE deactivating the target
        reward = compute_reward(valid_action, self.targets, self.sensors)

        if valid_action:
            for s in self.sensors:
                if s.id == valid_action.sensor_id:
                    s.available = False
            for t in self.targets:
                if t.id == valid_action.target_id:
                    t.active = False
            self._assignments.append({
                "sensor": valid_action.sensor_id,
                "target": valid_action.target_id
            })

        # Track missed high-priority targets (after assignment)
        missed = [t.id for t in self.targets if t.active and t.priority == 3]
        self._missed.extend(missed)

        # Update environment state
        self.targets = update_targets(self.targets)
        self.current_step += 1

        # Spawn new targets and reset sensor availability each step
        new_targets = spawn_targets(step=self.current_step, seed=self.seed)
        self.targets.extend(new_targets)
        for s in self.sensors:
            s.available = True  # sensors reset each step

        done = self.current_step >= self.max_steps or not any(t.active for t in self.targets)

        info = {
            "assignments": list(self._assignments),
            "missed_targets": list(self._missed),
            "step_count": self.current_step
        }

        return self.state(), reward, done, info

    def _coerce_action(self, action: Union[Action, dict, None]) -> Optional[Action]:
        """Convert dict → Action model. Returns None on failure."""
        if action is None:
            return None
        if isinstance(action, Action):
            return action
        # --- FIXED: dict input converted to typed Action model here ---
        if isinstance(action, dict):
            try:
                return Action(**action)
            except (ValidationError, TypeError):
                # Invalid dict shape — treat as no-op with penalty
                return None
        # Unexpected type — discard safely
        return None

    def _validate_action(self, action: Optional[Action]) -> Optional[Action]:
        """Validate a typed Action against live env state. Never accepts raw dict."""
        if action is None:
            return None
        # action is guaranteed to be an Action instance at this point
        sensor_ids = {s.id for s in self.sensors if s.available}
        target_ids = {t.id for t in self.targets if t.active}
        if action.sensor_id in sensor_ids and action.target_id in target_ids:
            return action
        return None
