"""
ARYA-X — Curriculum Engine
Adaptive difficulty scaling based on coordination quality.
Implements Theme 4 (Self-Improvement): environment adapts to the agents.

Three phases:
  Phase 1 — SCAFFOLDING        (episodes   0–500)
  Phase 2 — COORDINATION PRESS (episodes 500–2000)
  Phase 3 — ADAPTIVE SELF-PLAY (episodes 2000+)
"""

import logging
import random
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Agent pool for self-play freezing (CMD is never frozen)
_SELF_PLAY_AGENTS = ["SAT", "UAV", "RDR"]

# Phase boundaries
_PHASE1_END = 500
_PHASE2_END = 2000

# Self-play trigger interval (episodes)
_SELF_PLAY_INTERVAL = 200

# Max difficulty delta per episode
_MAX_DIFFICULTY_DELTA = 0.1


class CurriculumEngine:
    """
    Manages curriculum difficulty across three phases and exposes
    scenario configs for SentinelEnv + NegotiationLayer.
    """

    def __init__(
        self,
        escalate_threshold: float = 0.72,
        regress_threshold:  float = 0.35,
        window_size:        int   = 50,
    ) -> None:
        self.escalate_threshold = escalate_threshold
        self.regress_threshold  = regress_threshold
        self.window_size        = window_size

        self.episode_count:    int   = 0
        self.current_phase:    int   = 1
        self.difficulty_level: float = 0.0     # continuous [0.0, 1.0]
        self._score_history:   list[float] = []
        self._eff_history:     list[float] = []

        # Metrics counters
        self._escalations_count:        int = 0
        self._regressions_count:        int = 0
        self._self_play_triggers_count: int = 0
        self._self_play_cycle:          int = 0  # tracks rotation index

    # ── Scenario config ──────────────────────────────────────────────
    def get_scenario_config(self) -> dict:
        """
        Returns deterministic config given current phase and difficulty_level.
        """
        d = self.difficulty_level  # [0.0, 1.0]

        if self.current_phase == 1:
            # Scaffolding: simple, no failures, no injection
            return {
                "max_steps":              20,
                "targets_per_step_range": (2, 3),
                "sensor_failure_prob":    0.0,
                "conflict_injection":     False,
                "freeze_agent":           None,
                "seed_override":          None,
            }

        elif self.current_phase == 2:
            # Coordination pressure: engineered conflicts, moderate failures
            targets_min = 3 + int(d * 2)  # 3→5 as difficulty grows
            targets_max = targets_min + 2
            failure_prob = round(0.03 + d * 0.10, 3)  # 0.03→0.13
            return {
                "max_steps":              40,
                "targets_per_step_range": (targets_min, targets_max),
                "sensor_failure_prob":    failure_prob,
                "conflict_injection":     True,
                "freeze_agent":           None,
                "seed_override":          None,
            }

        else:
            # Adaptive self-play: high density, correlated failures, agent freezing
            targets_min = 4 + int(d * 4)  # 4→8 as difficulty grows
            targets_max = targets_min + 3
            failure_prob = round(0.05 + d * 0.20, 3)  # 0.05→0.25
            freeze_agent = self.get_frozen_agent() if self.should_trigger_self_play() else None
            return {
                "max_steps":              60,
                "targets_per_step_range": (targets_min, targets_max),
                "sensor_failure_prob":    failure_prob,
                "conflict_injection":     True,
                "freeze_agent":           freeze_agent,
                "seed_override":          None,  # fully dynamic in Phase 3
            }

    # ── Episode update ───────────────────────────────────────────────
    def update(self, coordination_score: float, efficiency_score: float) -> dict:
        """
        Called after each episode with the episode's coordination and efficiency scores.
        Returns {"phase": int, "difficulty": float, "action": str}.
        """
        self.episode_count += 1
        self._score_history.append(coordination_score)
        self._eff_history.append(efficiency_score)

        # Cap history
        max_history = self.window_size * 3
        if len(self._score_history) > max_history:
            self._score_history.pop(0)
        if len(self._eff_history) > max_history:
            self._eff_history.pop(0)

        rolling = self._rolling_coord()
        action  = "hold"

        # ── Phase transitions ──────────────────────────────────────
        old_phase = self.current_phase
        if self.current_phase == 1 and self.episode_count >= _PHASE1_END:
            self.current_phase = 2
            action = "phase_transition"
        elif self.current_phase == 2 and self.episode_count >= _PHASE2_END:
            self.current_phase = 3
            action = "phase_transition"

        if action == "phase_transition":
            logger.info(
                "[Curriculum] Phase %d → %d at episode %d | "
                "difficulty=%.3f rolling_coord=%.3f rolling_eff=%.3f",
                old_phase, self.current_phase, self.episode_count,
                self.difficulty_level, rolling, self._rolling_eff(),
            )
        else:
            # ── Difficulty adjustment ──────────────────────────────
            if rolling >= self.escalate_threshold:
                delta = min(_MAX_DIFFICULTY_DELTA, 1.0 - self.difficulty_level)
                self.difficulty_level = min(1.0, self.difficulty_level + delta)
                if delta > 0:
                    self._escalations_count += 1
                    action = "escalate"
            elif rolling < self.regress_threshold:
                delta = min(_MAX_DIFFICULTY_DELTA, self.difficulty_level)
                self.difficulty_level = max(0.0, self.difficulty_level - delta)
                if delta > 0:
                    self._regressions_count += 1
                    action = "regress"

        return {
            "phase":      self.current_phase,
            "difficulty": self.difficulty_level,
            "action":     action,
        }

    # ── Self-play helpers ────────────────────────────────────────────
    def should_trigger_self_play(self) -> bool:
        """True every 200 episodes in Phase 3."""
        if self.current_phase < 3:
            return False
        phase3_ep = self.episode_count - _PHASE2_END
        if phase3_ep > 0 and phase3_ep % _SELF_PLAY_INTERVAL == 0:
            return True
        return False

    def get_frozen_agent(self) -> str | None:
        """Rotates through SAT, UAV, RDR every self-play cycle. CMD never frozen."""
        if self.current_phase < 3:
            return None
        phase3_ep  = max(0, self.episode_count - _PHASE2_END)
        cycle_idx  = (phase3_ep // _SELF_PLAY_INTERVAL) % len(_SELF_PLAY_AGENTS)
        return _SELF_PLAY_AGENTS[cycle_idx]

    # ── Episode lifecycle ────────────────────────────────────────────
    def reset_episode(self) -> None:
        """
        Called at the START of each episode to check for self-play triggers.
        Does NOT increment episode_count — that is done by update() at episode end.
        """
        if self.should_trigger_self_play():
            self._self_play_triggers_count += 1
            logger.info(
                "[Curriculum] Self-play triggered at episode %d — freezing agent: %s",
                self.episode_count, self.get_frozen_agent(),
            )

    # ── Metrics ─────────────────────────────────────────────────────
    def get_metrics(self) -> dict:
        return {
            "episode_count":             self.episode_count,
            "phase":                     self.current_phase,
            "difficulty_level":          self.difficulty_level,
            "rolling_coordination_score":self._rolling_coord(),
            "rolling_efficiency_score":  self._rolling_eff(),
            "escalations_count":         self._escalations_count,
            "regressions_count":         self._regressions_count,
            "self_play_triggers_count":  self._self_play_triggers_count,
        }

    # ── Internal ─────────────────────────────────────────────────────
    def _rolling_coord(self) -> float:
        if not self._score_history:
            return 0.0
        window = self._score_history[-self.window_size:]
        return sum(window) / len(window)

    def _rolling_eff(self) -> float:
        if not self._eff_history:
            return 0.0
        window = self._eff_history[-self.window_size:]
        return sum(window) / len(window)
