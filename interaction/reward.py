"""
ARYA-X — Reward Engine
All 4 multi-agent reward components:
  1. Task Reward (per-agent, capability-weighted)
  2. Coordination Bonus (system-level, split equally)
  3. Conflict Penalty (involved agents only)
  4. Look-Ahead Planning Incentive (retroactive, per episode)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .negotiation import NegotiationResult

logger = logging.getLogger(__name__)

# ── Capability Matrix ────────────────────────────────────────────────────────
CAPABILITY_MATRIX: dict[tuple[str, str], float] = {
    ("satellite", "strategic"): 0.95,
    ("satellite", "kinetic"):   0.40,
    ("satellite", "airspace"):  0.60,
    ("drone",     "kinetic"):   0.95,
    ("drone",     "strategic"): 0.30,
    ("drone",     "airspace"):  0.50,
    ("radar",     "airspace"):  0.95,
    ("radar",     "kinetic"):   0.65,
    ("radar",     "strategic"): 0.45,
}

# Reward constants
TASK_P3_OPTIMAL     =  3.0
TASK_P3_NONOPTIMAL  =  2.0
TASK_P2             =  1.0
TASK_P1             =  0.5
TASK_IDLE_PENALTY   = -2.0

COORD_RESOLVE_NO_OVERRIDE =  1.5
COORD_ALL_P3_COVERED      =  2.0
COORD_ZERO_IDLE_WITH_P3   =  1.0

PENALTY_REDUNDANT         = -1.0
PENALTY_FORCED_ARBI       = -1.5
PENALTY_OVERRIDE_ONCE     = -0.5

OPTIMAL_CAPABILITY_THRESHOLD = 0.85


class RewardEngine:
    """
    Computes per-agent and system-level rewards for ARYA-X multi-agent episodes.
    """

    def __init__(self, gamma: float = 0.85, lookahead_window: int = 3) -> None:
        self.gamma             = gamma
        self.lookahead_window  = lookahead_window
        self._episode_buffer: list[dict] = []

        # Cumulative stats for compute_scores()
        self._total_covered_p3  = 0
        self._total_covered_p2  = 0
        self._total_covered_p1  = 0
        self._total_conflicts   = 0
        self._conflicts_resolved_auto = 0  # resolved without override

    # ── Step reward ──────────────────────────────────────────────────
    def compute_step_reward(
        self,
        final_assignments:  list[dict],
        negotiation_result: "NegotiationResult",
        world_state:        dict[str, Any],
        agent_ids:          list[str],
    ) -> dict[str, float]:
        """
        Returns {agent_id: float} step reward.
        Also appends a per-step snapshot to self._episode_buffer.
        """
        rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}

        if not final_assignments:
            # Append buffer entry even for empty step
            self._append_buffer(final_assignments, negotiation_result, world_state, rewards)
            return rewards

        target_map: dict[str, dict] = {
            t["id"]: t for t in world_state.get("targets", [])
        }
        sensor_map: dict[str, dict] = {
            s["id"]: s for s in world_state.get("sensors", [])
        }

        # Build reverse lookup: sensor_id → agent_id (from proposals)
        sensor_to_agent: dict[str, str] = {}
        for p in world_state.get("proposals", []):
            sensor_to_agent[p.get("sensor_id", "")] = p.get("agent_id", "")

        assigned_sensor_ids = {a["sensor_id"] for a in final_assignments}

        # ── Component 1 — Task Reward ─────────────────────────────
        covered_p3_targets: set[str] = set()
        covered_p2_targets: set[str] = set()
        covered_p1_targets: set[str] = set()

        for assignment in final_assignments:
            sid = assignment["sensor_id"]
            tid = assignment["target_id"]
            agent_id = sensor_to_agent.get(sid, agent_ids[0])

            tgt = target_map.get(tid, {})
            priority = tgt.get("priority", 0)
            target_type = tgt.get("type", "strategic")
            sensor_type = sensor_map.get(sid, {}).get("type", "")
            cap = CAPABILITY_MATRIX.get((sensor_type, target_type), 0.0)

            if priority == 3:
                covered_p3_targets.add(tid)
                r = TASK_P3_OPTIMAL if cap >= OPTIMAL_CAPABILITY_THRESHOLD else TASK_P3_NONOPTIMAL
            elif priority == 2:
                covered_p2_targets.add(tid)
                r = TASK_P2
            elif priority == 1:
                covered_p1_targets.add(tid)
                r = TASK_P1
            else:
                r = 0.0

            if agent_id in rewards:
                rewards[agent_id] += r

        # Idle-sensor penalty
        all_active_p3 = [
            t for t in world_state.get("targets", [])
            if t.get("active", True) and t.get("priority", 0) == 3
        ]
        idle_sensor_ids = [
            s["id"] for s in world_state.get("sensors", [])
            if s.get("available", True) and s["id"] not in assigned_sensor_ids
        ]
        for sid in idle_sensor_ids:
            sensor_type = sensor_map.get(sid, {}).get("type", "")
            for tgt in all_active_p3:
                if tgt["id"] in covered_p3_targets:
                    continue
                cap = CAPABILITY_MATRIX.get((sensor_type, tgt.get("type", "strategic")), 0.0)
                if cap > 0.5:
                    # This idle sensor could have covered a high threat
                    agent_id = sensor_to_agent.get(sid)
                    if agent_id and agent_id in rewards:
                        rewards[agent_id] += TASK_IDLE_PENALTY
                    break

        # ── Component 2 — Coordination Bonus (system-level) ──────
        coord_bonus = 0.0
        override_inv = negotiation_result.override_invoked
        all_p3_covered = bool(all_active_p3) and all(
            t["id"] in covered_p3_targets for t in all_active_p3
        )
        zero_idle = len(idle_sensor_ids) == 0

        if not override_inv and negotiation_result.conflicts_detected:
            coord_bonus += COORD_RESOLVE_NO_OVERRIDE  # self-resolved
        if all_p3_covered:
            coord_bonus += COORD_ALL_P3_COVERED
        if zero_idle and all_active_p3:
            coord_bonus += COORD_ZERO_IDLE_WITH_P3

        if coord_bonus and agent_ids:
            per_agent = coord_bonus / len(agent_ids)
            for aid in agent_ids:
                rewards[aid] += per_agent

        # ── Component 3 — Conflict Penalties ─────────────────────
        from .conflict import ConflictType
        for conflict in negotiation_result.conflicts_detected:
            if conflict.conflict_type == ConflictType.REDUNDANT_COVERAGE:
                penalty = PENALTY_REDUNDANT
            elif conflict.conflict_type == ConflictType.FORCED_ARBITRATION:
                penalty = PENALTY_FORCED_ARBI
            else:
                penalty = 0.0

            for aid in conflict.involved_agents:
                if aid in rewards:
                    rewards[aid] += penalty

        if override_inv:
            # Override penalty split across all agents once per step
            pen = PENALTY_OVERRIDE_ONCE / max(len(agent_ids), 1)
            for aid in agent_ids:
                rewards[aid] += pen

        # ── Update episode stats ──────────────────────────────────
        self._total_covered_p3 += len(covered_p3_targets)
        self._total_covered_p2 += len(covered_p2_targets)
        self._total_covered_p1 += len(covered_p1_targets)
        self._total_conflicts  += len(negotiation_result.conflicts_detected)
        if negotiation_result.conflicts_detected and not override_inv:
            self._conflicts_resolved_auto += 1

        # ── Debug logging ─────────────────────────────────────────
        logger.debug(
            "Step %d rewards: %s | coord_bonus=%.2f | override=%s",
            world_state.get("step", 0), rewards, coord_bonus, override_inv,
        )

        self._append_buffer(final_assignments, negotiation_result, world_state, rewards)
        return rewards

    # ── Episode look-ahead ───────────────────────────────────────────
    def compute_episode_lookahead(
        self,
        episode_history: list[dict],
    ) -> dict[str, float]:
        """
        Retroactively compute look-ahead planning bonuses over the full episode.
        Returns {agent_id: additional_reward}.
        Formula: R_la(t) = gamma * R_future(t+k) * I(idle_sensor_t, target_t+k)
        """
        if not episode_history:
            return {}

        agent_ids = list(episode_history[0].get("rewards", {}).keys()) if episode_history else []
        lookahead_rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}

        n = len(episode_history)
        for t, snap in enumerate(episode_history):
            idle_sensors = snap.get("idle_sensor_types", {})  # {sensor_id: sensor_type}
            if not idle_sensors:
                continue

            for k in range(1, self.lookahead_window + 1):
                future_t = t + k
                if future_t >= n:
                    break

                future_snap = episode_history[future_t]
                future_p3_targets = future_snap.get("active_p3_targets", [])
                # {target_id: target_type}

                for sid, stype in idle_sensors.items():
                    for tgt in future_p3_targets:
                        ttype = tgt.get("type", "strategic")
                        cap = CAPABILITY_MATRIX.get((stype, ttype), 0.0)
                        if cap >= OPTIMAL_CAPABILITY_THRESHOLD:
                            # I = 1 — sensor was reserved and ideal for this future target
                            future_reward = future_snap.get("system_reward", 0.0)
                            bonus = (self.gamma ** k) * future_reward
                            # Attribute to the agent that held this sensor idle
                            agent_id = snap.get("idle_sensor_agents", {}).get(sid)
                            if agent_id and agent_id in lookahead_rewards:
                                lookahead_rewards[agent_id] += bonus

        return lookahead_rewards

    # ── Episode score summary ────────────────────────────────────────
    def compute_scores(
        self,
        total_rewards: dict[str, float],
        step_count:    int,
        num_sensors:   int,
    ) -> dict[str, float]:
        """
        Returns efficiency, coordination_score, and final_score.
        """
        # Max possible reward per step: every sensor covers a p3 optimally
        max_possible_reward = num_sensors * step_count * TASK_P3_OPTIMAL

        # Covered counts accumulated across episode
        weighted_covered = (
            self._total_covered_p3 * 3.0
            + self._total_covered_p2 * 1.0
            + self._total_covered_p1 * 0.5
        )
        efficiency = weighted_covered / max(max_possible_reward, 1.0)
        efficiency = max(0.0, min(1.0, efficiency))

        coordination_score = 1.0 - (
            (self._total_conflicts - self._conflicts_resolved_auto)
            / max(self._total_conflicts, 1)
        )
        coordination_score = max(0.0, min(1.0, coordination_score))

        final_score = 0.6 * efficiency + 0.4 * coordination_score

        return {
            "efficiency":         efficiency,
            "coordination_score": coordination_score,
            "final_score":        final_score,
        }

    # ── Lifecycle ────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear episode buffer — call at the start of each episode."""
        self._episode_buffer.clear()
        self._total_covered_p3 = 0
        self._total_covered_p2 = 0
        self._total_covered_p1 = 0
        self._total_conflicts  = 0
        self._conflicts_resolved_auto = 0

    # ── Internal ─────────────────────────────────────────────────────
    def _append_buffer(
        self,
        final_assignments:  list[dict],
        negotiation_result: "NegotiationResult",
        world_state:        dict,
        rewards:            dict[str, float],
    ) -> None:
        sensor_map = {s["id"]: s for s in world_state.get("sensors", [])}
        targets    = world_state.get("targets", [])

        # Idle sensors with their types and responsible agents
        assigned_ids = {a["sensor_id"] for a in final_assignments}
        proposals    = world_state.get("proposals", [])
        sensor_to_agent = {p["sensor_id"]: p["agent_id"] for p in proposals}

        idle_sensor_types: dict[str, str] = {}
        idle_sensor_agents: dict[str, str] = {}
        for s in world_state.get("sensors", []):
            if s["id"] not in assigned_ids and s.get("available", True):
                idle_sensor_types[s["id"]] = s.get("type", "")
                idle_sensor_agents[s["id"]] = sensor_to_agent.get(s["id"], "")

        active_p3 = [
            {"id": t["id"], "type": t.get("type", "strategic")}
            for t in targets
            if t.get("active", True) and t.get("priority", 0) == 3
        ]

        system_reward = sum(rewards.values())

        self._episode_buffer.append({
            "step":                world_state.get("step", 0),
            "rewards":             dict(rewards),
            "system_reward":       system_reward,
            "idle_sensor_types":   idle_sensor_types,
            "idle_sensor_agents":  idle_sensor_agents,
            "active_p3_targets":   active_p3,
            "num_conflicts":       len(negotiation_result.conflicts_detected),
        })
