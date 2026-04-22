"""
ARYA-X — trainer.py
Full TRL/GRPO multi-agent training loop.
All 4 agents share one base LLM with per-agent LoRA adapters.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Graceful optional imports ─────────────────────────────────────────────────
try:
    from trl import GRPOTrainer, GRPOConfig
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False
    logger.warning(
        "trl not found. Install with: pip install trl>=0.8.0\n"
        "Training will run in reward-logging-only mode."
    )

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logger.warning("transformers not found. Install with: pip install transformers")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    logger.warning("peft not found. Install with: pip install peft")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── Internal imports ──────────────────────────────────────────────────────────
from env.environment import SentinelEnv
from env.multiagent import AryaXEnv, AGENT_TYPES
from env.dynamics import SENSOR_TYPES
from interaction import NegotiationLayer
from interaction.reward import RewardEngine
from curriculum import CurriculumEngine
from tasks.grader import grade_episode
from agents import SatelliteAgent, DroneAgent, RadarAgent, CommandAgent

# Agent IDs (short codes used internally; map to specialised agent classes)
AGENT_IDS = ["SAT", "UAV", "RDR", "CMD"]

# Map short-code → AryaXEnv agent_id string
AGENT_ID_MAP = {"SAT": "satellite", "UAV": "drone", "RDR": "radar", "CMD": "command"}

# Metrics export path
METRICS_LOG_PATH = Path("./logs/training_metrics.json")

# Capability matrix (mirrors conflict.py)
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

# Agent → preferred sensor type mapping
AGENT_SENSOR_TYPE: dict[str, str] = {
    "SAT": "satellite",
    "UAV": "drone",
    "RDR": "radar",
    "CMD": "satellite",  # CMD uses satellite-class sensors
}

# Target type for capability lookup (default if not specified)
DEFAULT_TARGET_TYPE = "strategic"


# ── Simple greedy proposal generator (no LLM) ────────────────────────────────
def _greedy_proposal(
    agent_id: str,
    sensors: list[dict],
    targets: list[dict],
    rng: random.Random,
) -> dict | None:
    """Generate a single greedy proposal for an agent."""
    available = [s for s in sensors if s.get("available", True)]
    active = [t for t in targets if t.get("active", True)]
    if not available or not active:
        return None

    sensor_type_pref = AGENT_SENSOR_TYPE.get(agent_id, "radar")
    # Pick sensor: prefer matching type
    preferred = [s for s in available if s.get("type", "") == sensor_type_pref]
    sensor = rng.choice(preferred) if preferred else rng.choice(available)

    # Pick target: highest priority with some noise
    active_sorted = sorted(active, key=lambda t: -t.get("priority", 0))
    # With 30% chance, pick second-best (simulating agent imperfection)
    if len(active_sorted) > 1 and rng.random() < 0.30:
        target = active_sorted[1]
    else:
        target = active_sorted[0]

    target_type = target.get("type", DEFAULT_TARGET_TYPE)
    cap = CAPABILITY_MATRIX.get((sensor.get("type", ""), target_type), 0.0)
    priority_est = max(1, min(3, target.get("priority", 1) + rng.randint(-1, 1)))

    return {
        "agent_id":          agent_id,
        "sensor_id":         sensor["id"],
        "target_id":         target["id"],
        "priority_estimate": priority_est,
        "confidence":        round(rng.uniform(0.5, 0.95), 3),
        "capability_score":  round(cap, 3),
    }


def _cmd_override(tied_proposals: list[dict]) -> dict:
    """Command agent override: pick proposal with highest capability_score."""
    return max(tied_proposals, key=lambda p: p.get("capability_score", 0.0))


def _build_world_state(obs, step: int, proposals: list[dict]) -> dict:
    """Convert SentinelEnv Observation to the world_state dict format."""
    sensors_list = [
        {"id": s.id, "type": s.type, "range": s.range, "available": s.available}
        for s in obs.sensors
    ]
    targets_list = [
        {"id": t.id, "priority": t.priority, "active": t.active, "type": DEFAULT_TARGET_TYPE}
        for t in obs.targets
    ]
    assigned_sids = {p["sensor_id"] for p in proposals}
    idle_sensors = [s["id"] for s in sensors_list if s["id"] not in assigned_sids]
    return {
        "sensors":      sensors_list,
        "targets":      targets_list,
        "idle_sensors": idle_sensors,
        "step":         step,
        "proposals":    proposals,
    }


# ── ARYAXTrainer ─────────────────────────────────────────────────────────────
class ARYAXTrainer:
    """
    Orchestrates GRPO training for all 4 ARYA-X agents.
    Architecture: shared base LLM + per-agent LoRA adapters.
    Falls back to reward-logging mode when trl/unsloth is unavailable.
    """

    def __init__(
        self,
        model_name:   str   = "unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit",
        num_episodes: int   = 3000,
        batch_size:   int   = 8,
        eval_every:   int   = 50,
        G:            int   = 4,
        temperature:  float = 0.8,
        use_wandb:    bool  = False,
        hf_repo:      str | None = None,
    ) -> None:
        self.model_name   = model_name
        self.num_episodes = num_episodes
        self.batch_size   = batch_size
        self.eval_every   = eval_every
        self.G            = G
        self.temperature  = temperature
        self.use_wandb    = use_wandb
        self.hf_repo      = hf_repo

        self.curriculum       = CurriculumEngine()
        self.negotiation      = NegotiationLayer()
        self.reward_engine    = RewardEngine()
        self.checkpoint_dir   = Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics log directory
        METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_log: list[dict] = []

        # Specialised agent instances (deterministic, no LLM)
        self._agents = {
            "satellite": SatelliteAgent(),
            "drone":     DroneAgent(),
            "radar":     RadarAgent(),
            "command":   CommandAgent(max_steps=num_episodes),
        }

        self._model    = None
        self._tokenizer= None
        self._adapters: dict[str, Any] = {}
        self._grpo_trainer = None

        if use_wandb:
            try:
                import wandb
                wandb.init(project="arya-x", name=f"grpo-{int(time.time())}")
            except ImportError:
                logger.warning("wandb not installed — skipping W&B logging.")

        self._load_model()

        # Initialise GRPOTrainer after model is loaded
        if _TRL_AVAILABLE and self._model is not None and self._tokenizer is not None:
            grpo_cfg = GRPOConfig(
                output_dir="./grpo_output",
                num_train_epochs=1,
                per_device_train_batch_size=self.batch_size,
                temperature=self.temperature,
            )
            self._grpo_trainer = GRPOTrainer(
                model=self._model,
                config=grpo_cfg,
                tokenizer=self._tokenizer,
            )
            logger.info("GRPOTrainer initialised.")

    # ── Model loading ─────────────────────────────────────────────────
    def _load_model(self) -> None:
        if not _HF_AVAILABLE:
            logger.warning("HuggingFace transformers unavailable — using greedy mock.")
            return

        logger.info("Loading base model: %s", self.model_name)
        try:
            # Try Unsloth first (4-bit, faster)
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,
            )
            self._model     = model
            self._tokenizer = tokenizer
            logger.info("Unsloth model loaded successfully.")
        except Exception as ue:
            logger.warning("Unsloth load failed (%s) — falling back to vanilla HF.", ue)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model     = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_4bit=True if _TORCH_AVAILABLE else False,
                    device_map="auto" if _TORCH_AVAILABLE else None,
                )
            except Exception as he:
                logger.warning("HF model load also failed (%s) — greedy mode.", he)
                return

        # ── Per-agent LoRA adapters ──────────────────────────────
        if _PEFT_AVAILABLE and self._model is not None:
            for agent_id in AGENT_IDS:
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                )
                adapter_model = get_peft_model(self._model, lora_cfg)
                self._adapters[agent_id] = adapter_model
                logger.info("LoRA adapter initialized for agent: %s", agent_id)

    # ── Main training loop ────────────────────────────────────────────
    def train(self) -> dict:
        """
        Run the full GRPO training loop for num_episodes.
        Returns final evaluation metrics dict.
        """
        logger.info(
            "Starting ARYA-X GRPO training | episodes=%d | G=%d | batch=%d",
            self.num_episodes, self.G, self.batch_size,
        )

        all_episode_rewards: list[float] = []
        rng = random.Random(0)

        for episode in range(1, self.num_episodes + 1):
            # ── Phase 1: Episode rollout ──────────────────────────
            self.curriculum.reset_episode()
            cfg = self.curriculum.get_scenario_config()

            ep_seed = cfg.get("seed_override") or rng.randint(0, 10000)
            conflict_injection = cfg.get("conflict_injection", False)

            env = AryaXEnv(
                max_steps=cfg["max_steps"],
                seed=ep_seed,
                density_factor=1.5 + self.curriculum.difficulty_level * 2.5,
                failure_prob=cfg["sensor_failure_prob"],
            )
            # ── PARTIAL OBSERVABILITY: use AryaXEnv.reset() → per-agent obs ──
            obs_map = env.reset()   # Dict[agent_id → AgentObservation (dataclass)]
            self.negotiation.reset()
            self.reward_engine.reset()

            # Reset specialised agent episode state
            for agent in self._agents.values():
                agent.reset_episode()

            # Per-agent experience buffers: list of (obs_text, proposal_text, reward)
            agent_experiences: dict[str, list[tuple[str, str, float]]] = {
                aid: [] for aid in AGENT_IDS
            }

            # Accumulate rewards keyed by canonical agent_id string
            episode_rewards: dict[str, float] = {aid: 0.0 for aid in AGENT_TYPES}
            done = False

            while not done:
                # ── Each agent observes ONLY its own partial observation ──
                all_proposals_obj = []  # env.models.Proposal objects
                sensors_list_by_agent: dict[str, list] = {}
                targets_list_by_agent: dict[str, list] = {}

                # First pass: satellite, drone, radar observe & propose
                for canonical_id in ["satellite", "drone", "radar"]:
                    agent = self._agents[canonical_id]
                    agent_obs = obs_map[canonical_id]  # AgentObservation dataclass

                    # Build models.AgentObservation for the agent's observe() method
                    from env.models import AgentObservation as ModelAgentObs, Sensor as ModelSensor, Target as ModelTarget
                    model_obs = ModelAgentObs(
                        agent_id=canonical_id,
                        agent_type=canonical_id,
                        sensors=[ModelSensor(**s) for s in agent_obs.sensors],
                        targets=[ModelTarget(**t) for t in agent_obs.targets],
                        timestep=agent_obs.timestep,
                        conflict_history=[],
                    )
                    agent.observe(model_obs)
                    proposals_obj = agent.propose()
                    all_proposals_obj.extend(proposals_obj)

                    sensors_list_by_agent[canonical_id] = agent_obs.sensors
                    targets_list_by_agent[canonical_id] = agent_obs.targets

                # Command observes last (sees all other proposals for gap-filling)
                cmd_agent = self._agents["command"]
                cmd_obs_raw = obs_map["command"]
                from env.models import AgentObservation as ModelAgentObs, Sensor as ModelSensor, Target as ModelTarget
                cmd_model_obs = ModelAgentObs(
                    agent_id="command",
                    agent_type="command",
                    sensors=[ModelSensor(**s) for s in cmd_obs_raw.sensors],
                    targets=[ModelTarget(**t) for t in cmd_obs_raw.targets],
                    timestep=cmd_obs_raw.timestep,
                    conflict_history=[],
                )
                cmd_agent.observe(cmd_model_obs, proposals=all_proposals_obj)
                all_proposals_obj.extend(cmd_agent.propose())

                # Convert to dict format for negotiation layer
                from env.multiagent import Proposal as EnvProposal
                final_proposals_dict = [
                    {
                        "agent_id":          p.agent_id,
                        "sensor_id":         p.sensor_id,
                        "target_id":         p.target_id,
                        "priority_estimate": p.priority_estimate,
                        "confidence":        p.confidence,
                        "capability_score":  p.confidence,  # proxy
                    }
                    for p in all_proposals_obj
                ]

                # Build world_state from command's (global) observation
                sensors_list = cmd_obs_raw.sensors
                targets_list = cmd_obs_raw.targets
                assigned_sids = {p["sensor_id"] for p in final_proposals_dict}
                idle_sensors = [s["id"] for s in sensors_list if s["id"] not in assigned_sids]
                world_state = {
                    "sensors":      sensors_list,
                    "targets":      targets_list,
                    "idle_sensors": idle_sensors,
                    "step":         env.current_step,
                    "proposals":    final_proposals_dict,
                }

                # Negotiate via interaction/ (unchanged)
                neg_result = self.negotiation.negotiate(
                    final_proposals_dict, world_state, _cmd_override
                )

                # Step multiagent env
                env_proposals = [
                    EnvProposal(
                        agent_id=p["agent_id"],
                        sensor_id=p["sensor_id"],
                        target_id=p["target_id"],
                    )
                    for p in final_proposals_dict
                ]
                obs_map, step_rewards_env, done, info = env.step_multiagent(env_proposals)

                # Also compute rewards via RewardEngine for GRPO shaping
                step_rewards = self.reward_engine.compute_step_reward(
                    neg_result.final_assignments, neg_result, world_state, list(AGENT_TYPES)
                )
                for canonical_id in AGENT_TYPES:
                    episode_rewards[canonical_id] += step_rewards.get(canonical_id, 0.0)

                # Update agent conflict history
                for conflict in neg_result.conflicts_detected:
                    for canonical_id in AGENT_TYPES:
                        if canonical_id in (conflict.involved_agents or []):
                            self._agents[canonical_id].update(
                                step_rewards.get(canonical_id, 0.0),
                                []
                            )

                # Store GRPO experiences (keyed by short AGENT_IDS for backward compat)
                for short_id in AGENT_IDS:
                    canonical_id = AGENT_ID_MAP[short_id]
                    obs_text  = json.dumps({"sensors": sensors_list, "targets": targets_list})
                    prop = next(
                        (p for p in final_proposals_dict if p["agent_id"] == canonical_id),
                        None
                    )
                    if prop:
                        prop_text = json.dumps(prop)
                        agent_experiences[short_id].append(
                            (obs_text, prop_text, step_rewards.get(canonical_id, 0.0))
                        )

            # ── Phase 2: Look-ahead reward shaping ───────────────
            lookahead = self.reward_engine.compute_episode_lookahead(
                self.reward_engine._episode_buffer
            )
            for aid, bonus in lookahead.items():
                episode_rewards[aid] = episode_rewards.get(aid, 0.0) + bonus

            total_ep_reward = sum(episode_rewards.values())
            all_episode_rewards.append(total_ep_reward)

            # ── Phase 3: GRPO batch formation ────────────────────
            if self._grpo_trainer is not None:
                self._grpo_update(agent_experiences)

            # ── Phase 4: Curriculum update ────────────────────────
            conflict_rate      = self.negotiation.get_conflict_rate()
            coordination_score = round(1.0 - conflict_rate, 4)
            # obs_map still holds last state; use satellite obs for sensor count
            last_obs = obs_map.get("satellite") or next(iter(obs_map.values()))
            sensor_count = len(last_obs.sensors) if last_obs else 1
            scores = self.reward_engine.compute_scores(
                episode_rewards, cfg["max_steps"], sensor_count
            )
            curr_update = self.curriculum.update(
                coordination_score, scores.get("efficiency", 0.0)
            )

            # ── Phase 5: Structured training log ─────────────────
            avg_reward = sum(all_episode_rewards[-10:]) / min(10, len(all_episode_rewards))
            per_agent_fmt = {k: round(v, 3) for k, v in episode_rewards.items()}
            print(
                f"\n[EP {episode}]\n"
                f"  conflict_rate={conflict_rate:.4f}\n"
                f"  coordination_score={coordination_score:.4f}\n"
                f"  rewards={per_agent_fmt}\n"
                f"  total_reward={round(total_ep_reward, 3)}\n"
                f"  difficulty={curr_update['difficulty']:.2f}  "
                f"phase={curr_update['phase']}  "
                f"avg10={avg_reward:.2f}"
            )

            # ── Phase 6: JSON metrics export ──────────────────────
            self._metrics_log.append({
                "episode":            episode,
                "conflict_rate":      round(conflict_rate, 4),
                "coordination_score": round(coordination_score, 4),
                "reward":             round(total_ep_reward, 4),
                "per_agent_rewards":  per_agent_fmt,
                "difficulty":         round(curr_update["difficulty"], 3),
                "phase":              curr_update["phase"],
                "conflict_injection": conflict_injection,
            })
            # Flush metrics every 10 episodes (avoid losing data on crash)
            if episode % 10 == 0:
                with open(METRICS_LOG_PATH, "w") as f:
                    json.dump(self._metrics_log, f, indent=2)

            # ── Phase 7: Eval checkpoint ──────────────────────────
            if episode % self.eval_every == 0:
                eval_result = self.evaluate()
                logger.info("[Eval @ %d] %s", episode, eval_result)

            if episode % 500 == 0:
                self.save_checkpoint(str(self.checkpoint_dir / f"episode_{episode}"))

        # Final metrics flush
        with open(METRICS_LOG_PATH, "w") as f:
            json.dump(self._metrics_log, f, indent=2)
        logger.info("Training metrics saved to %s", METRICS_LOG_PATH)

        # Final evaluation
        return self.evaluate()

    # ── GRPO update step (stub — full impl requires live model) ──────
    def _grpo_update(self, agent_experiences: dict[str, list[tuple]]) -> None:
        """
        Group G proposals per step, compute group relative advantage,
        format and pass to GRPOTrainer for policy gradient update.
        """
        if self._grpo_trainer is None:
            return

        for agent_id, experiences in agent_experiences.items():
            if not experiences:
                continue
            queries, responses, rewards_list = zip(*experiences)
            rewards_arr = list(rewards_list)
            if len(rewards_arr) >= self.G:
                group_mean = sum(rewards_arr) / len(rewards_arr)
                advantages = [r - group_mean for r in rewards_arr]
            else:
                advantages = rewards_arr

            try:
                self._grpo_trainer.step(
                    queries=list(queries),
                    completions=list(responses),
                    rewards=advantages,
                )
                logger.debug(
                    "GRPO update for %s: %d samples, max_advantage=%.3f",
                    agent_id, len(rewards_arr), max(advantages, default=0.0),
                )
            except Exception as exc:
                logger.warning("GRPOTrainer.step() failed for %s: %s", agent_id, exc)

    # ── Evaluation ────────────────────────────────────────────────────
    def evaluate(self) -> dict:
        """Run deterministic eval on easy/medium/hard tasks."""
        try:
            from tasks.grader import run_deterministic_eval
            # Pass None since we are using greedy agents, not LLM agents
            result = run_deterministic_eval(agents=None, verbose=True, mode="single")
            conflict_rate      = self.negotiation.get_conflict_rate()
            coordination_score = 1.0 - conflict_rate
            result["conflict_rate"]      = conflict_rate
            result["coordination_score"] = coordination_score
            return result
        except Exception as exc:
            logger.error("evaluate() failed: %s", exc)
            return {"error": str(exc)}

    # ── Checkpoint ────────────────────────────────────────────────────
    def save_checkpoint(self, path: str) -> None:
        """Save model adapters and training state to disk."""
        ckpt_path = Path(path)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Save curriculum state
        import json as _json
        with open(ckpt_path / "curriculum_metrics.json", "w") as f:
            _json.dump(self.curriculum.get_metrics(), f, indent=2)

        # Save negotiation history summary
        history = self.negotiation.get_step_metrics_history()[-100:]
        with open(ckpt_path / "negotiation_history.json", "w") as f:
            _json.dump(history, f, indent=2)

        # Save LoRA adapters
        if _PEFT_AVAILABLE and self._adapters:
            for agent_id, adapter in self._adapters.items():
                adapter_path = ckpt_path / f"adapter_{agent_id}"
                try:
                    adapter.save_pretrained(str(adapter_path))
                except Exception as exc:
                    logger.warning("Could not save adapter for %s: %s", agent_id, exc)

        # Push to HuggingFace Hub
        if self.hf_repo:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=str(ckpt_path),
                    repo_id=self.hf_repo,
                    repo_type="model",
                )
                logger.info("Pushed checkpoint to HuggingFace: %s", self.hf_repo)
            except Exception as exc:
                logger.warning("HuggingFace push failed: %s", exc)

        logger.info("Checkpoint saved to: %s", ckpt_path)
