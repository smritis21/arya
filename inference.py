"""
SentinelEnv — Inference Script
================================
Runs the LLM agent against all 3 tasks (Easy / Medium / Hard) and prints
a reproducible baseline score for each.

Environment variables required:
    API_BASE_URL   LLM API endpoint (fallback for single-agent)
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face API token (optional)
"""
import os
import json
import logging
from pathlib import Path

# Try to load local huggingface/peft dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    _LOCAL_HF_AVAILABLE = True
except ImportError:
    _LOCAL_HF_AVAILABLE = False

from openai import OpenAI
from env import SentinelEnv
from env.models import Action
from env.multiagent import AryaXEnv, Proposal, AGENT_TYPES
from interaction import NegotiationLayer
from interaction.reward import RewardEngine
from tasks.grader import grade_episode
from tasks.easy_task import get_easy_env
from tasks.medium_task import get_medium_env
from tasks.hard_task import get_hard_env

from agents.satellite import SatelliteAgent
from agents.drone import DroneAgent
from agents.radar import RadarAgent
from agents.command import CommandAgent

USE_MULTI_AGENT = True
CAPABILITY_MATRIX = {
    ("satellite","strategic"):0.95, ("satellite","kinetic"):0.40, ("satellite","airspace"):0.60,
    ("drone","kinetic"):0.95, ("drone","strategic"):0.30, ("drone","airspace"):0.50,
    ("radar","airspace"):0.95, ("radar","kinetic"):0.65, ("radar","strategic"):0.45,
}

sat_agent = SatelliteAgent()
drone_agent = DroneAgent()
radar_agent = RadarAgent()
command_agent = CommandAgent()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")

client = None

# Global references for our local models
_base_model = None
_tokenizer = None
_has_adapters = False

# Fallback mappings for local checkpoints (matching AGENT_IDS from trainer)
AGENT_ID_MAP = {"satellite": "SAT", "drone": "UAV", "radar": "RDR", "command": "CMD"}

TASKS = [
    {"name": "Easy",   "env_fn": get_easy_env,   "seed": 42},
    {"name": "Medium", "env_fn": get_medium_env, "seed": 7},
    {"name": "Hard",   "env_fn": get_hard_env,   "seed": 13},
]

def init_local_models():
    """Load the single shared LoRA checkpoint for all agents."""
    global _base_model, _tokenizer, _has_adapters
    if not _LOCAL_HF_AVAILABLE:
        print("[WARN] transformers or peft not installed. Will use greedy fallback.")
        return

    checkpoint_dir = Path("./checkpoints/arya_x_lora_3")
    if not (checkpoint_dir / "adapter_model.safetensors").exists():
        print(f"[WARN] No adapter found at {checkpoint_dir}. Will use greedy fallback.")
        return

    # Read base model name directly from the saved adapter config
    import json as _json
    base_model_name = _json.loads((checkpoint_dir / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"[LLM] Loading {base_model_name} + shared LoRA adapter...")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), padding_side="left")

        load_kwargs = {"device_map": "auto"}
        if torch.cuda.is_available():
            load_kwargs["load_in_4bit"] = True

        base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
        _base_model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        _base_model.eval()
        _has_adapters = True
        print("[LLM] Shared LoRA adapter loaded for all agents.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        _has_adapters = False

def build_prompt(obs) -> str:
    sensors = "\\n".join(
        f"  - id={s.id} type={s.type} range={s.range}km available={s.available}"
        for s in obs.sensors if s.available
    )
    targets = "\\n".join(
        f"  - id={t.id} priority={t.priority}"
        for t in obs.targets if t.active
    )
    n = sum(1 for s in obs.sensors if s.available)
    return f"""You are a military sensor allocation AI. Assign ALL {n} available sensors to threats.
Priority 3=HIGH (critical), 2=MED, 1=LOW. Cover HIGH threats first. Each sensor to a DIFFERENT target.

Timestep: {obs.timestep}

Available Sensors:
{sensors}

Active Threats:
{targets}

Respond ONLY with a JSON array, one object per available sensor:
[{{"sensor_id": "S1", "target_id": "T0_1"}}, {{"sensor_id": "S2", "target_id": "T0_2"}}]
"""


def parse_llm_actions(text: str, obs) -> list[Action]:
    try:
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        data  = json.loads(text[start:end])
        valid_sensors = {s.id for s in obs.sensors if s.available}
        valid_targets = {t.id for t in obs.targets if t.active}
        actions, used_s, used_t = [], set(), set()
        for item in data:
            sid, tid = item.get("sensor_id"), item.get("target_id")
            if sid in valid_sensors and tid in valid_targets \
                    and sid not in used_s and tid not in used_t:
                actions.append(Action(sensor_id=sid, target_id=tid))
                used_s.add(sid)
                used_t.add(tid)
        return actions
    except Exception:
        return []


def greedy_actions(obs) -> list[Action]:
    available = [s for s in obs.sensors if s.available]
    targets   = sorted([t for t in obs.targets if t.active], key=lambda t: -t.priority)
    actions, used = [], set()
    for sensor in available:
        for target in targets:
            if target.id not in used:
                actions.append(Action(sensor_id=sensor.id, target_id=target.id))
                used.add(target.id)
                break
    return actions


def parse_action(output: str, obs) -> Action | None:
    """Wrapper around parse_llm_actions — returns first valid Action or None."""
    actions = parse_llm_actions(output, obs)
    return actions[0] if actions else None


def fallback_action(obs) -> Action | None:
    """Wrapper around greedy_actions — returns highest-priority Action or None."""
    actions = greedy_actions(obs)
    return actions[0] if actions else None


def get_actions(obs) -> tuple[list[Action], str]:
    if HF_TOKEN and client is not None and not _has_adapters: # Only use HF API if no local adapters
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": build_prompt(obs)}],
                max_tokens=256,
                temperature=0.0
            )
            raw     = response.choices[0].message.content.strip()
            actions = parse_llm_actions(raw, obs)
            if actions:
                return actions, "llm_remote"
            print(f"  [WARN] Bad LLM response, using greedy. Raw: {raw!r}")
        except Exception as e:
            print(f"  [WARN] LLM error: {e}. Using greedy.")
    return greedy_actions(obs), "greedy"


def log_end(task: str, score: float, steps: int) -> None:
    # Clamp INSIDE log_end — score is always strictly within (0.01, 0.99) when printed
    safe_score = min(max(float(score), 0.01), 0.99)
    print(f"[END] task={task} score={safe_score:.2f} steps={steps}", flush=True)


def run_task(name: str, env: SentinelEnv) -> float:
    print(f"\\n{'='*50}")
    print(f"  TASK: {name.upper()}  |  max_steps={env.max_steps}  |  seed={env.seed}")
    print(f"{'='*50}")

    obs          = env.reset()
    total_reward = 0.0
    done         = False
    llm_steps    = 0
    greedy_steps = 0
    steps_taken  = 0

    print(f"[START] task={name} env=sentinel model={MODEL_NAME}", flush=True)
    print(f"Sensors={len(obs.sensors)} | Targets={len(obs.targets)}")

    while not done:
        actions, source = get_actions(obs)
        obs, reward, done, info = env.step_batch(actions)
        total_reward += reward
        steps_taken  += 1
        if source.startswith("llm"):
            llm_steps += 1
        else:
            greedy_steps += 1
        assignments_str = ", ".join(f"{a.sensor_id}->{a.target_id}" for a in actions) if actions else "none"

        print(
            f"[STEP] step={steps_taken} action={assignments_str} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True
        )

    raw_score = grade_episode(total_reward, info["step_count"], num_sensors=env.initial_sensor_count)
    score = min(max(float(raw_score), 0.01), 0.99)

    print(f"\\n  Total Reward : {total_reward:.1f}")
    print(f"  Steps        : {info['step_count']}")
    print(f"  LLM steps    : {llm_steps}  |  Greedy fallback: {greedy_steps}")
    print(f"  Missed HIGH  : {len(info['missed_targets'])}")
    print(f"  SCORE        : {score:.4f}  (strictly in 0.01 – 0.99)")

    log_end(task=name, score=score, steps=steps_taken)
    return score


# ── Multi-agent greedy proposals ─────────────────────────────────────────────
def _greedy_multi_proposals(obs_map: dict) -> list[Proposal]:
    """Coordinated greedy proposals — agents share used_sensors and used_targets globally."""
    proposals: list[Proposal] = []
    used_sensors: set = set()
    used_targets: set = set()
    affinity = {"satellite": "satellite", "drone": "drone", "radar": "radar", "command": None}
    
    ordered = [a for a in AGENT_TYPES if a != "command"] + ["command"]
    for agent_id in ordered:
        agent_obs = obs_map[agent_id]
        sensors_raw = agent_obs.sensors if hasattr(agent_obs, "sensors") else []
        targets_raw = agent_obs.targets if hasattr(agent_obs, "targets") else []
        aff = affinity[agent_id]
        my_sensors = [
            s for s in sensors_raw
            if s["available"] and s["id"] not in used_sensors
            and (aff is None or s["type"] == aff)
        ]
        active_targets = sorted(
            [t for t in targets_raw if t["active"] and t["id"] not in used_targets],
            key=lambda t: -t["priority"]
        )
        for sensor in my_sensors:
            for target in active_targets:
                if target["id"] not in used_targets:
                    proposals.append(Proposal(
                        agent_id=agent_id,
                        sensor_id=sensor["id"],
                        target_id=target["id"],
                    ))
                    used_sensors.add(sensor["id"])
                    used_targets.add(target["id"])
                    break
    return proposals

# ── Multi-agent Local LoRA proposals ─────────────────────────────────────────
def _lora_multi_proposals(obs_map: dict) -> list[Proposal]:
    """Generates multi-agent proposals using the local LoRA adapters."""
    proposals: list[Proposal] = []
    
    # We maintain independence per-agent (partial observability)
    for agent_id in AGENT_TYPES:
        agent_obs = obs_map[agent_id]
        
        # Build prompt using existing logic (adapt agent_obs to pseudo-SingleAgentObs)
        class _PseudoObs:
            def __init__(self, s, t, t_step):
                self.sensors = [type("O",(object,),s_dict)() for s_dict in s]
                self.targets = [type("O",(object,),t_dict)() for t_dict in t]
                self.timestep = t_step
                
        pseudo_obs = _PseudoObs(
            agent_obs.sensors if hasattr(agent_obs, "sensors") else [],
            agent_obs.targets if hasattr(agent_obs, "targets") else [],
            agent_obs.timestep if hasattr(agent_obs, "timestep") else 0
        )
        
        prompt = build_prompt(pseudo_obs)
        chat_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        
        try:
            inputs = _tokenizer(chat_prompt, return_tensors="pt").to(_base_model.device)
            outputs = _base_model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True, pad_token_id=_tokenizer.eos_token_id)
            full_response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant response
            if "assistant" in full_response:
                response = full_response.split("assistant")[-1].strip()
            else:
                response = full_response
                
            actions = parse_llm_actions(response, pseudo_obs)
            for a in actions:
                proposals.append(Proposal(
                    agent_id=agent_id,
                    sensor_id=a.sensor_id,
                    target_id=a.target_id,
                ))
        except Exception as e:
            print(f"  [WARN] LoRA generation failed for {agent_id}: {e}")
            
    return proposals


def run_multi_task(name: str, env: AryaXEnv) -> float:
    print(f"\\n{'='*50}")
    print(f"  TASK (MULTI): {name.upper()}  |  max_steps={env.max_steps}  |  seed={env.seed}")
    print(f"{'='*50}")

    obs_map       = env.reset()
    negotiation   = NegotiationLayer()
    reward_eng    = RewardEngine()
    total_rewards = {a: 0.0 for a in AGENT_TYPES}
    done          = False
    steps_taken   = 0

    print(f"[START] task={name} env=arya-x (multi-agent)", flush=True)

    while not done:
        if USE_MULTI_AGENT:
            sat_agent.observe(obs_map["satellite"])
            drone_agent.observe(obs_map["drone"])
            radar_agent.observe(obs_map["radar"])
            command_agent.observe(obs_map["command"])

            proposal_objs = []
            proposal_objs += sat_agent.propose()
            proposal_objs += drone_agent.propose()
            proposal_objs += radar_agent.propose()
            proposal_objs += command_agent.propose()
        else:
            if _has_adapters:
                proposal_objs = _lora_multi_proposals(obs_map)
            else:
                proposal_objs = _greedy_multi_proposals(obs_map)
            
        sat_obs = obs_map.get("satellite") or next(iter(obs_map.values()))
        sensors_list = sat_obs.sensors
        targets_list = sat_obs.targets

        proposals_raw = []
        for p in proposal_objs:
            s_type = next((s.get("type", "unknown") if isinstance(s, dict) else s.type for s in sensors_list if (s.get("id") if isinstance(s, dict) else s.id) == p.sensor_id), "unknown")
            t_type = next((t.get("type", "strategic") if isinstance(t, dict) else getattr(t, "type", "strategic") for t in targets_list if (t.get("id") if isinstance(t, dict) else t.id) == p.target_id), "strategic")
            cap_score = CAPABILITY_MATRIX.get((s_type, t_type), 0.5)
            proposals_raw.append({
                "agent_id": p.agent_id, 
                "sensor_id": p.sensor_id,
                "target_id": p.target_id, 
                "capability_score": getattr(p, "confidence", cap_score)
            })

        # Build world_state from satellite's (global) observation
        assigned_sids = {p.sensor_id for p in proposal_objs}
        idle_sensors  = [(s.get("id") if isinstance(s, dict) else s.id) for s in sensors_list if (s.get("id") if isinstance(s, dict) else s.id) not in assigned_sids]
        world_state = {
            "sensors":      sensors_list,
            "targets":      targets_list,
            "idle_sensors": idle_sensors,
            "step":         steps_taken,
            "proposals":    proposals_raw,
        }

        # Run full NegotiationLayer pipeline
        neg_result = negotiation.negotiate(proposals_raw, world_state, lambda tied: tied[0])

        # Step env with resolved assignments
        obs_map, step_rewards, done, info = env.step_multiagent(proposal_objs)
        steps_taken += 1
        for aid in AGENT_TYPES:
            total_rewards[aid] += step_rewards.get(aid, 0.0)

        # Log conflicts
        n_conflicts = len(neg_result.conflicts_detected)
        if n_conflicts:
            override = neg_result.override_invoked
            print(
                f"[CONFLICT] step={steps_taken} n={n_conflicts} override={override}",
                flush=True,
            )

    # Episode summary
    conflict_rate      = negotiation.get_conflict_rate()
    coordination_score = 1.0 - conflict_rate
    scores             = reward_eng.compute_scores(total_rewards, env.max_steps, env.initial_sensor_count)

    print(f"\\nconflict_rate={conflict_rate:.4f}")
    print(f"coordination_score={coordination_score:.4f}")
    print(f"total_reward={sum(total_rewards.values()):.2f}")
    print(f"per_agent_rewards={total_rewards}")
    print(f"  efficiency  : {scores['efficiency']:.4f}")
    print(f"  final_score : {scores['final_score']:.4f}")
    return scores["final_score"]


if __name__ == "__main__":
    print("\\nInitializing Environment and Models...")
    init_local_models()

    if HF_TOKEN and not _has_adapters:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        print(f"[LLM] Using remote HF API: {MODEL_NAME}\\n")
    elif not _has_adapters:
        client = None
        print("[WARN] Local adapters missing and HF_TOKEN not set — running in greedy fallback.\\n")

    # ── Single-agent tasks ────────────────────────────────────────────
    results = {}
    for task in TASKS:
        env = task["env_fn"]()
        env.seed = task["seed"]
        score = run_task(task["name"], env)
        results[task["name"]] = score

    print(f"\\n{'='*50}")
    print("  FINAL SCORES (single-agent)")
    print(f"{'='*50}")
    for name, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {name:<8} {score:.4f}  {bar}")
    print(f"\\n  Average: {sum(results.values()) / len(results):.4f}")
    print(f"{'='*50}")

    # ── Multi-agent tasks ─────────────────────────────────────────────
    MULTI_TASKS = [
        {"name": "Easy",   "seed": 42, "max_steps": 20},
        {"name": "Medium", "seed": 7,  "max_steps": 40},
        {"name": "Hard",   "seed": 13, "max_steps": 60},
    ]
    multi_results = {}
    for task in MULTI_TASKS:
        mx_env = AryaXEnv(max_steps=task["max_steps"], seed=task["seed"])
        score  = run_multi_task(task["name"], mx_env)
        multi_results[task["name"]] = score

    print(f"\\n{'='*50}")
    print("  FINAL SCORES (multi-agent)")
    print(f"{'='*50}")
    for name, score in multi_results.items():
        bar = "█" * int(score * 20)
        print(f"  {name:<8} {score:.4f}  {bar}")
    print(f"\\n  Average: {sum(multi_results.values()) / len(multi_results):.4f}")
    print(f"{'='*50}")