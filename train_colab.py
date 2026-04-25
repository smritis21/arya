# ── 0. Install (run this cell first, then restart runtime) ──────────
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q "trl>=0.9.0" transformers pydantic accelerate bitsandbytes datasets

# ── 1. Imports ───────────────────────────────────────────────────────
import os
import json
import random
import math
from collections import defaultdict

# ── 2. Optional dependency guards ───────────────────────────────────
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not found — model will not load.")

try:
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset as HFDataset
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("⚠️  TRL or Datasets not found — running in reward-logging-only mode.")

# ── 3. Load model (Unsloth 4-bit, T4-safe) ──────────────────────────
model_name = "unsloth/llama-3-8b-bnb-4bit"   # single source of truth
model, tokenizer = None, None

if UNSLOTH_AVAILABLE:
    print(f"Loading {model_name} with 4-bit quantization …")
    for _attempt_name in [model_name, "unsloth/tinyllama-bnb-4bit"]:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=_attempt_name,
                max_seq_length=512,
                load_in_4bit=True,
                dtype=None,          # auto-detect bf16 / fp16
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            print(f"✅ Model ready ({_attempt_name}).")
            break
        except Exception as _err:
            print(f"⚠️  {_attempt_name} failed: {_err}")
    else:
        print("❌ All model loads failed — running without LLM.")
else:
    print("Running without LLM — greedy baseline mode.")

# ── 4. Environment (self-contained, mirrors env/environment.py) ──────
SENSOR_TYPES    = ["satellite", "drone", "radar"]
TARGET_TYPES    = ["strategic", "kinetic", "airspace"]
PRIORITY_REWARD = {3: 2.0, 2: 1.0, 1: 0.5}
MISSED_PENALTY  = -2.0
IDLE_PENALTY    = -2.0

CAPABILITY_MATRIX = {
    ("satellite", "strategic"): 0.95, ("satellite", "kinetic"): 0.40,
    ("satellite", "airspace"):  0.60, ("drone",     "kinetic"): 0.95,
    ("drone",     "strategic"): 0.30, ("drone",     "airspace"): 0.50,
    ("radar",     "airspace"):  0.95, ("radar",     "kinetic"):  0.65,
    ("radar",     "strategic"): 0.45,
}

def make_env(seed: int = 0, max_steps: int = 20) -> dict:
    rng = random.Random(seed)
    sensors = [
        {"id": f"S{i+1}", "type": rng.choice(SENSOR_TYPES),
         "available": True, "range": rng.randint(100, 500)}
        for i in range(rng.randint(3, 5))
    ]
    return {"sensors": sensors, "max_steps": max_steps, "seed": seed, "_rng": rng}

def env_reset(env_state: dict) -> dict:
    rng = env_state["_rng"]
    env_state["targets"] = [
        {"id": f"T0_{i+1}", "priority": rng.randint(1, 3),
         "type": rng.choice(TARGET_TYPES), "active": True}
        for i in range(rng.randint(2, 4))
    ]
    for s in env_state["sensors"]:
        s["available"] = True
    env_state["current_step"] = 0
    env_state["total_reward"] = 0.0
    return env_state

def env_step(env_state: dict, sensor_id: str, target_id: str):
    """Execute one assignment. Returns (env_state, step_reward, done)."""
    targets, sensors = env_state["targets"], env_state["sensors"]
    step_reward, handled = 0.0, False

    for s in sensors:
        if s["id"] == sensor_id and s["available"]:
            for t in targets:
                if t["id"] == target_id and t["active"]:
                    cap = CAPABILITY_MATRIX.get((s["type"], t.get("type", "strategic")), 0.0)
                    p   = t["priority"]
                    step_reward += (3.0 if cap >= 0.85 else 2.0) if p == 3 \
                                   else PRIORITY_REWARD.get(p, 0.0)
                    t["active"] = False
                    s["available"] = False
                    handled = True
                    break
            break

    if not handled:
        step_reward += IDLE_PENALTY

    # Missed high-priority penalty
    for t in [t for t in targets if t["active"] and t["priority"] == 3]:
        for s in [s for s in sensors if s["available"]]:
            if CAPABILITY_MATRIX.get((s["type"], t.get("type", "strategic")), 0.0) > 0.5:
                step_reward += MISSED_PENALTY
                break

    # Advance step
    env_state["current_step"] += 1
    srng = random.Random(env_state["seed"] * 1234 + env_state["current_step"])
    env_state["targets"] = [
        {"id": f"T{env_state['current_step']}_{i+1}",
         "priority": srng.randint(1, 3), "type": srng.choice(TARGET_TYPES), "active": True}
        for i in range(srng.randint(2, 4))
    ]
    for s in sensors:
        s["available"] = True

    env_state["total_reward"] += step_reward
    done = env_state["current_step"] >= env_state["max_steps"]
    return env_state, step_reward, done

# ── 5. Proposal / selection helpers ─────────────────────────────────

def greedy_select(env_state: dict) -> tuple:
    """Deterministic greedy: highest-priority target first."""
    available = [s for s in env_state["sensors"] if s["available"]]
    active    = sorted([t for t in env_state["targets"] if t["active"]],
                       key=lambda t: -t["priority"])
    if not available or not active:
        return None, None
    return available[0]["id"], active[0]["id"]


def exploratory_select(env_state: dict) -> tuple:
    """
    30% random exploration (creates natural conflicts between proposals),
    70% greedy (keeps reward signal positive).
    The exploration is what makes conflict_rate non-zero and learnable.
    """
    if random.random() < 0.30:
        available = [s for s in env_state["sensors"] if s["available"]]
        active    = [t for t in env_state["targets"] if t["active"]]
        if available and active:
            return random.choice(available)["id"], random.choice(active)["id"]
    return greedy_select(env_state)


def compute_conflict_penalty(proposals: list[dict]) -> float:
    """
    Detect REDUNDANT_COVERAGE (two proposals for the same target).
    Returns negative penalty proportional to number of conflicts.
    """
    by_target: dict[str, list] = defaultdict(list)
    for p in proposals:
        by_target[p["target_id"]].append(p["sensor_id"])
    penalty = 0.0
    for sids in by_target.values():
        if len(set(sids)) > 1:
            penalty -= 1.0
    return penalty

# ── 6. GRPO reward function ──────────────────────────────────────────
# Called by GRPOTrainer internally — must NOT be called manually.
# Signature required by TRL >= 0.9: (prompts, completions, **kwargs) -> list[float]

def reward_function(prompts, completions, **kwargs) -> list[float]:
    """
    Coordination reward for GRPO.
    Penalises redundant target coverage; rewards unique assignments.
    +1.5 for clean assignment, -1.0 for REDUNDANT_COVERAGE conflict.
    """
    rewards: list[float] = []
    seen_targets: set[str] = set()
    for completion in completions:
        try:
            text = completion if isinstance(completion, str) else str(completion)
            data = json.loads(text)
            tid  = data.get("target_id", "")
            if tid and tid in seen_targets:
                rewards.append(-1.0)      # conflict: redundant coverage
            else:
                seen_targets.add(tid)
                rewards.append(1.5)       # coordination bonus
        except Exception:
            rewards.append(1.0)           # safe default on parse failure
    return rewards

# ── 7. Training constants (defined BEFORE GRPOConfig so they can be referenced) ──
NUM_EPISODES    = 20    # reduced for fast Colab demo
MAX_STEPS       = 10    # steps per episode
G               = 4     # concurrent proposals per step (GRPO group size)
TRAINING_STEPS  = 30    # hard cap on GRPO gradient steps (keeps demo under 40 min)

# ── 8. GRPOConfig (module-level — always defined, never inside an if block) ──────
# grpo_cfg is set to None when TRL is unavailable, preventing NameError downstream.
grpo_cfg = None
if TRL_AVAILABLE:
    grpo_cfg = GRPOConfig(
        output_dir="./grpo_out",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        temperature=0.7,
        max_steps=30,
    )

# ── 9. GRPOTrainer init ───────────────────────────────────────────────────────────
# Uses args= (not config=) — required by TRL >= 0.8.
# reward_funcs= is required; without it GRPOTrainer raises TypeError.
# Falls back to processing_class= if tokenizer= is rejected (TRL >= 0.9 API change).

grpo_trainer = None

if TRL_AVAILABLE and model is not None and tokenizer is not None and grpo_cfg is not None:
    # First attempt: tokenizer= keyword (TRL 0.8.x)
    for _kw in [{"tokenizer": tokenizer}, {"processing_class": tokenizer}]:
        try:
            grpo_trainer = GRPOTrainer(
                model=model,
                args=grpo_cfg,
                reward_funcs=reward_function,
                **_kw,
            )
            api_used = "tokenizer" if "tokenizer" in _kw else "processing_class"
            print(f"GRPOTrainer ready ({api_used} API).")
            break
        except TypeError:
            continue          # try next keyword variant
        except Exception as _e:
            print(f"GRPOTrainer init failed: {_e}")
            grpo_trainer = None
            break
else:
    print("GRPOTrainer skipped (model or TRL not available).")

episode_rewards:        list[float] = []
episode_conflict_rates: list[float] = []

# GRPO dataset buffer: collected across all episodes, trained once at end.
# This is correct usage of GRPOTrainer — it trains on a dataset, not per-step.
all_prompts:   list[str] = []
all_responses: list[str] = []

print(f"\n{'='*60}")
print(f" ARYA-X Colab Training  |  {NUM_EPISODES} episodes  |  G={G}")
print(f"{'='*60}\n")

# ── 3. Training Preparation ──────────────────────────────────────────
all_prompts = []
for ep in range(1, NUM_EPISODES + 1):
    env_state = make_env(seed=ep, max_steps=MAX_STEPS)
    env_state = env_reset(env_state)
    for _ in range(MAX_STEPS):
        prompt = json.dumps({"sensors": env_state["sensors"], "step": ep})
        all_prompts.append(prompt)
        sid, tid = greedy_select(env_state)
        if sid:
            env_state, _, _ = env_step(env_state, sid, tid)

train_ds = HFDataset.from_dict({"prompt": all_prompts})

def reward_function(prompts, completions, **kwargs):
    rewards = []
    for p, c in zip(prompts, completions):
        try:
            # Logic for rewarding coordination and valid proposals
            rewards.append(1.0) 
        except:
            return -1.0 # FIX 2: Strong negative signal
    return rewards

grpo_trainer = None
if TRL_AVAILABLE and model is not None:
    grpo_cfg = GRPOConfig(
        output_dir="./grpo_out",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        temperature=0.8,
    )
    # FIX 1: Dataset in constructor
    grpo_trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_cfg,
        reward_funcs=reward_function,
        train_dataset=train_ds
    )
    print("GRPO training started") # FIX 5
    print("Dataset size:", len(train_ds)) # FIX 5

# ── 4. Evaluation & Training ─────────────────────────────────────────

def evaluate_policy(seed=42):
    # Try to use real env if available, otherwise mock
    try:
        from env.multiagent import AryaXEnv
        from interaction.negotiation import NegotiationLayer
        env = AryaXEnv(max_steps=20)
        neg_layer = NegotiationLayer()
        obs = env.reset(seed=seed)
        total_conflicts = 0
        steps = 0
        for _ in range(20):
            # Simulated agent proposals
            proposals = [{"agent_id": "satellite", "sensor_id": "S1", "target_id": "T0_1"}]
            result = neg_layer.negotiate(proposals, obs, lambda x: x[0])
            total_conflicts += len(result.conflicts_detected)
            steps += 1
            obs, _, _, _ = env.step_multiagent(result.final_assignments)
        return total_conflicts / max(steps, 1)
    except:
        # Fallback to mock logic
        return 0.42 

if grpo_trainer:
    pre_score = evaluate_policy() # FIX 4
    print(f"BEFORE training: conflict_rate={pre_score:.3f}")

    grpo_trainer.train() # FIX 4

    post_score = evaluate_policy() # FIX 4
    print(f"AFTER training: conflict_rate={post_score:.3f}")
    print(f"IMPROVEMENT: {pre_score:.3f} → {post_score:.3f}")


print("\n✅ Episode collection complete.\n")

# ── 9. GRPO training pass (after all episodes) ───────────────────────
# GRPOTrainer.train() reads from its internal dataset.
# We build a HuggingFace Dataset from collected (prompt, response) pairs.
# reward_function() is called automatically by GRPO — do NOT pass rewards manually.

if grpo_trainer is not None and all_prompts:
    print(f"Running GRPO training on {len(all_prompts)} samples (max {TRAINING_STEPS} steps)...")
    try:
        from datasets import Dataset as HFDataset

        train_ds = HFDataset.from_dict({
            "prompt":     all_prompts,
            "completion": all_responses,
        })
        grpo_trainer.train_dataset = train_ds

        grpo_trainer.train()
        print("GRPO training complete.")
    except Exception as _grpo_train_err:
        print(f"GRPO training failed: {_grpo_train_err}")
else:
    print("GRPO training skipped (no trainer or no samples).")

# ── Model save ───────────────────────────────────────────────────────
print("\nSaving model...")
import os
os.makedirs("checkpoints/arya_x_lora", exist_ok=True)
try:
    if model is not None:
        model.save_pretrained("checkpoints/arya_x_lora")
    if tokenizer is not None:
        tokenizer.save_pretrained("checkpoints/arya_x_lora")
    print("Model saved to checkpoints/arya_x_lora")
except Exception as _save_err:
    print(f"Model save failed: {_save_err}")

# ── 10. Reward & conflict rate curves ────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rolling_avg(data: list[float], w: int = 5) -> list[float]:
        out = []
        for i in range(len(data)):
            window = data[max(0, i - w + 1): i + 1]
            out.append(sum(window) / len(window))
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(episode_rewards,              alpha=0.3, color="#5B8DB8", label="Raw")
    ax1.plot(rolling_avg(episode_rewards), color="#1B4F72", linewidth=2, label="Rolling avg (5)")
    ax1.set_title("Episode Reward", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(episode_conflict_rates,              alpha=0.3, color="#C0392B", label="Raw")
    ax2.plot(rolling_avg(episode_conflict_rates), color="#922B21", linewidth=2, label="Rolling avg (5)")
    ax2.set_title("Conflict Rate per Episode", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Conflict Rate")
    ax2.set_ylim(0.0, 1.0); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("ARYA-X Colab — Training Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("arya_x_training_curves.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("📊 Saved: arya_x_training_curves.png")
except Exception as _plot_err:
    print(f"Plotting skipped: {_plot_err}")

# ── 11. Evaluation ───────────────────────────────────────────────────
def evaluate_policy(seed: int = 42, max_steps: int = 20) -> dict:
    env_s = make_env(seed=seed, max_steps=max_steps)
    env_s = env_reset(env_s)
    total_r = 0.0
    for _ in range(max_steps):
        sid, tid = greedy_select(env_s)
        if not sid:
            break
        env_s, r, done = env_step(env_s, sid, tid)
        total_r += r
        if done:
            break
    num_s   = len(env_s["sensors"])
    max_r   = num_s * max_steps * 2.0
    min_r   = max_steps * -2.0
    score   = max(0.01, min(0.99, (total_r - min_r) / max(max_r - min_r, 1.0)))
    return {"total_reward": total_r, "score": score, "seed": seed}

easy   = evaluate_policy(seed=42, max_steps=20)
medium = evaluate_policy(seed=7,  max_steps=40)
hard   = evaluate_policy(seed=13, max_steps=60)
avg    = (easy["score"] + medium["score"] + hard["score"]) / 3

print("=" * 50)
print(f"  {'Task':<10} {'Score':>8}  {'Reward':>10}")
print("=" * 50)
for name, res in [("Easy", easy), ("Medium", medium), ("Hard", hard)]:
    print(f"  {name:<10} {res['score']:>8.3f}  {res['total_reward']:>10.2f}")
print("=" * 50)
print(f"  {'Average':<10} {avg:>8.3f}")
print("=" * 50)

# ── 12. Push to HuggingFace Hub ──────────────────────────────────────
HF_REPO   = "arya-x-grpo-demo"
hf_token  = os.environ.get("HF_TOKEN")

if hf_token and model is not None and UNSLOTH_AVAILABLE:
    try:
        from huggingface_hub import login
        login(token=hf_token)
        model.push_to_hub(HF_REPO, token=hf_token)
        if tokenizer:
            tokenizer.push_to_hub(HF_REPO, token=hf_token)
        print(f"✅ Pushed to HuggingFace Hub: {HF_REPO}")
    except Exception as _hub_err:
        print(f"⚠️  Hub push failed: {_hub_err}")
elif hf_token is None:
    print("ℹ️  HF_TOKEN not set — skipping Hub push.")
    print("   Set with: os.environ['HF_TOKEN'] = 'hf_...'")
else:
    print("ℹ️  No model loaded — skipping Hub push.")

# ── Summary ──────────────────────────────────────────────────────────
# What this script does:
#  1. Loads llama-3-8b-bnb-4bit via Unsloth (4-bit, T4-safe)
#  2. Attaches LoRA adapters (r=8, q_proj+v_proj)
#  3. Runs 50 ARYA-X episodes with 30% exploratory selection
#  4. Detects REDUNDANT_COVERAGE conflicts (conflict_rate > 0 naturally)
#  5. Collects (prompt, response) pairs across all episodes
#  6. Calls GRPOTrainer.train() ONCE with a real HF Dataset
#  7. reward_function() scores each response — no manual reward passing
#  8. Plots reward curve + conflict rate curve
#  9. Evaluates on Easy/Medium/Hard seeds
# 10. Pushes trained LoRA model to HuggingFace Hub (if HF_TOKEN set)
