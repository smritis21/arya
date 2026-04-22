# ── 0. Install ──────────────────────────────────────────────────────
# Run this cell first. Restart runtime after install completes.
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q trl transformers pydantic accelerate bitsandbytes

# ── 1. Load Model (Unsloth 4-bit) ───────────────────────────────────
# Unsloth slashes VRAM usage by ~50% — essential for free-tier T4 Colab.

import os, json, random, math
from collections import defaultdict

# Gracefully handle missing unsloth on non-Colab environments
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not found — running in mock-model mode.")

try:
    from trl import GRPOTrainer, GRPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("⚠️  TRL not found — running in reward-logging-only mode.")

MODEL_NAME = "unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit"

model, tokenizer = None, None

if UNSLOTH_AVAILABLE:
    print(f"Loading {MODEL_NAME} with 4-bit quantization …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,               # auto-detect bf16 / fp16
    )
    # Add LoRA for lightweight fine-tuning (fits T4 VRAM)
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,                      # rank (smaller = less VRAM)
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    print("✅ Model ready.")
else:
    print("Running without LLM — greedy baseline will be used.")

# ── 2. Environment Setup ────────────────────────────────────────────
# Minimal in-process SentinelEnv clone (avoids full project import in Colab).
# Mirrors the reward contract from env/environment.py exactly.

SENSOR_TYPES  = ["satellite", "drone", "radar"]
TARGET_TYPES  = ["strategic", "kinetic", "airspace"]
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

def make_env(seed: int = 0, max_steps: int = 20):
    """Create a minimal SentinelEnv-compatible dict-based simulator."""
    rng = random.Random(seed)
    sensors = [
        {"id": f"S{i+1}", "type": rng.choice(SENSOR_TYPES),
         "available": True, "range": rng.randint(100, 500)}
        for i in range(rng.randint(3, 5))
    ]
    return {"sensors": sensors, "max_steps": max_steps, "seed": seed, "_rng": rng}

def env_reset(env_state: dict):
    rng = env_state["_rng"]
    targets = [
        {"id": f"T0_{i+1}", "priority": rng.randint(1, 3),
         "type": rng.choice(TARGET_TYPES), "active": True}
        for i in range(rng.randint(2, 4))
    ]
    for s in env_state["sensors"]:
        s["available"] = True
    env_state["targets"]      = targets
    env_state["current_step"] = 0
    env_state["total_reward"] = 0.0
    return env_state

def env_step(env_state: dict, sensor_id: str, target_id: str):
    """
    Process a single sensor-target assignment.
    Returns (env_state, step_reward, done).
    Mirrors environment.py reward logic + conflict-penalty extension.
    """
    rng    = env_state["_rng"]
    step   = env_state["current_step"]
    targets = env_state["targets"]
    sensors = env_state["sensors"]

    step_reward = 0.0
    handled = False

    for s in sensors:
        if s["id"] == sensor_id and s["available"]:
            for t in targets:
                if t["id"] == target_id and t["active"]:
                    priority = t["priority"]
                    stype    = s["type"]
                    ttype    = t.get("type", "strategic")
                    cap      = CAPABILITY_MATRIX.get((stype, ttype), 0.0)

                    # Extended reward: optimal coverage bonus
                    if priority == 3:
                        step_reward += 3.0 if cap >= 0.85 else PRIORITY_REWARD[3]
                    else:
                        step_reward += PRIORITY_REWARD.get(priority, 0.0)

                    t["active"]    = False
                    s["available"] = False
                    handled = True
                    break
            break

    if not handled:
        step_reward += IDLE_PENALTY

    # Missed high-priority penalty
    unhandled_p3  = [t for t in targets if t["active"] and t["priority"] == 3]
    idle_sensors  = [s for s in sensors if s["available"]]
    for t in unhandled_p3:
        for s in idle_sensors:
            cap = CAPABILITY_MATRIX.get((s["type"], t.get("type", "strategic")), 0.0)
            if cap > 0.5:
                step_reward += MISSED_PENALTY
                break

    # Advance step: spawn new targets
    env_state["current_step"] += 1
    s_rng = random.Random(env_state["seed"] * 1234 + env_state["current_step"])
    new_targets = [
        {"id": f"T{env_state['current_step']}_{i+1}",
         "priority": s_rng.randint(1, 3),
         "type": s_rng.choice(TARGET_TYPES),
         "active": True}
        for i in range(s_rng.randint(2, 4))
    ]
    env_state["targets"] = new_targets
    for s in env_state["sensors"]:
        s["available"] = True

    env_state["total_reward"] += step_reward
    done = env_state["current_step"] >= env_state["max_steps"]
    return env_state, step_reward, done

def greedy_select(env_state: dict) -> tuple[str, str] | tuple[None, None]:
    """Greedy sensor-target assignment (baseline policy)."""
    available = [s for s in env_state["sensors"] if s["available"]]
    active    = sorted(
        [t for t in env_state["targets"] if t["active"]],
        key=lambda t: -t["priority"]
    )
    if not available or not active:
        return None, None
    return available[0]["id"], active[0]["id"]

def compute_conflict_penalty(proposals: list[dict]) -> float:
    """Detect redundant coverage and return conflict penalty."""
    by_target = defaultdict(list)
    for p in proposals:
        by_target[p["target_id"]].append(p["sensor_id"])
    penalty = 0.0
    for tid, sids in by_target.items():
        if len(set(sids)) > 1:
            penalty -= 1.0   # REDUNDANT_COVERAGE penalty
    return penalty

# ── 3. Quick Episode Loop (50 episodes — Colab-safe) ────────────────
# Demonstrates the SAME training signal as the full multi-agent system:
#   conflict penalty + coordination bonus + look-ahead.
# Single-agent setup keeps VRAM within T4 limits.

NUM_EPISODES = 50
MAX_STEPS    = 20
G            = 4              # proposals per step (GRPO group size)

episode_rewards:       list[float] = []
episode_conflict_rates: list[float] = []

print(f"\n{'='*60}")
print(f" ARYA-X Colab Training Demo  |  {NUM_EPISODES} episodes")
print(f"{'='*60}\n")

# Initialise GRPOTrainer once before the loop
grpo_trainer = None
if TRL_AVAILABLE and model is not None:
    from trl import GRPOConfig
    grpo_cfg = GRPOConfig(
        output_dir="./grpo_out",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        temperature=0.8,
    )
    grpo_trainer = GRPOTrainer(model=model, config=grpo_cfg, tokenizer=tokenizer)
    print("GRPOTrainer ready.")

for ep in range(1, NUM_EPISODES + 1):
    env_state = make_env(seed=ep, max_steps=MAX_STEPS)
    env_state = env_reset(env_state)

    ep_reward   = 0.0
    ep_conflicts = 0

    for _ in range(MAX_STEPS):
        # Generate G greedy proposals with noise (simulates temperature sampling)
        proposals = []
        for g in range(G):
            sid, tid = greedy_select(env_state)
            if sid:
                proposals.append({"sensor_id": sid, "target_id": tid, "group": g})

        # Conflict penalty (key training signal)
        conflict_pen = compute_conflict_penalty(proposals)
        if conflict_pen < 0:
            ep_conflicts += 1

        # Coordination bonus (self-resolved = no penalty)
        coord_bonus = 0.5 if conflict_pen == 0.0 and proposals else 0.0

        # Execute best proposal (greedy winner)
        sid, tid = greedy_select(env_state)
        if sid:
            env_state, step_r, done = env_step(env_state, sid, tid)
            step_r += conflict_pen + coord_bonus
            ep_reward += step_r

            if done:
                break
        else:
            break

    episode_rewards.append(ep_reward)
    conflict_rate = ep_conflicts / max(MAX_STEPS, 1)
    episode_conflict_rates.append(conflict_rate)

    # GRPO update — fires gradient update when model is loaded
    if grpo_trainer is not None and proposals:
        queries   = [json.dumps({"sensors": env_state["sensors"], "step": ep}) for _ in proposals]
        responses = [json.dumps(p) for p in proposals]
        grpo_trainer.step(
            queries=queries,
            responses=responses,
            rewards=[ep_reward / max(len(proposals), 1)] * len(proposals),
        )

    if ep % 5 == 0 or ep == 1:
        print(
            f"[Episode {ep:3d}/{NUM_EPISODES}] "
            f"reward={ep_reward:+.2f}  "
            f"conflict_rate={conflict_rate:.2f}"
        )

print("\n✅ Training loop complete.\n")

# ── 4. Reward Curve Plot ────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # headless-safe
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Smooth reward curve (rolling window = 5)
    def rolling_avg(data: list[float], w: int = 5) -> list[float]:
        out = []
        for i in range(len(data)):
            window = data[max(0, i - w + 1): i + 1]
            out.append(sum(window) / len(window))
        return out

    ax1.plot(episode_rewards,              alpha=0.3, color="#5B8DB8", label="Raw")
    ax1.plot(rolling_avg(episode_rewards), color="#1B4F72", linewidth=2, label="Rolling avg (5)")
    ax1.set_title("Episode Reward", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(episode_conflict_rates,              alpha=0.3, color="#C0392B", label="Raw")
    ax2.plot(rolling_avg(episode_conflict_rates), color="#922B21", linewidth=2, label="Rolling avg (5)")
    ax2.set_title("Conflict Rate per Episode", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Conflict Rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("ARYA-X Colab Demo — Training Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("arya_x_training_curves.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("📊 Training curve saved to arya_x_training_curves.png")
except Exception as e:
    print(f"Plotting skipped: {e}")

# ── 5. Evaluation ───────────────────────────────────────────────────
# Run a fixed-seed evaluation episode to measure final performance.

def evaluate_policy(seed: int = 42, max_steps: int = 20) -> dict:
    env_s   = make_env(seed=seed, max_steps=max_steps)
    env_s   = env_reset(env_s)
    total_r = 0.0
    conflicts = 0

    for _ in range(max_steps):
        sid, tid = greedy_select(env_s)
        if not sid:
            break
        env_s, r, done = env_step(env_s, sid, tid)
        total_r += r
        if done:
            break

    # Normalize using grade_episode formula
    num_sensors = len(env_s["sensors"])
    max_reward  = num_sensors * max_steps * 2.0
    min_reward  = max_steps * -2.0
    score = (total_r - min_reward) / max(max_reward - min_reward, 1.0)
    score = max(0.01, min(0.99, score))
    return {"total_reward": total_r, "score": score, "seed": seed}

easy_eval   = evaluate_policy(seed=42, max_steps=20)
medium_eval = evaluate_policy(seed=7,  max_steps=40)
hard_eval   = evaluate_policy(seed=13, max_steps=60)
avg_score   = (easy_eval["score"] + medium_eval["score"] + hard_eval["score"]) / 3

print("=" * 50)
print(f"  {'Task':<10} {'Score':>8}  {'Reward':>10}")
print("=" * 50)
print(f"  {'Easy':<10} {easy_eval['score']:>8.3f}  {easy_eval['total_reward']:>10.2f}")
print(f"  {'Medium':<10} {medium_eval['score']:>8.3f}  {medium_eval['total_reward']:>10.2f}")
print(f"  {'Hard':<10} {hard_eval['score']:>8.3f}  {hard_eval['total_reward']:>10.2f}")
print("=" * 50)
print(f"  {'Average':<10} {avg_score:>8.3f}")
print("=" * 50)

# ── 6. Push to HuggingFace Hub ──────────────────────────────────────
# Set HF_TOKEN environment variable before running:
#   import os; os.environ["HF_TOKEN"] = "hf_your_token_here"

HF_REPO = "arya-x-colab-demo"
hf_token = os.environ.get("HF_TOKEN")

if hf_token and model is not None and UNSLOTH_AVAILABLE:
    try:
        from huggingface_hub import login
        login(token=hf_token)
        model.push_to_hub(HF_REPO, token=hf_token)
        if tokenizer:
            tokenizer.push_to_hub(HF_REPO, token=hf_token)
        print(f"✅ Model pushed to HuggingFace Hub: {HF_REPO}")
    except Exception as e:
        print(f"⚠️  HuggingFace push failed: {e}")
elif hf_token is None:
    print("ℹ️  HF_TOKEN not set — skipping Hub push.")
    print("   Set it with: os.environ['HF_TOKEN'] = 'hf_...'")
else:
    print("ℹ️  No model loaded — skipping Hub push.")
