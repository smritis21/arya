"""
ARYA-X GRPO Training Script
Run on HF Spaces A10G: ~15 min for 200 steps
Saves LoRA adapter to HF Hub repo specified by HF_REPO env var.
"""
import os, json, random, torch
from collections import defaultdict

# ── Model ─────────────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
print(f"Loading {MODEL_NAME}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=512,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
print("✅ Model ready.")

# ── Environment ───────────────────────────────────────────────────────────────
SENSOR_TYPES    = ["satellite", "drone", "radar"]
TARGET_TYPES    = ["strategic", "kinetic", "airspace"]
CAPABILITY_MATRIX = {
    ("satellite", "strategic"): 0.95, ("satellite", "kinetic"): 0.40,
    ("satellite", "airspace"):  0.60, ("drone",     "kinetic"): 0.95,
    ("drone",     "strategic"): 0.30, ("drone",     "airspace"): 0.50,
    ("radar",     "airspace"):  0.95, ("radar",     "kinetic"):  0.65,
    ("radar",     "strategic"): 0.45,
}

def make_env(seed=0):
    rng = random.Random(seed)
    sensors = [
        {"id": f"S{i+1}", "type": rng.choice(SENSOR_TYPES),
         "available": True, "range": rng.randint(100, 500)}
        for i in range(rng.randint(3, 5))
    ]
    return {"sensors": sensors, "seed": seed, "_rng": rng}

def env_reset(env_state):
    rng = env_state["_rng"]
    targets = [
        {"id": f"T0_{i+1}", "priority": rng.randint(1, 3),
         "type": rng.choice(TARGET_TYPES), "active": True}
        for i in range(rng.randint(2, 4))
    ]
    for s in env_state["sensors"]:
        s["available"] = True
    env_state.update({"targets": targets, "current_step": 0, "total_reward": 0.0})
    return env_state

# ── Dataset ───────────────────────────────────────────────────────────────────
import datasets as hf_datasets
from trl import GRPOTrainer, GRPOConfig

def make_prompt(seed):
    rng = random.Random(seed)
    sensors = [
        {"id": f"S{i+1}", "type": rng.choice(SENSOR_TYPES), "range": rng.randint(100, 500)}
        for i in range(rng.randint(3, 5))
    ]
    targets = [
        {"id": f"T0_{i+1}", "priority": rng.randint(1, 3), "type": rng.choice(TARGET_TYPES)}
        for i in range(rng.randint(2, 4))
    ]
    sensor_lines = "\n".join(f"  {s['id']} type={s['type']} range={s['range']}km" for s in sensors)
    target_lines = "\n".join(f"  {t['id']} priority={t['priority']} type={t['type']}" for t in targets)
    sensor_ids   = [s["id"] for s in sensors]
    target_ids   = [t["id"] for t in targets]
    prompt = (
        f"You are an ISR sensor allocation agent.\n"
        f"Assign ONE sensor to ONE target. Priority 3=HIGH, 2=MED, 1=LOW.\n"
        f"Cover the highest priority target first.\n\n"
        f"Available sensors:\n{sensor_lines}\n\n"
        f"Active targets:\n{target_lines}\n\n"
        f"Respond ONLY with valid JSON using these exact IDs:\n"
        f"sensors={sensor_ids}  targets={target_ids}\n\n"
        f"{{\"sensor_id\": \"S1\", \"target_id\": \"T0_1\"}}"
    )
    return prompt, sensor_ids, target_ids, targets

prompt_list   = []
metadata_list = []
for seed in range(500):
    prompt, sensor_ids, target_ids, targets = make_prompt(seed)
    prompt_list.append(prompt)
    metadata_list.append({"sensor_ids": sensor_ids, "target_ids": target_ids,
                           "targets": targets, "seed": seed})

train_dataset = hf_datasets.Dataset.from_dict({"prompt": prompt_list})
print(f"✅ Dataset ready: {len(train_dataset)} prompts")

# ── Reward function ───────────────────────────────────────────────────────────
NUM_GENERATIONS = 8

def arya_reward_func(completions, prompts=None, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        try:
            prompt_idx = i // NUM_GENERATIONS
            meta = metadata_list[prompt_idx % len(metadata_list)]

            text  = completion if isinstance(completion, str) else str(completion)
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start == -1 or end == 0:
                rewards.append(-2.0); continue

            data      = json.loads(text[start:end])
            sensor_id = data.get("sensor_id", "")
            target_id = data.get("target_id", "")

            if sensor_id not in meta["sensor_ids"] or target_id not in meta["target_ids"]:
                rewards.append(-1.0); continue

            target = next((t for t in meta["targets"] if t["id"] == target_id), None)
            env_s  = env_reset(make_env(seed=meta["seed"]))
            sensor = next((s for s in env_s["sensors"] if s["id"] == sensor_id), None)
            if not target or not sensor:
                rewards.append(-1.0); continue

            cap = CAPABILITY_MATRIX.get((sensor["type"], target.get("type", "strategic")), 0.0)
            p   = target["priority"]
            reward = (3.0 if cap >= 0.85 else 2.0) if p == 3 else {2: 1.0, 1: 0.5}.get(p, 0.0)
            if cap >= 0.85:
                reward += 0.5
            rewards.append(float(reward))
        except Exception:
            rewards.append(-2.0)
    return rewards

# Sanity check
_test = arya_reward_func(['{"sensor_id": "S1", "target_id": "T0_1"}'] * 8)
print(f"✅ Reward sanity check: {_test}")

# ── Train ─────────────────────────────────────────────────────────────────────
grpo_args = GRPOConfig(
    output_dir="./grpo_out",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_steps=200,
    learning_rate=2e-5,
    warmup_steps=10,
    logging_steps=10,
    save_steps=50,
    report_to="none",
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    tokenizer=tokenizer,
    reward_funcs=arya_reward_func,
    train_dataset=train_dataset,
)

print(f"\nStarting training — 200 steps on A10G (~15 min)...\n")
train_result = grpo_trainer.train()
print(f"\n✅ Training complete. Loss: {train_result.training_loss:.4f}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
FastLanguageModel.for_inference(model)

total_reward, conflicts, valid_outputs = 0.0, 0, 0
for seed in range(50):
    prompt, sensor_ids, target_ids, targets = make_prompt(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    new_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text    = tokenizer.decode(new_ids, skip_special_tokens=True)
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        data      = json.loads(text[start:end])
        sensor_id = data.get("sensor_id", "")
        target_id = data.get("target_id", "")
        if sensor_id not in sensor_ids or target_id not in target_ids:
            conflicts += 1; total_reward -= 1.0; continue
        target = next((t for t in targets if t["id"] == target_id), None)
        env_s  = env_reset(make_env(seed=seed))
        sensor = next((s for s in env_s["sensors"] if s["id"] == sensor_id), None)
        if target and sensor:
            cap    = CAPABILITY_MATRIX.get((sensor["type"], target.get("type", "strategic")), 0.0)
            p      = target["priority"]
            reward = (3.0 if cap >= 0.85 else 2.0) if p == 3 else {2: 1.0, 1: 0.5}.get(p, 0.0)
            total_reward += reward; valid_outputs += 1
        else:
            conflicts += 1
    except Exception:
        conflicts += 1; total_reward -= 2.0

results = {
    "avg_reward":         round(total_reward / 50, 3),
    "conflict_rate":      round(conflicts / 50, 3),
    "coordination_score": round(1.0 - conflicts / 50, 3),
    "valid_outputs":      valid_outputs,
}
print(f"\n{'='*45}")
print(f"  TRAINED MODEL RESULTS")
print(f"{'='*45}")
for k, v in results.items():
    print(f"  {k:<22}: {v}")
print(f"{'='*45}")

# ── Save + push to Hub ────────────────────────────────────────────────────────
os.makedirs("checkpoints/arya_x_lora", exist_ok=True)
model.save_pretrained("checkpoints/arya_x_lora")
tokenizer.save_pretrained("checkpoints/arya_x_lora")

metrics = {**results, "training_loss": train_result.training_loss,
           "steps": 200, "model": "Llama-3.2-3B-Instruct", "method": "GRPO + LoRA r=4"}
os.makedirs("logs", exist_ok=True)
with open("logs/training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("✅ Saved: logs/training_metrics.json")

HF_REPO  = os.environ.get("HF_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_REPO and HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"✅ Pushed to Hub: {HF_REPO}")
else:
    print("ℹ️  Set HF_REPO and HF_TOKEN env vars to push to Hub.")
