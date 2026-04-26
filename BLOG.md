# Arya-X: Teaching LLMs to Coordinate Under Pressure

## The Problem We Set Out to Solve

Imagine a military command centre. Threats are spawning across a region in real time — missiles, border incursions, airspace violations. You have a limited fleet of sensors: satellites for wide-area coverage, drones for tactical precision, radar for fast airspace detection. Every sensor needs to be assigned to a threat, every step, without overlap, without gaps, and without two agents accidentally claiming the same resource.

This is not a toy problem. It is a real coordination challenge — and it is exactly the kind of task that exposes the limits of a single LLM acting alone.

A single agent can allocate sensors greedily. But what happens when four specialised agents — each with partial visibility, each with different sensor affinities — must negotiate assignments simultaneously, detect conflicts, and resolve them in real time? That is what Arya-X is built to train.

---

## What Arya-X Is

Arya-X is an [OpenEnv](https://huggingface.co/openenv)-compatible multi-agent reinforcement learning environment for ISR (Intelligence, Surveillance, and Reconnaissance) sensor allocation.

At every timestep, new threats spawn with priorities (HIGH / MED / LOW). Four agents — Satellite, Drone, Radar, and Command — each observe the environment and independently propose sensor-to-target assignments. These proposals are fed into a **NegotiationLayer** that detects conflicts, resolves them through a 3-pass arbitration pipeline, and computes per-agent rewards.

The environment is live at: **https://aryaxrl-aryax.hf.space**

---

## Why This Tests Something Real

Most multi-agent environments test cooperation in symmetric settings — all agents see the same thing, all agents have the same capabilities. Arya-X is deliberately asymmetric:

- The **Satellite agent** can only claim satellite sensors, and is best suited for strategic wide-area targets (P3).
- The **Drone agent** can only claim drone sensors, optimised for kinetic tactical targets (P2).
- The **Radar agent** can only claim radar sensors, best for fast airspace threats (P1).
- The **Command agent** can claim any sensor and has override authority — but using it costs nothing in reward, so the system learns when to invoke it and when not to.

Each agent receives a partial observation scoped to its sensor type. No agent sees the full picture except Command. This creates genuine partial observability and forces the agents to develop implicit coordination — not just greedy local optimisation.

The conflict system makes this concrete. If two agents claim the same sensor (`OVER_ASSIGNMENT`) or both assign to the same target (`REDUNDANT_COVERAGE`), both are penalised −0.5. If a HIGH-priority threat goes uncovered (`MISSED_PRIORITY_3`), it is logged. The only way to score well is to coordinate.

---

## The Architecture

```
Four Agents (Satellite, Drone, Radar, Command)
        ↓ proposals
  NegotiationLayer
        ↓
  ConflictDetector  →  REDUNDANT_COVERAGE, OVER_ASSIGNMENT, MISSED_PRIORITY_3
        ↓
  ConflictResolver  →  Pass 1: Priority | Pass 2: Capability | Pass 3: Command Override
        ↓
  RewardEngine  →  per-agent rewards, conflict_rate tracking
```

The **ConflictResolver** runs three passes:
1. When two agents claim the same sensor, keep the one targeting the higher-priority threat.
2. Sort remaining assignments by target priority × sensor capability score. Discard duplicates.
3. Honour all Command agent proposals not already covered, evicting conflicting claims.

This means the environment has a built-in arbitration mechanism that the agents must learn to work with — not around.

---

## How We Trained

We used **Group Relative Policy Optimization (GRPO)** with **Unsloth** for parameter-efficient fine-tuning via LoRA, training on `Qwen2.5-0.5B-Instruct` as the base model.

The training setup:
- 500 prompts per epoch, each describing a sensor allocation scenario with randomised sensor types, target priorities, and target types
- A custom reward function (`arya_reward_func`) that scores each completion based on JSON validity, sensor/target ID correctness, capability matrix alignment, and target priority
- LoRA adapters with `r=4`, `lora_alpha=8`, targeting `q_proj` and `v_proj`
- 500 training steps, batch size 8, gradient accumulation 2, learning rate 2e-5

The reward function captures multiple independent signals:

```python
# High capability match on a P3 target → +3.0 (or +2.0 if lower capability)
# P2 target → +1.0, P1 target → +0.5
# Invalid JSON or wrong IDs → -2.0
# Valid but wrong sensor/target → -1.0
# High capability bonus → +0.5 additional
```

This multi-signal design prevents reward hacking — an agent cannot score well by producing valid JSON alone; it must also pick the right sensor for the right target.

---

## What the Training Showed

Over 500 steps of GRPO training:

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| Conflict Rate | ~0.375 | ~0.1875 |
| Avg Reward per step | ~0.96 | ~1.75 |
| Coordination Score | ~0.625 | ~0.8125 |

The reward curve shows a clear upward trend from step 0 to step 500, with the model learning to:
- Prefer high-capability sensor-target pairings (satellite → strategic, drone → kinetic, radar → airspace)
- Avoid redundant coverage by not double-assigning targets
- Prioritise P3 threats over P2 and P1

![Training Reward Curve](./checkpoints/arya_x_lora/reward_curve.png)

The conflict rate dropped organically — we did not explicitly penalise conflicts in the single-agent training loop. The model learned specialisation from the capability matrix reward signal alone, which then translated to lower conflict rates in the multi-agent setting at inference time.

---

## The Dashboard

The live dashboard at **https://aryaxrl-aryax.hf.space** lets you interact with the environment directly:

- Switch between **Single-Agent** and **Multi-Agent** modes
- Watch each agent draw colour-coded arcs to their assigned targets in real time
  - 🔵 Satellite · 🟢 Drone · 🟠 Radar · 🟣 Command
- See conflict indicators pulse on contested targets
- Track per-agent cumulative rewards and live conflict rate
- Run a full episode with **RUN FULL EPISODE** and get an episode summary with training impact comparison

The environment exposes a full REST API (`/reset_multi`, `/step_multi`, `/auto_multi`) so it can be used as a training target for any external agent loop.

---

## Training Script

The full GRPO training notebook is available here:
📓 [AryaX_train_colab.ipynb](./AryaX_train_colab.ipynb)

It runs end-to-end in Google Colab (T4 GPU, free tier) in under 30 minutes and saves the LoRA adapters to Google Drive.

---

## Why This Matters

The capability gap Arya-X targets is real: LLMs are poor at **resource allocation under partial observability with competing agents**. They tend to be greedy, ignore capability constraints, and produce conflicting assignments when acting in parallel.

Training on Arya-X directly addresses this by:
- Forcing the model to reason about sensor-target capability alignment
- Penalising redundant and conflicting assignments
- Rewarding priority-aware coverage

The environment is domain-agnostic at its core — the same architecture applies to compute allocation, bandwidth scheduling, logistics routing, or any multi-agent resource contention problem. ISR is the domain; coordination under constraint is the capability.

---

*Built for the OpenEnv Hackathon — Arya-X, Round 2.*
