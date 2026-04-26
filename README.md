# Arya-X — Multi-Agent ISR Sensor Allocation

> *Four AI agents. A limited sensor fleet. Threats spawning every step. No communication allowed. Can they stop fighting over the same sensors and start covering the skies?*

---

🚀 **Live Demo** → [https://sm21s-sentinel-env.hf.space/](https://sm21s-sentinel-env.hf.space/)
📦 **HF Space** → [huggingface.co/spaces/sm21s/sentinel-env](https://huggingface.co/spaces/sm21s/sentinel-env)
🎓 **Training Notebook** → [Open in Colab](https://colab.research.google.com/drive/1WWX5skZToqne_rVulpE5vvR-lH_VK2gC?usp=sharing)
🧠 **Trained LoRA Adapter** → [checkpoints/arya_x_lora/](./checkpoints/arya_x_lora/)
📊 **Reward Curve** → [reward_curve.png](./checkpoints/arya_x_lora/reward_curve.png)
📝 **Full Technical Blog** → [BLOG.md](./BLOG.md)
🏗️ **OpenEnv Spec** → [openenv.yaml](./openenv.yaml)

---

## The Problem

Picture a military command centre. A region to monitor. Threats incoming — some low priority, some critical. A limited fleet of sensors: satellites for wide-area coverage, drones for kinetic targets, radar for airspace.

Here's what breaks every naive system: **multiple agents are trying to assign those sensors at the same time.**

Satellite agent sees a P3 (critical) threat and claims Sensor S1. Drone agent sees the same threat and also claims S1. Radar piles on. Result: one threat gets triple coverage, two others go completely unmonitored, and a high-priority target that nobody noticed just escalated.

This is the **OVER_ASSIGNMENT problem**. It happens in real ISR (Intelligence, Surveillance, and Reconnaissance) operations. And it's the exact problem Arya-X is built to solve.

The capability gap: **no existing open RL environment models multi-agent sensor allocation with conflict detection, capability-weighted rewards, and adaptive curriculum training.** Arya-X fills that gap.

---

## From SentinelEnv to Arya-X

This started as **SentinelEnv** — a single-agent baseline where one policy assigned all sensors to all targets. It worked. It was boring.

The real question was: *what happens when you split that into four specialised agents — each with their own sensor affinity, priority focus, and partial view of the world — and make them negotiate every single step?*

That became **Arya-X**. Built for the OpenEnv Hackathon Round 2 (Meta × Hugging Face × PyTorch, Bangalore 2026).

---

## What the Agent Sees, Does, and Gets Rewarded For

### The Observation (Partial, Noisy, Intentionally Incomplete)

No agent sees the full picture. This is by design.

| Agent | Sensors Visible | Targets Visible | Priority Noise |
|---|---|---|---|
| Satellite | Satellite sensors only | All targets | ±10% |
| Drone | Drone sensors only | Kinetic + strategic | ±5% |
| Radar | Radar sensors only | Airspace + kinetic | ±15% |
| Command | All sensors | All targets | ±10% |

Agents don't see each other's proposals. They don't communicate. They observe independently and submit.

### The Action

Each agent submits one proposal per available sensor — a JSON assignment:

```json
{ "agent_id": "satellite", "sensor_id": "S1", "target_id": "T0_1" }
{ "agent_id": "drone",     "sensor_id": "S2", "target_id": "T0_2" }
{ "agent_id": "radar",     "sensor_id": "S1", "target_id": "T0_3" }
```

That last two proposals just created a conflict — S1 claimed by both satellite and radar. The NegotiationLayer catches it before any sensor moves.

### The Reward

The **RewardEngine** computes four components every step:

**Task Reward (per-agent, capability-weighted)**

| Condition | Reward |
|---|---|
| P3 target covered, optimal sensor (capability ≥ 0.85) | +3.0 |
| P3 target covered, non-optimal sensor | +2.0 |
| P2 target covered | +1.0 |
| P1 target covered | +0.5 |
| Capable sensor idle while P3 uncovered | −2.0 |

**Coordination Bonus (system-level, split equally)**

Agents can only earn this together:

| Condition | Bonus |
|---|---|
| Conflicts self-resolved without Command override | +1.5 |
| All P3 targets covered this step | +2.0 |
| No idle sensors when P3 targets exist | +1.0 |

**Conflict Penalties**

| Conflict Type | Penalty |
|---|---|
| REDUNDANT_COVERAGE | −1.0 per involved agent |
| FORCED_ARBITRATION (Command override used) | −1.5 per involved agent |

**Look-Ahead Planning Incentive (retroactive)**

At episode end, the RewardEngine asks: were there steps where an agent held a sensor idle, and that sensor turned out to be the optimal choice for a P3 target that appeared later? If yes: `γ^k × future_reward` retroactive bonus. This teaches agents to hold sensors in reserve rather than greedily filling every slot every step.

---

## How the System Works

![Arya-X Architecture](./static/img/architecture.png)

**Step-by-step per episode:**

1. Environment resets — sensors and threats (P1/P2/P3) spawned
2. Each agent receives a partial, noisy observation scoped to its sensor type
3. Each agent independently proposes a sensor-to-target assignment
4. NegotiationLayer validates and collects all proposals
5. ConflictDetector flags REDUNDANT_COVERAGE, OVER_ASSIGNMENT, MISSED_PRIORITY_3
6. ConflictResolver runs 3 passes: Priority → Capability Score → Command Override
7. RewardEngine scores each agent individually
8. Environment advances — new threats spawn, sensors reset
9. Repeat until max_steps; Grader normalises total reward to [0.01, 0.99]

### The Conflict Resolution Pipeline

**Pass 1 — Priority:** When two agents claim the same sensor, keep the proposal targeting the highest-priority target.

**Pass 2 — Capability:** Sort remaining conflicts by `target_priority × sensor_capability_score`. The CAPABILITY_MATRIX drives this — satellite scores 0.95 on strategic targets, drone 0.95 on kinetic, radar 0.95 on airspace.

**Pass 3 — Command Override:** Command agent breaks any remaining tie. But invoking override costs −0.5 split across all agents, so the other three agents are incentivised to not create conflicts that require it.

---

## What Changed After Training

We trained **Qwen2.5-0.5B-Instruct** with **GRPO + LoRA** for 500 steps on a single T4 GPU, using AryaXEnv as the live reward source. Each step: structured prompt → LLM proposal → NegotiationLayer → GRPO update.

| Metric | Step 50 | Step 500 | Δ |
|---|---|---|---|
| Avg Reward/Step | 0.963 | 1.724 | **+79%** |
| Conflict Rate | 0.375 | 0.188 | **−50%** |
| Coordination Score | 0.625 | 0.813 | **+30%** |

**Critically — this happened organically.** No hard-coded coordination rules were added between step 50 and step 500. Conflict rate halved purely through the reward signal.

What each agent actually learned:

- **Satellite** learned to defer on kinetic targets (capability: 0.40) and preserve its sensors for strategic threats (0.95). The reward differential made this learnable.
- **Drone** stopped competing with Radar on airspace targets once the capability gap showed up in rewards.
- **Command** learned when to override and when to hold back. Override invocations dropped sharply after ~150 steps — the other agents learned to not create conflicts that required arbitration in the first place.
- **All agents** learned to leave sensors idle when no suitable targets exist — the look-ahead incentive made idling profitable rather than penalised.

### Training Curves

![Reward Curve](./checkpoints/arya_x_lora/reward_curve.png)
![Loss Curve](./checkpoints/arya_x_lora/loss_curve.png)

---

## Dashboard

![Multi-Agent Mode](./static/img/multi_agent.png)

![Single-Agent Mode](./static/img/single_agent.png)

---

## Before vs After Training

**Before — untrained agents conflict on every step:**

![Conflicts Before Training](./static/img/multi_agent_llm_conflicts.png)

**After — coordinated assignments, conflict rate ~9%:**

![After Training](./static/img/multi_agent_greedy_summary.png)

![Greedy Multi-Agent Baseline](./static/img/multi_agent_greedy.png)

---

## The Adaptive Curriculum

The **CurriculumEngine** adapts to the agents' collective performance in real time — not a fixed difficulty ladder.

**Phase 1 — Scaffolding (0–500 episodes):** Simple scenarios, 20 steps, 2–3 targets per step, no sensor failures. Agents learn basic sensor affinity and priority triage.

**Phase 2 — Coordination Pressure (500–2000 episodes):** Engineered conflicts, 40 steps, 3–5 targets, sensor failure probability 3–13%, conflict injection enabled. Agents are forced to handle OVER_ASSIGNMENT and REDUNDANT_COVERAGE constantly.

**Phase 3 — Adaptive Self-Play (2000+ episodes):** High density, correlated failures, agent freezing. Every 200 episodes, one specialist agent (Satellite/Drone/Radar) is frozen — its policy stops updating. The others must compensate.

Difficulty within each phase is a continuous value [0.0, 1.0] that escalates when the rolling 50-episode coordination score clears 0.72 and regresses when it drops below 0.35. **The environment adapts to the agents, not the other way around.**

---

## Live Dashboard

The dashboard at [https://sm21s-sentinel-env.hf.space/](https://sm21s-sentinel-env.hf.space/) shows everything in real time.

Four agents propose simultaneously. Each agent's assignments draw arcs in their own colour (🔵 Satellite · 🟢 Drone · 🟠 Radar · 🟣 Command). Conflicts trigger pulsing red rings on affected targets. The conflict log and live conflict rate update every step.

Select a mission (Easy / Medium / Hard), switch to **MULTI-AGENT** mode, click **RUN FULL EPISODE**, then **COMPUTE SCORE** for the full graded breakdown.

---

## Baseline Numbers

### Single-Agent (greedy)

| Task | Score |
|---|---|
| Easy (seed=42, 20 steps) | ~0.75 |
| Medium (seed=7, 40 steps) | ~0.65 |
| Hard (seed=13, 60 steps) | ~0.55 |

### Multi-Agent (greedy, 4 agents)

| Task | Conflict Rate | Avg Total Reward |
|---|---|---|
| Easy | ~0.10 | ~18.0 |
| Medium | ~0.20 | ~32.0 |
| Hard | ~0.35 | ~48.0 |

### Trained LoRA Agents (500 steps, Qwen2.5-0.5B, T4)

| Metric | Start | End | Δ |
|---|---|---|---|
| Avg Reward | 0.963 | 1.724 | **+79%** |
| Conflict Rate | 0.375 | 0.188 | **−50%** |
| Coordination Score | 0.625 | 0.813 | **+30%** |

---

## Why It Matters

**Who would care:**

Defense and intelligence agencies running real sensor fleets. Autonomous drone swarm coordinators. Any multi-robot system where agents compete for shared limited resources. Researchers studying emergent cooperation in LLM-based multi-agent systems.

**The real-world parallel is direct.** In actual ISR operations, sensor redundancy is expensive and coverage gaps are dangerous. A system that autonomously coordinates sensor allocation — detecting and resolving conflicts before they cause failures — has immediate operational value.

**The research contribution is concrete.** Arya-X shows that:

1. A well-designed conflict-aware reward signal can drive emergent cooperative behaviour in LLM agents with zero hard-coded coordination rules.
2. Capability-weighted rewards produce genuine specialisation — agents respect their own capability matrix without being told to.
3. Look-ahead planning incentives teach agents to hold resources in reserve instead of reacting greedily every step.
4. An adaptive curriculum that responds to collective coordination quality outperforms a fixed difficulty ladder.

**The explainability layer matters.** In real defense and intelligence contexts, operators must understand and trust autonomous recommendations before acting on them. Every conflict, every resolution pass, every reward component in Arya-X is exposed — nothing is a black box.

---

## Try It

### Live Demo (no setup)

👉 [https://sm21s-sentinel-env.hf.space/](https://sm21s-sentinel-env.hf.space/)

### Run Locally

```bash
git clone https://huggingface.co/spaces/sm21s/sentinel-env
cd sentinel-env
pip install -r requirements.txt

# Optional: set HF token for LLM mode (greedy fallback works without it)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python server.py
# Dashboard at http://localhost:7860
```

### Docker

```bash
docker build -t arya-x .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  arya-x
```

### Train Your Own Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WWX5skZToqne_rVulpE5vvR-lH_VK2gC?usp=sharing)

Full GRPO + LoRA training pipeline on a T4 GPU (free tier), under 30 minutes. Adapters save to `checkpoints/arya_x_lora/`.

---

## API Reference

### Multi-Agent Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset_multi` | POST | Reset env, returns per-agent observations |
| `/step_multi` | POST | Submit proposals from all agents, returns resolved assignments |
| `/auto_multi` | POST | Auto-run one step (LLM or greedy fallback) |
| `/grade_multi` | POST | Full episode + graded score breakdown |

### Single-Agent Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset single-agent env |
| `/step` | POST | Manual action `{sensor_id, target_id}` |
| `/step/auto` | POST | LLM or greedy assigns all sensors |
| `/grade` | POST | Full episode, normalised score [0–1] |

### Example: One Multi-Agent Step

```bash
# Reset
curl -X POST http://localhost:7860/reset_multi \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "max_steps": 20}'

# Step with proposals
curl -X POST http://localhost:7860/step_multi \
  -H "Content-Type: application/json" \
  -d '{
    "proposals": [
      {"agent_id": "satellite", "sensor_id": "S1", "target_id": "T0_1"},
      {"agent_id": "drone",     "sensor_id": "S2", "target_id": "T0_2"},
      {"agent_id": "radar",     "sensor_id": "S3", "target_id": "T0_3"}
    ]
  }'
```

---

## Project Structure

```
arya-x/
├── env/
│   ├── environment.py    # SentinelEnv — single-agent baseline
│   ├── multiagent.py     # AryaXEnv — multi-agent environment
│   ├── models.py         # Pydantic models (Sensor, Target, Proposal, etc.)
│   ├── dynamics.py       # Sensor init, target spawning, correlated failures
│   └── world_model.py    # Partial obs masking, noise, schema drift
├── interaction/
│   ├── conflict.py       # ConflictDetector — 4 conflict types
│   ├── resolver.py       # ConflictResolver — 3-pass resolution
│   ├── negotiation.py    # NegotiationLayer — full pipeline orchestrator
│   └── reward.py         # RewardEngine — 4-component reward
├── agents/
│   ├── base_agent.py     # BaseAgent — observe/propose/update interface
│   ├── satellite.py      # Satellite specialist
│   ├── drone.py          # Drone specialist
│   ├── radar.py          # Radar specialist
│   └── command.py        # Command agent — override authority
├── tasks/
│   ├── easy_task.py      # Easy: 3 sensors, 3 targets, 20 steps
│   ├── medium_task.py    # Medium: 4 sensors, 5 targets, 40 steps
│   ├── hard_task.py      # Hard: 5 sensors, 8 targets, 60 steps
│   └── grader.py         # grade_episode() + grade_multi_agent_episode()
├── checkpoints/
│   └── arya_x_lora/      # Trained adapter + training metrics + curves
├── curriculum.py         # CurriculumEngine — adaptive difficulty
├── server.py             # Flask API — all endpoints
├── train_colab.ipynb     # GRPO + LoRA training notebook
├── openenv.yaml          # OpenEnv spec
└── Dockerfile
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| RL Environment | Python, Pydantic, AryaXEnv |
| Training | HuggingFace TRL, GRPO, PEFT LoRA |
| Base Model | Qwen2.5-0.5B-Instruct |
| Inference | Qwen/Qwen2.5-72B-Instruct via HF Router |
| Backend API | Flask |
| Frontend | Leaflet.js, vanilla JS, CSS animations |
| Deployment | Docker, Hugging Face Spaces |
| Environment Spec | OpenEnv (openenv.yaml) |

---

## Hackathon Themes Addressed

**Theme #1 — Multi-Agent Interactions** *(primary)*
Four specialised agents submit proposals simultaneously without communicating. The NegotiationLayer is a literal negotiation protocol. The reward function makes cooperation strictly more profitable than competition — the coordination bonus can only be earned collectively.

**Theme #2 — Super Long-Horizon Planning**
The Hard task runs 60 steps. Component 4 of the reward function (`compute_episode_lookahead()`) retroactively rewards agents for keeping a sensor idle at step *t* if that sensor turned out to be the optimal choice for a P3 target at step *t+k*. Explicit long-horizon planning incentive baked into the reward signal.

**Theme #3.1 — World Modeling / Professional Tasks**
Three partial observability mechanisms: `apply_mask()` filters each agent's target list by sensor type, `add_observation_noise()` perturbs priority values with per-agent noise profiles, `get_priority_mapping()` introduces schema drift every 20 episodes. The domain — ISR sensor allocation — is a real professional task with real operational stakes.

**Theme #4 — Self-Improvement**
The CurriculumEngine runs three phases with continuous difficulty [0.0, 1.0] that adapts to the rolling 50-episode coordination score. Phase 3 introduces self-play: one agent is frozen every 200 episodes and the others must compensate. The environment adapts to the agents.

---

## Citation

```bibtex
@misc{arya-x-2026,
  title   = {Arya-X: Multi-Agent ISR Sensor Allocation with Conflict-Aware GRPO Training},
  year    = {2026},
  url     = {https://huggingface.co/spaces/sm21s/sentinel-env},
  note    = {OpenEnv Hackathon Round 2 — Meta × Hugging Face × PyTorch, Bangalore 2026}
}
```

---

*Built for the OpenEnv Hackathon Round 2, Bangalore 2026.*
*🔗 Live: [https://sm21s-sentinel-env.hf.space/](https://sm21s-sentinel-env.hf.space/) · 📦 HF: [huggingface.co/spaces/sm21s/sentinel-env](https://huggingface.co/spaces/sm21s/sentinel-env) · 📝 Blog: [BLOG.md](./BLOG.md)*
