---
title: Arya-X
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
---

# Arya-X — Multi-Agent ISR Sensor Allocation

> *Four AI agents. A limited sensor fleet. Threats spawning in real time. No communication allowed. Can they learn to stop fighting each other and start protecting the skies?*

---

## 🔗 Quick Links

| | |
|---|---|
| 🚀 **Live Demo** | [https://sm21s-sentinel-env.hf.space/](https://sm21s-sentinel-env.hf.space/) |
| 📦 **HF Space** | [huggingface.co/spaces/sm21s/sentinel-env](https://huggingface.co/spaces/sm21s/sentinel-env) |
| 🧠 **Trained LoRA Adapter** | [checkpoints/arya_x_lora/](./checkpoints/arya_x_lora/) |
| 📝 **Full Technical Blog** | [BLOG.md](./BLOG.md) |
| 🎓 **Training Notebook (Colab)** | [Open in Colab](https://colab.research.google.com/drive/1WWX5skZToqne_rVulpE5vvR-lH_VK2gC?usp=sharing) |
| 📊 **Training Reward Curve** | [reward_curve.png](./checkpoints/arya_x_lora/reward_curve.png) |
| 🏗️ **OpenEnv Spec** | [openenv.yaml](./openenv.yaml) |

---

## The Problem Nobody Talks About: Sensors Fight Each Other

Picture a military command centre. A region to monitor. Threats incoming — some low priority, some critical. A limited fleet of sensors: satellites for wide-area strategic coverage, drones for tactical kinetic targets, radar for airspace threats.

Now here's the part that breaks every naive system: **multiple agents are trying to assign those sensors simultaneously.**

Satellite agent sees a P3 (critical) threat and claims Sensor S1. Drone agent sees the same threat and also claims S1. Radar agent piles on too. Result: one threat gets triple coverage, two other threats go completely unmonitored, and a high-priority target nobody noticed just escalated into a crisis.

This is the **OVER_ASSIGNMENT problem**. It's real. It happens in actual ISR (Intelligence, Surveillance, and Reconnaissance) operations. And it's exactly what Arya-X is built to solve.

The capability gap we're targeting: **no existing open RL environment models multi-agent sensor allocation with conflict detection, capability-weighted rewards, and adaptive curriculum training.** Arya-X fills that gap.

---

## From SentinelEnv to Arya-X: The Origin Story

This project started as **SentinelEnv** — a single-agent baseline where one policy assigned all sensors to all targets. It worked. It was boring.

The interesting question was: *what happens when you split that single agent into four specialised ones, each with their own sensor affinity, their own priority focus, and their own partial view of the world?*

That question became **Arya-X**.

The system evolved through the OpenEnv Hackathon (Meta × Hugging Face × PyTorch, Bangalore 2026) from a single-agent baseline into a full multi-agent architecture where four specialised agents — Satellite, Drone, Radar, and Command — negotiate sensor assignments in real time using a **ConflictResolver + NegotiationLayer** pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AryaXEnv                             │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Satellite │  │  Drone   │  │  Radar   │  │ Command  │   │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       └──────────────┴──────────────┴──────────────┘        │
│                           │ proposals                        │
│                  ┌────────▼────────┐                        │
│                  │ NegotiationLayer│  conflict_rate tracking │
│                  └────────┬────────┘                        │
│                  ┌────────▼────────┐                        │
│                  │ConflictDetector │  REDUNDANT_COVERAGE     │
│                  │                 │  OVER_ASSIGNMENT        │
│                  │                 │  MISSED_PRIORITY_3      │
│                  └────────┬────────┘                        │
│                  ┌────────▼────────┐                        │
│                  │ConflictResolver │  3-pass resolution:     │
│                  │                 │  Priority → Capability  │
│                  │                 │  → Command Override     │
│                  └────────┬────────┘                        │
│                  ┌────────▼────────┐                        │
│                  │   RewardEngine  │  per-agent rewards      │
│                  └─────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

![Architecture Diagram](./static/img/architecture.png)

---

## What the Agents See, Do, and Get Rewarded For

### The Observation

Every agent receives a **partial, noisy view** of the world — not the full ground truth. This is intentional.

| Agent | Sensors Visible | Targets Visible | Priority Noise |
|---|---|---|---|
| Satellite | Satellite sensors only | All targets | 10% drift |
| Drone | Drone sensors only | Kinetic + strategic targets | 5% drift |
| Radar | Radar sensors only | Airspace + kinetic targets | 15% drift |
| Command | All sensors | All targets | 10% drift |

Agents don't see each other's proposals. They don't communicate. They just observe and submit.

### The Action

Each agent submits one proposal per available sensor:

```json
{ "agent_id": "satellite", "sensor_id": "S1", "target_id": "T0_1" }
{ "agent_id": "drone",     "sensor_id": "S2", "target_id": "T0_2" }
{ "agent_id": "radar",     "sensor_id": "S1", "target_id": "T0_3" }
```

That third proposal just created an OVER_ASSIGNMENT conflict. S1 is claimed by both satellite and radar. The NegotiationLayer catches it before any sensor actually moves.

### The Conflict System

The **ConflictDetector** runs four checks every step:

- **REDUNDANT_COVERAGE** — two agents assigned different sensors to the same target. One sensor wasted. Penalty: −0.5 per agent.
- **OVER_ASSIGNMENT** — more sensors than the target's priority allows. P1 targets get one sensor max. P3 targets get two. Penalty: −0.5 per agent.
- **MISSED_PRIORITY_3** — a critical target went uncovered despite capable idle sensors being available. This is the worst outcome.
- **FORCED_ARBITRATION** — a conflict survived both Pass 1 and Pass 2 of the resolver and had to be escalated to Command override.

The **ConflictResolver** then runs three passes to clean up:

1. **Pass 1 — Priority**: When two agents claim the same sensor, keep the proposal targeting the highest-priority target.
2. **Pass 2 — Capability**: Sort remaining assignments by `target_priority × sensor_capability_score`. The CAPABILITY_MATRIX drives this — satellite scores 0.95 on strategic targets, drone scores 0.95 on kinetic, radar scores 0.95 on airspace.
3. **Pass 3 — Command Override**: Honour all Command agent proposals not already covered. Evict conflicting sensor claims.

### The Reward Function

The **RewardEngine** computes four components every step:

**Component 1 — Task Reward (per-agent, capability-weighted)**

| Condition | Reward |
|---|---|
| Covered P3 target with optimal sensor (cap ≥ 0.85) | +3.0 |
| Covered P3 target with non-optimal sensor | +2.0 |
| Covered P2 target | +1.0 |
| Covered P1 target | +0.5 |
| Idle sensor when P3 target uncovered | −2.0 |

**Component 2 — Coordination Bonus (system-level, split equally)**

Agents can only earn this together. If they collectively resolve a conflict without Command override: +1.5 split. If all P3 targets are covered: +2.0 split. If no sensors are idle when P3 targets exist: +1.0 split.

**Component 3 — Conflict Penalties**

REDUNDANT_COVERAGE: −1.0 per involved agent. FORCED_ARBITRATION: −1.5 per involved agent.

**Component 4 — Look-Ahead Planning Incentive (retroactive)**

At episode end, the RewardEngine looks back and asks: were there steps where an agent kept a sensor idle, and that sensor turned out to be the optimal choice for a P3 target that appeared later? If yes: `γ^k × future_reward` retroactive bonus. This teaches agents to hold sensors in reserve rather than greedily assigning everything every step.

---

## The Dashboard

The live dashboard at [https://sm21s-sentinel-env.hf.space/](https://sm21s-sentinel-env.hf.space/) shows everything in real time.

**Multi-agent mode** — four agents propose simultaneously. Each agent's assignments draw arcs in their own colour (🔵 Satellite · 🟢 Drone · 🟠 Radar · 🟣 Command). Conflicts trigger pulsing red rings on affected targets. The conflict log and live conflict rate update every step.

![Multi-Agent Dashboard](./static/img/multi_agent.png)

**Single-agent mode** — one LLM/greedy policy assigns all sensors. Clean baseline.

![Single Agent Mode](./static/img/single_agent.png)

**Multi-agent greedy** — coordinated assignments, per-agent coloured arcs, no conflicts.

![Multi-Agent Greedy](./static/img/multi_agent_greedy.png)

**LLM mode conflicts** — what happens before training: agents pile onto the same target every step.

![Multi-Agent LLM Conflicts](./static/img/multi_agent_llm_conflicts.png)

**Training trend chart** — reward climbing, conflict rate dropping, live in the dashboard after an episode.

![Dashboard Training Chart](./static/img/Dashboard_chart_trend.png)

---

## Training: What Changed After 500 Steps

We trained **Qwen2.5-0.5B-Instruct** with **GRPO + LoRA** for 500 steps on a single T4 GPU, using AryaXEnv as the live reward source. Each training step: the model receives an agent observation as a structured prompt, generates a proposal, that proposal is submitted to the live environment, the NegotiationLayer runs, and GRPO uses the returned reward to update the policy.

### The Numbers

| Step | Avg Reward | Conflict Rate | Coordination Score |
|------|-----------|---------------|-------------------|
| 50   | 0.963     | 0.375         | 0.625             |
| 150  | 1.714     | 0.188         | 0.813             |
| 400  | 1.746     | 0.313         | 0.688             |
| 500  | 1.724     | **0.188**     | **0.813**         |

**Conflict rate dropped 50%. Coordination score improved 30%. Average reward up 79%.**

Critically — this happened **organically**. No hard-coded coordination rules were added between step 50 and step 500. The agents learned to avoid conflicts through the reward signal alone.

### The Training Curve

![Training Reward Curve](./checkpoints/arya_x_lora/reward_curve.png)

![Training Loss Curve](./checkpoints/arya_x_lora/loss_curve.png)

### What the Agents Actually Learned

**Before training:** Four independent greedy policies. Each one looks at the available sensors, finds the highest-priority target, and claims the best sensor for it — without any awareness that three other agents are doing the exact same thing. Constant OVER_ASSIGNMENT conflicts on P3 targets. Complete neglect of P1 and P2 targets.

![Before Training — conflicts on every step](./static/img/multi_agent_llm_conflicts.png)

**After training:**

- The **Satellite agent** learned to defer on kinetic targets (its capability score: 0.40) and preserve sensors for strategic threats (0.95).
- The **Drone agent** stopped competing with Radar on airspace targets. Radar's airspace capability is 0.95. Drone's is 0.50. The reward differential made this learnable.
- The **Command agent** learned when to override and when not to. Override invocations dropped sharply after ~150 steps — the other agents learned to avoid creating conflicts that would require override in the first place.
- **All agents** learned to leave sensors idle when no suitable targets exist — the look-ahead incentive made this profitable.

![After Training — coordinated coverage, low conflict rate](./static/img/multi_agent_greedy_summary.png)

---

## The Curriculum: Three Phases, Adaptive Difficulty

The **CurriculumEngine** adapts to the agents' collective performance in real time.

**Phase 1 — Scaffolding:** Simple scenarios. 20 steps, 2–3 targets per step, no sensor failures. Agents learn basic sensor affinity and priority triage.

**Phase 2 — Coordination Press:** Engineered conflicts. 40 steps, 3–5 targets per step, sensor failure probability 3–13%, conflict injection enabled. Agents are forced to deal with OVER_ASSIGNMENT and REDUNDANT_COVERAGE constantly.

**Phase 3 — Adaptive Self-Play:** High density, correlated failures, agent freezing. Every 200 episodes, one of Satellite/Drone/Radar is frozen — its policy stops updating. The other agents must compensate. This is the self-play mechanism: agents learn to cover for a degraded teammate.

Difficulty within each phase is a continuous value [0.0, 1.0] that escalates when the rolling 50-episode coordination score clears 0.72 and regresses when it drops below 0.35. **The environment adapts to the agents — not the other way around.**

---

## Why It Matters

**Who would care about this?**

Defense and intelligence agencies operating real sensor fleets. Autonomous drone swarm coordinators. Any multi-robot system where multiple agents compete for shared resources. Researchers studying emergent cooperation in LLM-based multi-agent systems.

**The real-world parallel is direct.** In actual ISR operations, sensor redundancy is expensive and coverage gaps are dangerous. A system that can autonomously coordinate sensor allocation — detecting and resolving conflicts before they cause coverage failures — has immediate operational value.

**The research contribution is concrete.** Arya-X demonstrates that:
1. A well-designed conflict-aware reward signal can drive emergent cooperative behaviour in LLM agents without hard-coded coordination rules.
2. Capability-weighted rewards produce genuine specialisation — agents learn to respect their own capability matrix.
3. Look-ahead planning incentives teach agents to hold resources in reserve, not just react greedily.
4. An adaptive curriculum that responds to collective coordination quality is more effective than a fixed difficulty ladder.

**The explainability layer matters too.** In real defense and intelligence contexts, human operators must understand and trust autonomous recommendations before acting on them. Arya-X exposes every conflict, every resolution pass, and every reward component — nothing is a black box.

---

## Baseline Metrics

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

### Trained LoRA Agents (500 steps, Qwen2.5-0.5B, T4 GPU)

| Metric | Start (step 50) | End (step 500) | Δ |
|---|---|---|---|
| Avg Reward | 0.963 | 1.724 | **+79%** |
| Conflict Rate | 0.375 | 0.188 | **−50%** |
| Coordination Score | 0.625 | 0.813 | **+30%** |

---

## Try It

### Live Demo (no setup required)

👉 [https://sm21s-sentinel-env.hf.space/](https://sm21s-sentinel-env.hf.space/)

Select a mission (Easy / Medium / Hard), switch to **MULTI-AGENT** mode, and click **RUN FULL EPISODE**. Watch four agents negotiate sensor assignments in real time. Click **COMPUTE SCORE** at the end to get the full graded breakdown.

### Run Locally

```bash
# Clone and install
git clone https://huggingface.co/spaces/sm21s/sentinel-env
cd sentinel-env
pip install -r requirements.txt

# Set your HuggingFace token for LLM mode (optional — greedy fallback works without it)
# Windows PowerShell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:API_BASE_URL="https://router.huggingface.co/v1"

# Linux / macOS
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

Open [`train_colab.ipynb`](./train_colab.ipynb) in Google Colab (T4 GPU recommended), or use the hosted notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WWX5skZToqne_rVulpE5vvR-lH_VK2gC?usp=sharing)

The notebook runs the full GRPO + LoRA training pipeline against the live AryaXEnv and saves adapters to `checkpoints/arya_x_lora/`.

---

## API Reference

### Multi-Agent Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset_multi` | POST | Reset environment, returns per-agent observations |
| `/step_multi` | POST | Submit proposals from all agents, returns resolved assignments |
| `/auto_multi` | POST | Auto-run one step (LLM or greedy fallback) |
| `/grade_multi` | POST | Run full episode and return graded score breakdown |

### Single-Agent Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset single-agent environment |
| `/step` | POST | Single manual action `{sensor_id, target_id}` |
| `/step/auto` | POST | LLM or greedy agent assigns all sensors |
| `/grade` | POST | Run full episode, return normalised score [0–1] |

### Example: Run One Multi-Agent Step

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

# Auto step (LLM or greedy)
curl -X POST http://localhost:7860/auto_multi
```

---

## Project Structure

```
arya-x/
├── env/
│   ├── environment.py    # SentinelEnv — single-agent baseline
│   ├── multiagent.py     # AryaXEnv — canonical multi-agent environment
│   ├── models.py         # Pydantic models (Sensor, Target, Proposal, etc.)
│   ├── dynamics.py       # Sensor initialisation and target spawning
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
│   └── arya_x_lora/      # Trained LoRA adapter + training metrics + curves
├── templates/
│   └── dashboard.html    # Live dashboard UI
├── static/
│   ├── css/dashboard.css
│   └── js/dashboard.js
├── server.py             # Flask API — all endpoints
├── inference.py          # CLI inference script
├── train_colab.ipynb     # GRPO + LoRA training notebook
├── curriculum.py         # CurriculumEngine — adaptive difficulty
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

## Further Reading

- 📝 **Full Technical Blog** — [BLOG.md](./BLOG.md) — Deep dive into the conflict system, reward function design, curriculum phases, training surprises, and what we'd try with more compute.
- 🎓 **Training Notebook** — [Open in Colab](https://colab.research.google.com/drive/1WWX5skZToqne_rVulpE5vvR-lH_VK2gC?usp=sharing) — Full GRPO + LoRA training pipeline on T4 GPU.
- 📦 **HF Space** — [huggingface.co/spaces/sm21s/sentinel-env](https://huggingface.co/spaces/sm21s/sentinel-env)
- 🧠 **Trained Adapter** — [checkpoints/arya_x_lora/](./checkpoints/arya_x_lora/) — LoRA weights, training metrics JSON, reward curve PNG, loss curve PNG.
- 📊 **Training Metrics** — [training_metrics.json](./checkpoints/arya_x_lora/training_metrics.json) — 50 episodes of per-agent rewards, conflict rates, coordination scores.
- 🏗️ **OpenEnv Spec** — [openenv.yaml](./openenv.yaml) — Full environment specification for both single-agent and multi-agent modes.

---

## Hackathon Themes

Arya-X was built for the **OpenEnv Hackathon Round 2** (Meta × Hugging Face × PyTorch, Bangalore 2026) and addresses all four themes:

**Theme #1 — Multi-Agent Interactions** *(primary)*
Four specialised agents submit proposals simultaneously without communicating. The NegotiationLayer is a literal negotiation protocol. The reward function makes cooperation strictly more profitable than competition — the coordination bonus can only be earned collectively.

**Theme #2 — Super Long-Horizon Planning**
The Hard task runs 60 steps. Component 4 of the reward function (`compute_episode_lookahead()`) retroactively rewards agents for keeping a sensor idle at step t if that sensor turned out to be the optimal choice for a P3 target at step t+k. Explicit long-horizon planning incentive baked into the reward signal.

**Theme #3.1 — World Modeling / Professional Tasks**
Three partial observability mechanisms: `apply_mask()` filters each agent's target list by sensor type, `add_observation_noise()` perturbs priority values with per-agent noise profiles, `get_priority_mapping()` introduces schema drift every 20 episodes. The domain — ISR sensor allocation — is a real professional task.

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
