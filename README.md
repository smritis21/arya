---
title: Arya-X — Multi-Agent ISR Sensor Allocation
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Arya-X — Multi-Agent ISR Sensor Allocation System

## Live Demo

- 🔗 Live App: https://sm21s-sentinel-env.hf.space/
- 📦 Hugging Face Repo: https://huggingface.co/spaces/sm21s/sentinel-env

---

## Overview

**Arya-X** is an [OpenEnv](https://huggingface.co/openenv)-compatible reinforcement learning environment that simulates **multi-agent sensor allocation** for ISR (Intelligence, Surveillance, and Reconnaissance) operations.

The system evolved from a **single-agent baseline** (SentinelEnv) to the full **Arya-X multi-agent architecture**, where four specialised agents — satellite, drone, radar, and command — negotiate sensor assignments in real time using a **ConflictResolver + NegotiationLayer** pipeline.

**The core problem:** A military command centre monitors a region with a limited sensor fleet. At every timestep, new threats spawn. Multiple agents must claim sensors and propose assignments — without creating coverage conflicts — before the step window expires.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AryaXEnv                               │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Satellite │  │  Drone   │  │  Radar   │  │ Command  │   │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
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

---

## Agent Roles

| Agent     | Sensor Affinity | Priority Focus | Override Authority |
|-----------|----------------|----------------|--------------------|
| Satellite | `satellite`    | Wide-area / P3 | No                 |
| Drone     | `drone`        | Tactical / P2  | No                 |
| Radar     | `radar`        | Precision / P1 | No                 |
| Command   | Any            | Strategic      | **Yes**            |

The **Command agent** has final override authority; its proposals are honoured in Pass 3 of the resolver even when conflicts exist.

---

## Conflict System

### Conflict Types

| Type                  | Trigger                                               | Penalty       |
|-----------------------|-------------------------------------------------------|---------------|
| `REDUNDANT_COVERAGE`  | Two agents assign to the same target                  | −0.5 per agent|
| `OVER_ASSIGNMENT`     | Same sensor claimed by two different agents           | −0.5 per agent|
| `MISSED_PRIORITY_3`   | High-priority (P3) target uncovered by any agent      | logged only   |
| `FORCED_ARBITRATION`  | Command agent override resolves a conflict            | none          |

### ConflictResolver — 3-Pass Resolution

1. **Pass 1 – Priority**: When two agents claim the same sensor, keep the proposal targeting the highest-priority target.
2. **Pass 2 – Capability**: Sort remaining assignments by target priority × sensor capability score (`satellite=3`, `drone=2`, `radar=1`). Discard duplicates.
3. **Pass 3 – Command Override**: Honour all command-agent proposals not already covered; evict conflicting sensor claims.

### NegotiationLayer

Wraps the detector + resolver and tracks a running **conflict_rate**:

```
conflict_rate = steps_with_any_conflict / total_steps_completed
```

This metric is exposed in every `/step_multi` and `/auto_multi` response and displayed live on the dashboard.

---

## Reward Function

### Multi-Agent (per-agent, per-step)

| Condition                              | Reward       |
|----------------------------------------|--------------|
| Assigned sensor to P3 target           | `+2.0`       |
| Assigned sensor to P2 target           | `+1.0`       |
| Assigned sensor to P1 target           | `+0.5`       |
| Agent proposed nothing useful (idle)   | `−2.0`       |
| Agent involved in OVER_ASSIGNMENT      | `−0.5`       |
| Agent involved in REDUNDANT_COVERAGE   | `−0.5`       |

### Single-Agent (unchanged)

| Condition                                    | Reward |
|----------------------------------------------|--------|
| Assigned sensor to P3 target                 | `+2.0` |
| Assigned sensor to P2 target                 | `+1.0` |
| Assigned sensor to P1 target                 | `+0.5` |
| Idle sensor when unhandled HIGH threat exists | `−2.0` |

---

## Observation Space

### Single-Agent (`Observation`)

| Field      | Type           | Description                       |
|------------|----------------|-----------------------------------|
| `sensors`  | `List[Sensor]` | All sensors and availability      |
| `targets`  | `List[Target]` | Active threats this timestep      |
| `timestep` | `int`          | Current step (0-indexed)          |

### Multi-Agent (`AgentObservation` keyed by agent_id)

Each agent receives the same global sensor + target list, scoped to its perspective:

| Field        | Type   | Description                              |
|--------------|--------|------------------------------------------|
| `agent_id`   | `str`  | Agent identifier                         |
| `agent_type` | `str`  | `satellite \| drone \| radar \| command` |
| `sensors`    | `list` | All sensors with availability state      |
| `targets`    | `list` | All targets with priority and active flag|
| `timestep`   | `int`  | Current step                             |

---

## Action Space

### Single-Agent

```json
{ "sensor_id": "S1", "target_id": "T0_1" }
```

### Multi-Agent (`/step_multi` body)

```json
{
  "proposals": [
    { "agent_id": "satellite", "sensor_id": "S1", "target_id": "T0_1" },
    { "agent_id": "drone",     "sensor_id": "S2", "target_id": "T0_3" },
    { "agent_id": "command",   "sensor_id": "S4", "target_id": "T0_2" }
  ]
}
```

---

## API Documentation

### Single-Agent Endpoints (unchanged)

| Endpoint          | Method | Description                                            |
|-------------------|--------|--------------------------------------------------------|
| `/`               | GET    | Interactive dashboard                                  |
| `/status`         | GET    | Health check + LLM status                              |
| `/reset`          | POST   | Reset environment, returns initial observation         |
| `/state`          | GET    | Current observation without stepping                   |
| `/step`           | POST   | Single action `{sensor_id, target_id}`                 |
| `/step/auto`      | POST   | LLM or greedy agent assigns all sensors                |
| `/grade`          | POST   | Run full episode, return normalised score [0–1]        |
| `/targets/custom` | POST   | Register custom threat `{id, priority, lat, lon}`      |

### Multi-Agent Endpoints (new)

| Endpoint       | Method | Description                                                    |
|----------------|--------|----------------------------------------------------------------|
| `/reset_multi` | POST   | Reset multi-agent env, returns per-agent observations          |
| `/step_multi`  | POST   | Submit proposals from all agents, returns resolved assignments |
| `/auto_multi`  | POST   | NegotiationLayer runs auto proposals (LLM or greedy fallback) |

---

### Example Requests & Responses

#### `POST /reset_multi`

**Request:**
```json
{ "seed": 42, "max_steps": 20 }
```

**Response:**
```json
{
  "seed": 42,
  "max_steps": 20,
  "conflict_rate": 0.0,
  "agent_rewards": { "satellite": 0.0, "drone": 0.0, "radar": 0.0, "command": 0.0 },
  "observations": {
    "satellite": {
      "agent_id": "satellite", "agent_type": "satellite", "timestep": 0,
      "sensors": [...], "targets": [...]
    },
    "drone": { "agent_id": "drone", "agent_type": "drone", "timestep": 0, ... },
    "radar": { ... },
    "command": { ... }
  }
}
```

---

#### `POST /step_multi`

**Request:**
```json
{
  "proposals": [
    { "agent_id": "satellite", "sensor_id": "S1", "target_id": "T0_1" },
    { "agent_id": "drone",     "sensor_id": "S2", "target_id": "T0_2" },
    { "agent_id": "radar",     "sensor_id": "S1", "target_id": "T0_3" }
  ]
}
```

**Response:**
```json
{
  "observations": { "satellite": {...}, "drone": {...}, "radar": {...}, "command": {...} },
  "step_rewards":  { "satellite": 2.0, "drone": 1.0, "radar": -0.5, "command": -2.0 },
  "agent_rewards": { "satellite": 2.0, "drone": 1.0, "radar": -0.5, "command": -2.0 },
  "conflict_rate": 0.333,
  "conflicts": [
    { "type": "OVER_ASSIGNMENT", "agents": ["satellite", "radar"], "sensor_id": "S1", "target_id": null }
  ],
  "done": false,
  "info": {
    "step_count": 1,
    "assignments": [{"sensor": "S1", "target": "T0_1", "agent": "satellite"}, ...],
    "missed_targets": [],
    "conflict_rate": 0.333
  }
}
```

---

#### `POST /auto_multi`

Requires no body. Internally runs `_get_multi_proposals()` (LLM → greedy fallback), then `step_multiagent()`.

**Response** includes an additional `proposals` field listing what each agent submitted before conflict resolution:

```json
{
  "agent": "greedy",
  "proposals": [
    { "agent_id": "satellite", "sensor_id": "S1", "target_id": "T0_1" },
    { "agent_id": "drone",     "sensor_id": "S2", "target_id": "T0_2" }
  ],
  "conflict_rate": 0.0,
  "conflicts": [],
  ...
}
```

---

## Tasks & Graders

| Task   | Sensors | Targets/Step | Steps | Seed | Difficulty        |
|--------|---------|--------------|-------|------|-------------------|
| Easy   | 3–5     | 2–3          | 20    | 42   | Baseline allocation|
| Medium | 3–5     | 3–4          | 40    | 7    | More steps         |
| Hard   | 3–5     | 4–6          | 60    | 13   | Long horizon       |

---

## Project Structure

```
arya-x/
├── env/
│   ├── __init__.py
│   ├── environment.py    # SentinelEnv — single-agent (reset / step / state)
│   ├── models.py         # Pydantic models (Sensor, Target, Observation, Action)
│   ├── dynamics.py       # Sensor initialisation and target spawning
│   ├── multiagent.py     # AryaXEnv, ConflictDetector, ConflictResolver, NegotiationLayer
│   └── reward.py         # (deprecated stub)
├── tasks/
│   ├── easy_task.py
│   ├── medium_task.py
│   ├── hard_task.py
│   └── grader.py         # grade_episode() → normalised score [0.0–1.0]
├── templates/
│   └── dashboard.html    # UI — mode toggle, multi-agent panels, conflict log
├── static/
│   ├── css/dashboard.css # Styles — agent colors, conflict highlighting, metrics
│   └── js/dashboard.js   # Client-side logic — single + multi-agent modes
├── server.py             # Flask API — single-agent + multi-agent endpoints
├── inference.py          # CLI inference script (OpenAI client)
├── openenv.yaml          # OpenEnv spec — single + multi-agent config
├── Dockerfile            # HF Spaces container
├── requirements.txt
└── README.md
```

---

## Setup

### Docker

```bash
docker build -t arya-x .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  arya-x
```

Open `http://localhost:7860` for the dashboard.

### Local (Python)

```bash
pip install -r requirements.txt

# Windows (PowerShell)
$env:HF_TOKEN="hf_xxxxxxxxxxxx"
$env:MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
$env:API_BASE_URL="https://router.huggingface.co/v1"
python server.py

# Linux / macOS
export HF_TOKEN=hf_xxxxxxxxxxxx
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python server.py
```

Dashboard available at `http://localhost:7860`.

### Run Inference Script

```bash
python inference.py
```

---

## Environment Variables

| Variable      | Description                              |
|---------------|------------------------------------------|
| `API_BASE_URL` | LLM API endpoint (HuggingFace router)   |
| `MODEL_NAME`   | Model identifier (default: Llama-3-8B)  |
| `HF_TOKEN`     | HuggingFace API token (optional)        |

Without `HF_TOKEN`, the system uses the **greedy fallback** agent for all auto steps.

---

## Dashboard

The Arya-X dashboard supports both **single-agent** and **multi-agent** modes, togglable at any time from the top bar.

### Single-Agent Mode
- Live Leaflet map with sensor and threat markers
- Manual assign (sensor → target) or auto step
- Priority-based pulse animations for HIGH threats
- Animated arcs showing assignments
- Episode score bar and grader

### Multi-Agent Mode (NEW)
- **Mode toggle**: Switch between single / multi-agent instantly
- **Agent proposals**: Each agent draws arcs in its own color
  - 🔵 Satellite · 🟢 Drone · 🟠 Radar · 🟣 Command
- **Conflict indicators**: Targets involved in conflicts get a pulsing dashed ring
- **Conflict log**: Step-by-step history of conflict types and agents involved
- **Conflict rate**: Live metric, color-coded (green → amber → red)
- **Per-agent reward cards**: Cumulative rewards per agent, colored by agent
- **Episode summary**: Full multi-agent report with conflict rate and per-agent totals

---

## Baseline Metrics

### Single-Agent (greedy, `inference.py`)

| Task            | Score  |
|-----------------|--------|
| Easy (seed=42)  | ~0.75  |
| Medium (seed=7) | ~0.65  |
| Hard (seed=13)  | ~0.55  |

### Multi-Agent (greedy, 4 agents)

| Task   | Conflict Rate | Avg Total Reward |
|--------|---------------|------------------|
| Easy   | ~0.10         | ~18.0            |
| Medium | ~0.20         | ~32.0            |
| Hard   | ~0.35         | ~48.0            |

Conflict rate drops to near-zero when the LLM backend is active, as agents better coordinate their coverage.

---

*Built for the OpenEnv Hackathon — Arya-X upgrade, Round 2.*
