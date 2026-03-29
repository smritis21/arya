---
title: SentinelEnv
emoji: 🛰️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# SentinelEnv — AI Sensor Allocation Environment

An [OpenEnv](https://huggingface.co/openenv)-compatible reinforcement learning environment where an LLM agent must allocate limited surveillance sensors (satellites, drones, radars) to high-priority threats (missile activity, border movements, airspace intrusions) under real-time conditions.

---

## What It Simulates

A military command centre monitoring threats across a region has a fixed set of sensors deployed at various locations. At every timestep, new threats appear across the map. The agent must decide which sensor covers which threat before the window expires — threats that go unhandled disappear and are counted as missed.

This directly models **ISR (Intelligence, Surveillance and Reconnaissance)** — a real operational problem where limited sensor capacity must be allocated across many simultaneous threats in priority order.

---

## Observation Space

Each `reset()` and `step()` returns an `Observation` object:

| Field | Type | Description |
|---|---|---|
| `sensors` | `List[Sensor]` | All sensors and their current state |
| `targets` | `List[Target]` | Active threats this timestep |
| `timestep` | `int` | Current step index (0-indexed) |

**Sensor fields:**

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique sensor ID e.g. `S1` |
| `type` | `str` | `satellite`, `drone`, or `radar` |
| `range` | `float` | Detection range in km (100–500) |
| `available` | `bool` | Whether sensor can be assigned this step |

**Target fields:**

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique target ID e.g. `T0_1` (step\_index) |
| `priority` | `int` | `3`=high, `2`=medium, `1`=low |
| `active` | `bool` | Whether target is unhandled this step |

---

## Action Space

Each `step_batch()` accepts a list of assignments — one per available sensor:

```json
[
  {"sensor_id": "S1", "target_id": "T0_1"},
  {"sensor_id": "S2", "target_id": "T0_3"}
]
```

| Field | Type | Description |
|---|---|---|
| `sensor_id` | `str` | ID of an available sensor |
| `target_id` | `str` | ID of an active target |

Invalid or empty actions incur an idle penalty of `-2.0`.

---

## Reward Function

| Condition | Reward |
|---|---|
| Assigned sensor to priority-3 target | `+2.0` |
| Assigned sensor to priority-2 target | `+1.0` |
| Assigned sensor to priority-1 target | `+0.5` |
| Idle sensor that could have covered a HIGH threat | `-2.0` |
| No assignments at all | `-2.0` |

Targets that exceed sensor capacity are **not penalised** — only wasted sensors are. Unhandled threats expire at the end of each step and do not carry over.

---

## Tasks & Graders

All graders return a normalised score in `[0.0, 1.0]`.

| Task | Sensors | Steps | Seed | Difficulty |
|---|---|---|---|---|
| Easy | 3–5 | 20 | 42 | Baseline allocation |
| Medium | 3–5 | 40 | 7 | More steps, varied priorities |
| Hard | 3–5 | 60 | 13 | Long horizon, high threat density |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/status` | GET | Health check + LLM connection status |
| `/reset` | POST | Reset environment, returns initial observation |
| `/state` | GET | Current observation without stepping |
| `/step` | POST | Single action `{"sensor_id": "S1", "target_id": "T0_1"}` |
| `/step/auto` | POST | LLM agent assigns all sensors (greedy fallback if no token) |
| `/grade` | POST | Run full episode and return normalised score |
| `/targets/custom` | POST | Register a custom threat `{"id","priority","lat","lon"}` |
| `/ui` | GET | Interactive dashboard |

---

## Project Structure

```
├── env/
│   ├── environment.py   # SentinelEnv — reset / step / step_batch / state
│   ├── models.py        # Typed Pydantic models
│   ├── dynamics.py      # Sensor init, target spawning
│   └── reward.py        # Reward computation
├── tasks/
│   ├── easy_task.py
│   ├── medium_task.py
│   ├── hard_task.py
│   └── grader.py        # grade_episode(), grade_summary()
├── agent/
│   └── policy.py        # Greedy + random baseline policies
├── templates/
│   └── dashboard.html   # Interactive Leaflet map dashboard
├── static/
│   ├── css/dashboard.css
│   └── js/dashboard.js
├── inference.py         # LLM agent loop — runs all 3 tasks, prints scores
├── server.py            # Flask server exposing OpenEnv HTTP API + dashboard
├── prevalidate.py       # Pre-submission validation script
├── openenv.yaml         # Environment configuration
├── Dockerfile
└── requirements.txt
```

---

## Setup

### Docker (recommended)

```bash
docker build -t sentinel-env .
docker run -p 5000:5000 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  sentinel-env
```

### Local

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

Open `http://localhost:5000/ui` for the interactive dashboard.

### Run inference script

```bash
python inference.py
```

### Pre-submission validation

```bash
python prevalidate.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face API token |

---

## Baseline Scores

Scores produced by `inference.py` with `meta-llama/Meta-Llama-3-8B-Instruct`:

| Task | Score |
|---|---|
| Easy (seed=42, 20 steps) | ~0.75 |
| Medium (seed=7, 40 steps) | ~0.65 |
| Hard (seed=13, 60 steps) | ~0.55 |

Scores are reproducible — fixed seeds ensure identical threat sequences every run.

---

## Dashboard Features

- Live Leaflet map with OpenStreetMap tiles
- Draggable sensor markers with reverse geocoding (shows real place names)
- Threat markers with priority-based colours and pulse animation for HIGH threats
- Animated arcs showing sensor-to-threat assignments
- Manual override — assign any sensor to any threat
- Auto step — LLM agent makes all assignments, greedy fallback if token not set
- Operation log showing `[LLM]` or `[greedy]` per step
- Episode summary in plain English after each run

---

*Built for the OpenEnv Hackathon — Round 1.*
