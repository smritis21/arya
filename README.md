# SentinelEnv — AI Sensor Allocation Environment

An [OpenEnv](https://huggingface.co/openenv)-compatible reinforcement learning environment where an agent must allocate limited surveillance sensors (satellites, drones, radars) to high-priority targets (missile activity, border movements, airspace intrusions) under real-time threat conditions.

---

## Observation Space

Each `reset()` and `step()` returns an `Observation` object:

| Field | Type | Description |
|---|---|---|
| `sensors` | `List[Sensor]` | All sensors and their current state |
| `targets` | `List[Target]` | All active targets this timestep |
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

Each `step()` accepts a single assignment:

```json
{"sensor_id": "S1", "target_id": "T0_3"}
```

| Field | Type | Description |
|---|---|---|
| `sensor_id` | `str` | ID of an available sensor |
| `target_id` | `str` | ID of an active target |

Invalid or `null` actions incur an idle penalty of `-2.0`.

---

## Reward Function

| Condition | Reward |
|---|---|
| Assigned sensor to priority-3 target | `+10.0` |
| Assigned sensor to priority-2 target | `+5.0` |
| Assigned sensor to priority-1 target | `+2.0` |
| Each unassigned priority-3 target | `-10.0` |
| Invalid action or no action | `-2.0` |

---

## Tasks & Graders

All graders return a normalized score in `[0.0, 1.0]`.

| Task | Sensors | Targets | Steps | Difficulty |
|---|---|---|---|---|
| Easy | 2 | 3 static | 20 | Baseline allocation |
| Medium | 3 | 5 dynamic | 40 | Moving targets, 5% sensor failure |
| Hard | 4 | 8 dynamic | 60 | High-risk zones, 15% sensor failure |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/status` | GET | Health check |
| `/reset` | POST | Reset environment, returns initial observation |
| `/state` | GET | Current observation without stepping |
| `/step` | POST | Take action `{"sensor_id": "S1", "target_id": "T0_3"}` |
| `/step/auto` | POST | Auto-step with greedy fallback policy |
| `/grade` | POST | Run full episode and return normalized score |

---

## Setup

### Docker (recommended)

```bash
docker build -t sentinel-env .
docker run -p 5000:5000 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  sentinel-env
```

### Local

```bash
pip install -r requirements.txt
python server.py
```

### Run inference script

```bash
export HF_TOKEN=your_token
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export API_BASE_URL=https://router.huggingface.co/v1
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

## Project Structure

```
├── env/
│   ├── environment.py   # SentinelEnv — reset/step/state
│   ├── models.py        # Typed Pydantic models
│   ├── dynamics.py      # Sensor init, target spawning
│   └── reward.py        # Reward computation
├── tasks/
│   ├── easy_task.py
│   ├── medium_task.py
│   ├── hard_task.py
│   └── grader.py        # grade(), grade_episode(), grade_summary()
├── agent/
│   └── policy.py        # Greedy + random baseline policies
├── inference.py         # LLM agent loop with graded score output
├── server.py            # Flask server exposing OpenEnv HTTP API
├── prevalidate.py       # Pre-submission validation script
├── openenv.yaml         # Environment configuration
├── Dockerfile
└── requirements.txt
```

---

*Built for the OpenEnv Hackathon — Round 1.*
