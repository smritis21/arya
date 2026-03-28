# SentinelRL — Adaptive Sensor Allocation Environment

A real-world OpenEnv environment where an AI agent allocates limited surveillance sensors (satellites, drones, radars) to dynamic high-priority targets under resource constraints.

## Environment Description

At each timestep, targets appear with different priority levels (1=low, 2=medium, 3=high). The agent must assign available sensors to active targets to maximize coverage. Sensors can fail, targets can move, and high-priority threats must not be missed.

## Observation Space

```json
{
  "sensors": [{ "id": "S1", "type": "satellite", "range": 350.0, "available": true }],
  "targets": [{ "id": "T0_1", "priority": 3, "active": true }],
  "timestep": 0
}
```

## Action Space

```json
{ "sensor_id": "S1", "target_id": "T0_1" }
```

## Reward Function

| Event | Reward |
|---|---|
| Track priority-3 target | +10.0 |
| Track priority-2 target | +5.0 |
| Track priority-1 target | +2.0 |
| Miss a high-priority target | -10.0 |
| No action / idle | -2.0 |

## Tasks

| Task | Sensors | Targets | Steps | Difficulty |
|---|---|---|---|---|
| Easy | 2 | 3 static | 20 | No failures, no movement |
| Medium | 3 | 5 dynamic | 40 | 5% failure prob, targets move |
| Hard | 4 | 8 dynamic | 60 | 15% failure prob, high-risk zones |

## Setup

```bash
pip install -r requirements.txt
python inference.py
```

## Docker

```bash
docker build -t sentinelrl .
docker run -e HF_TOKEN=your_token sentinelrl
```

## Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace API token |
| `API_BASE_URL` | LLM API endpoint (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Model identifier (default: `mistralai/Mistral-7B-Instruct-v0.1`) |

## Project Structure

```
sentinel_rl/
├── env/
│   ├── environment.py   # SentinelEnv: reset(), step(), state()
│   ├── models.py        # Typed Pydantic models
│   ├── dynamics.py      # Target spawning and movement
│   └── reward.py        # Reward computation
├── tasks/
│   ├── easy_task.py     # Static scenario
│   ├── medium_task.py   # Dynamic scenario
│   ├── hard_task.py     # High-risk scenario
│   └── grader.py        # Episode scoring (0.0–1.0)
├── agent/
│   └── policy.py        # Greedy baseline policy
├── inference.py         # LLM agent inference script
├── openenv.yaml         # OpenEnv spec
├── Dockerfile
└── requirements.txt
```
