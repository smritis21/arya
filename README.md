<<<<<<< HEAD
# AI Sensor Allocation System for Defense Surveillance

## 🛡️ Project Description
The **AI Sensor Allocation System** is an advanced, reinforcement learning-based platform designed to optimally assign limited surveillance sensors (such as satellites, drones, and radars) to high-priority targets, including missile activity, border movements, and airspace intrusions. 

By leveraging a custom Gym-style RL environment and real-time data streaming, the system maximizes threat coverage and minimizes blind spots to ensure maximum situational awareness.

## ✨ Features
- **Smart Sensor Allocation**: Uses Reinforcement Learning to dynamically allocate limited resources.
- **Custom RL Environment**: Built with Gymnasium to simulate defense scenarios.
- **Real-Time Dashboard**: React-based frontend for visualizing threat levels and sensor deployments.
- **Secure Authentication**: JWT-based access control for API endpoints.
- **REST API**: Fully-featured backend for manual sensor overrides and integrations.

## 🛠️ Tech Stack
- **Backend**: Python, Flask, Flask-RESTful
- **Frontend**: React.js
- **Machine Learning**: scikit-learn, Gymnasium (RL environment)
- **Data Handling**: pandas, numpy
- **Authentication**: JWT (JSON Web Tokens)
- **Database**: SQLite (via SQLAlchemy)

---

## 🚀 Installation & Setup

### Option 1: Using Docker (Recommended)
This is the fastest way to get the system running with all dependencies pre-configured.

1. Build the Docker image:
   ```bash
   docker build -t ai-sensor-system .
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 ai-sensor-system
   ```
3. The API will be available at `http://localhost:5000`.

### Option 2: Local Setup (Development)

1. Clone the repository and navigate into the project directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up the environment variables by editing `openenv.yaml` or creating a `.env` file.
5. Initialize the SQLite database:
   ```bash
   flask db upgrade
   ```
6. Start the Flask Backend:
   ```bash
   flask run --port=5000
   ```
7. In a separate terminal, navigate to the frontend directory and start the React app:
   ```bash
   cd frontend
   npm install
   npm start
   ```

---

## 📡 Example API Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/api/auth/login` | POST | Authenticate and retrieve JWT token. | No |
| `/api/sensors/status` | GET | Retrieve the current status of all sensors. | Yes |
| `/api/targets/active` | GET | List all currently tracked high-priority targets. | Yes |
| `/api/allocation/optimize` | POST | Trigger the RL model to re-allocate sensors based on new threats. | Yes |
| `/api/allocation/manual` | POST | Manually override an AI allocation decision. | Yes |

---

## 🔮 Future Improvements
- **Multi-Agent RL**: Transition from single-agent model to cooperative multi-agent reinforcement learning (MARL) for swarming drones.
- **Live Satellite Feed Integration**: Consume mock/real satellite API data streams.
- **PostgreSQL Migration**: Move from SQLite to PostgreSQL for production scalability.
- **Mobile Application**: Extend the dashboard to a mobile app for field commanders.

---
*Created as a final-year AI/ML capstone project / hackathon submission.*
=======
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
| Assigned sensor to priority-3 target | `+2.0` |
| Assigned sensor to priority-2 target | `+1.0` |
| Assigned sensor to priority-1 target | `+0.5` |
| Each unassigned priority-3 target | `-2.0` |
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
  -e API_BASE_URL=https://router.huggingface.co/hf-inference/v1 \
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
export API_BASE_URL=https://router.huggingface.co/hf-inference/v1
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
>>>>>>> round1-submission
