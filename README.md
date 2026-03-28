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
