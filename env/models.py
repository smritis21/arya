from pydantic import BaseModel
from typing import List, Dict, Optional


class Sensor(BaseModel):
    id: str
    type: str
    range: float
    available: bool


class Target(BaseModel):
    id: str
    priority: int  # 1=low, 2=medium, 3=high
    active: bool
    type: str = "strategic"  # "strategic" | "kinetic" | "airspace"


class Observation(BaseModel):
    sensors: List[Sensor]
    targets: List[Target]
    timestep: int


class Action(BaseModel):
    sensor_id: str
    target_id: str


class Reward(BaseModel):
    value: float


class Proposal(BaseModel):
    sensor_id: str
    target_id: str
    agent_id: str  # "satellite" | "drone" | "radar" | "command"
    priority_estimate: int
    confidence: float  # 0.0–1.0


class AgentObservation(BaseModel):
    agent_id: str
    agent_type: str  # "satellite" | "drone" | "radar" | "command"
    sensors: List[Sensor]
    targets: List[Target]
    timestep: int
    conflict_history: List[dict] = []


class ConflictRecord(BaseModel):
    conflict_type: str  # "redundant_coverage" | "missed_priority3" | "forced_arbitration"
    agents_involved: List[str]
    target_id: str
    resolution: str  # "priority_pass" | "capability_pass" | "command_override" | "unresolved"
    step: int


class EpisodeMetrics(BaseModel):
    coordination_score: float
    conflict_rate: float
    efficiency_score: float
    final_score: float
    conflicts_total: int
    conflicts_unresolved: int
    per_agent_reward: Dict[str, float]
