from pydantic import BaseModel
from typing import List


class Sensor(BaseModel):
    id: str
    type: str
    range: float
    available: bool


class Target(BaseModel):
    id: str
    priority: int  # 1=low, 2=medium, 3=high
    active: bool


class Observation(BaseModel):
    sensors: List[Sensor]
    targets: List[Target]
    timestep: int


class Action(BaseModel):
    sensor_id: str
    target_id: str


class Reward(BaseModel):
    value: float
