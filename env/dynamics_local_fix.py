# LOCAL FIX — guarantee one sensor per type so every agent has something to claim
# Apply this to initialize_sensors() in dynamics.py if agents get locked out
#
# def initialize_sensors(seed: int = 42) -> List[Sensor]:
#     rng = random.Random(seed)
#     count = rng.randint(3, 5)
#     sensors = [
#         Sensor(id=f"S{i+1}", type=SENSOR_TYPES[i], range=round(rng.uniform(100.0, 500.0), 2), available=True)
#         for i in range(min(count, len(SENSOR_TYPES)))
#     ]
#     for i in range(len(SENSOR_TYPES), count):
#         sensors.append(Sensor(id=f"S{i+1}", type=rng.choice(SENSOR_TYPES), range=round(rng.uniform(100.0, 500.0), 2), available=True))
#     return sensors
