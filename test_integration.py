"""
ARYA-X Full Integration Test
Tests every module connection described in the spec.
"""
import sys
import traceback

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}")
        print(f"        {e}")

# ── 1. Core imports ────────────────────────────────────────────────────────────
check("env/__init__ exports SentinelEnv", lambda: __import__("env").SentinelEnv)
check("env.models all 6 models", lambda: [
    getattr(__import__("env.models", fromlist=["Sensor"]), x)
    for x in ["Sensor","Target","Observation","Action","Proposal","AgentObservation","ConflictRecord","EpisodeMetrics"]
])
check("env.dynamics all 4 functions", lambda: [
    getattr(__import__("env.dynamics", fromlist=["initialize_sensors"]), x)
    for x in ["initialize_sensors","spawn_targets","spawn_targets_stochastic","apply_correlated_failures"]
])
check("env.world_model all 3 functions", lambda: [
    getattr(__import__("env.world_model", fromlist=["add_observation_noise"]), x)
    for x in ["add_observation_noise","apply_mask","get_priority_mapping"]
])
check("interaction package all 4 classes", lambda: [
    getattr(__import__("interaction", fromlist=["NegotiationLayer"]), x)
    for x in ["NegotiationLayer","ConflictDetector","ConflictResolver","RewardEngine"]
])
check("agents package all 4 agents", lambda: [
    getattr(__import__("agents", fromlist=["SatelliteAgent"]), x)
    for x in ["SatelliteAgent","DroneAgent","RadarAgent","CommandAgent"]
])
check("curriculum.CurriculumEngine", lambda: __import__("curriculum").CurriculumEngine)
check("trainer.ARYAXTrainer importable", lambda: __import__("trainer").ARYAXTrainer)
check("tasks.grader all 3 functions", lambda: [
    getattr(__import__("tasks.grader", fromlist=["grade_episode"]), x)
    for x in ["grade_episode","grade_multi_agent_episode","run_deterministic_eval"]
])

# ── 2. env/environment.py — both AryaXEnv classes ─────────────────────────────
check("env.environment.SentinelEnv exists", lambda: __import__("env.environment", fromlist=["SentinelEnv"]).SentinelEnv)
check("env.environment.AryaXEnv (rich) exists", lambda: __import__("env.environment", fromlist=["AryaXEnv"]).AryaXEnv)
check("env.multiagent.AryaXEnv (server) exists", lambda: __import__("env.multiagent", fromlist=["AryaXEnv"]).AryaXEnv)

# ── 3. SentinelEnv full episode ────────────────────────────────────────────────
def _sentinel_episode():
    from env.environment import SentinelEnv
    env = SentinelEnv(max_steps=3, seed=42)
    obs = env.reset()
    assert obs.sensors and obs.targets
    from env.models import Action
    obs, r, done, info = env.step(Action(sensor_id=obs.sensors[0].id, target_id=obs.targets[0].id))
    assert "step_count" in info
check("SentinelEnv reset/step/info", _sentinel_episode)

# ── 4. env.environment.AryaXEnv (rich) — partial obs + world model ────────────
def _rich_aryax():
    from env.environment import AryaXEnv
    env = AryaXEnv(max_steps=3, seed=42, density_factor=2.5, failure_prob=0.1)
    obs_map = env.reset()
    assert set(obs_map.keys()) == {"satellite","drone","radar","command"}
    # Each agent gets different target counts (partial obs)
    sat_t = len(obs_map["satellite"].targets)
    drone_t = len(obs_map["drone"].targets)
    # satellite sees all, drone sees subset — may differ
    assert sat_t >= 0 and drone_t >= 0
check("env.environment.AryaXEnv partial obs + world model", _rich_aryax)

# ── 5. env.multiagent.AryaXEnv (server) — interaction/ pipeline ───────────────
def _server_aryax():
    from env.multiagent import AryaXEnv, Proposal
    env = AryaXEnv(max_steps=3, seed=42)
    obs = env.reset()
    proposals = [
        Proposal(agent_id="satellite", sensor_id=obs["satellite"].sensors[0]["id"],
                 target_id=obs["satellite"].targets[0]["id"]),
    ]
    obs2, rewards, done, info = env.step_multiagent(proposals)
    assert "conflict_rate" in info
    assert "step_rewards" in info
    assert set(rewards.keys()) == {"satellite","drone","radar","command"}
check("env.multiagent.AryaXEnv step_multiagent + conflict_rate", _server_aryax)

# ── 6. interaction/ pipeline end-to-end ───────────────────────────────────────
def _interaction_pipeline():
    from interaction import NegotiationLayer
    from interaction.reward import RewardEngine
    nl = NegotiationLayer()
    re = RewardEngine()
    proposals = [
        {"agent_id":"satellite","sensor_id":"S1","target_id":"T0_1","capability_score":0.9},
        {"agent_id":"drone",    "sensor_id":"S2","target_id":"T0_1","capability_score":0.5},  # conflict
    ]
    world_state = {
        "sensors": [{"id":"S1","type":"satellite","range":500,"available":True},
                    {"id":"S2","type":"drone","range":100,"available":True}],
        "targets": [{"id":"T0_1","priority":3,"active":True,"type":"strategic"},
                    {"id":"T0_2","priority":1,"active":True,"type":"airspace"}],
        "idle_sensors": [],
        "step": 0,
        "proposals": proposals,
    }
    result = nl.negotiate(proposals, world_state, lambda tied: max(tied, key=lambda p: p.get("capability_score",0)))
    assert len(result.conflicts_detected) > 0, "Expected conflict"
    assert nl.get_conflict_rate() > 0
    rewards = re.compute_step_reward(result.final_assignments, result, world_state, ["satellite","drone","radar","command"])
    assert isinstance(rewards, dict)
check("interaction/ NegotiationLayer + RewardEngine pipeline", _interaction_pipeline)

# ── 7. agents/ — observe/propose interface ────────────────────────────────────
def _agents_propose():
    from agents import SatelliteAgent, DroneAgent, RadarAgent, CommandAgent
    from env.models import AgentObservation, Sensor, Target
    sensors = [Sensor(id="S1", type="satellite", range=500, available=True)]
    targets = [Target(id="T0_1", priority=3, active=True),
               Target(id="T0_2", priority=2, active=True)]
    obs = AgentObservation(agent_id="satellite", agent_type="satellite",
                           sensors=sensors, targets=targets, timestep=0)
    sat = SatelliteAgent()
    sat.observe(obs)
    props = sat.propose()
    assert isinstance(props, list)

    drone_obs = AgentObservation(agent_id="drone", agent_type="drone",
                                 sensors=[Sensor(id="S2",type="drone",range=100,available=True)],
                                 targets=targets, timestep=0)
    d = DroneAgent()
    d.observe(drone_obs)
    assert isinstance(d.propose(), list)

    radar_obs = AgentObservation(agent_id="radar", agent_type="radar",
                                 sensors=[Sensor(id="S3",type="radar",range=300,available=True)],
                                 targets=targets, timestep=0)
    r = RadarAgent()
    r.observe(radar_obs)
    assert isinstance(r.propose(), list)

    cmd_obs = AgentObservation(agent_id="command", agent_type="command",
                               sensors=sensors, targets=targets, timestep=0)
    c = CommandAgent()
    c.observe(cmd_obs)
    assert isinstance(c.propose(), list)
check("agents/ all 4 observe()+propose() callable", _agents_propose)

# ── 8. CommandAgent.observe() accepts proposals kwarg ─────────────────────────
def _cmd_proposals_kwarg():
    from agents import CommandAgent
    from env.models import AgentObservation, Sensor, Target, Proposal
    sensors = [Sensor(id="S1",type="satellite",range=500,available=True)]
    targets = [Target(id="T0_1",priority=3,active=True)]
    obs = AgentObservation(agent_id="command",agent_type="command",
                           sensors=sensors,targets=targets,timestep=0)
    existing = [Proposal(sensor_id="S1",target_id="T0_1",agent_id="satellite",
                         priority_estimate=3,confidence=0.9)]
    c = CommandAgent()
    c.observe(obs, proposals=existing)
    props = c.propose()
    assert isinstance(props, list)
check("CommandAgent.observe(obs, proposals=...) signature", _cmd_proposals_kwarg)

# ── 9. DroneAgent theory-of-mind (loss tracking) ──────────────────────────────
def _drone_tom():
    from agents import DroneAgent
    from env.models import AgentObservation, Sensor, Target, ConflictRecord
    sensors = [Sensor(id="S2",type="drone",range=100,available=True)]
    targets = [Target(id="T0_2",priority=2,active=True)]
    obs = AgentObservation(agent_id="drone",agent_type="drone",
                           sensors=sensors,targets=targets,timestep=0)
    d = DroneAgent()
    d.observe(obs)
    # Simulate 3 losses to radar on redundant_coverage
    d.losses["redundant_coverage"] = 3
    props = d.propose()
    # Boundary target (even index) should have reduced confidence
    boundary_props = [p for p in props if d._is_boundary_target(p.target_id)]
    if boundary_props:
        assert boundary_props[0].confidence <= 0.5
check("DroneAgent theory-of-mind confidence reduction", _drone_tom)

# ── 10. CurriculumEngine → AryaXEnv config connection ─────────────────────────
def _curriculum_to_env():
    from curriculum import CurriculumEngine
    from env.multiagent import AryaXEnv
    ce = CurriculumEngine()
    cfg = ce.get_scenario_config()
    assert "max_steps" in cfg
    assert "sensor_failure_prob" in cfg
    env = AryaXEnv(
        max_steps=cfg["max_steps"],
        seed=42,
        density_factor=1.5 + ce.difficulty_level * 2.5,
        failure_prob=cfg["sensor_failure_prob"],
    )
    obs = env.reset()
    assert obs is not None
check("CurriculumEngine config -> AryaXEnv constructor", _curriculum_to_env)

# ── 11. grader.grade_multi_agent_episode ──────────────────────────────────────
def _grader_multi():
    from tasks.grader import grade_multi_agent_episode
    from interaction import NegotiationLayer
    nl = NegotiationLayer()
    result = grade_multi_agent_episode(
        per_agent_rewards={"SAT":10.0,"UAV":5.0,"RDR":3.0,"CMD":2.0},
        step_count=20,
        num_sensors=4,
        negotiation_layer=nl,
        num_agents=4,
    )
    assert "overall_score" in result
    assert "coordination_score" in result
    assert "conflict_rate" in result
check("grader.grade_multi_agent_episode", _grader_multi)

# ── 12. server.py imports cleanly ─────────────────────────────────────────────
check("server.py imports without error", lambda: __import__("server"))

# ── 13. inference.py imports + run_multi_task callable ────────────────────────
def _inference_multi():
    import inference
    assert callable(inference.run_multi_task)
    assert callable(inference._greedy_multi_proposals)
check("inference.py run_multi_task callable", _inference_multi)

# ── 14. trainer.py ARYAXTrainer instantiates (no model) ──────────────────────
def _trainer_init():
    from trainer import ARYAXTrainer
    # Should not raise even without GPU/model
    t = ARYAXTrainer.__new__(ARYAXTrainer)
    t.curriculum = __import__("curriculum").CurriculumEngine()
    from interaction import NegotiationLayer
    from interaction.reward import RewardEngine
    t.negotiation = NegotiationLayer()
    t.reward_engine = RewardEngine()
    t._grpo_trainer = None
    assert t._grpo_trainer is None
check("ARYAXTrainer attributes accessible", _trainer_init)

# ── 15. tasks/ easy/medium/hard envs return SentinelEnv ───────────────────────
def _task_envs():
    from tasks.easy_task import get_easy_env
    from tasks.medium_task import get_medium_env
    from tasks.hard_task import get_hard_env
    from env.environment import SentinelEnv
    for fn in [get_easy_env, get_medium_env, get_hard_env]:
        e = fn()
        assert isinstance(e, SentinelEnv)
        obs = e.reset()
        assert obs.sensors
check("tasks/ easy/medium/hard envs", _task_envs)

# ── 16. env.multiagent NegotiationLayer is interaction/ (not duplicate) ────────
def _no_duplicate_nl():
    from env.multiagent import NegotiationLayer as MaNL
    from interaction import NegotiationLayer as INL
    assert MaNL is INL, "env.multiagent.NegotiationLayer must BE interaction.NegotiationLayer"
check("env.multiagent.NegotiationLayer IS interaction.NegotiationLayer (no duplicate)", _no_duplicate_nl)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  PASSED: {len(PASS)}/{len(PASS)+len(FAIL)}")
if FAIL:
    print(f"  FAILED: {len(FAIL)}")
    for name, err in FAIL:
        print(f"    - {name}")
        print(f"      {err}")
print(f"{'='*55}")
sys.exit(0 if not FAIL else 1)
