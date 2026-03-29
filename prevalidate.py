"""
Pre-submission validator — run before `openenv push`.
Usage: python prevalidate.py
"""
import sys, os, json, traceback

sys.path.insert(0, ".")

G = "\033[92m✓\033[0m"
R = "\033[91m✗\033[0m"
failures = []


def check(label, fn):
    try:
        fn()
        print(f"  {G} {label}")
    except Exception as e:
        print(f"  {R} {label}: {e}")
        failures.append((label, traceback.format_exc()))


# ── 1. Mandatory env vars are set ────────────────────────────────────────────
def check_env_vars():
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)]
    assert not missing, f"Missing env vars: {missing}  →  set them before pushing"


# ── 2. inference.py is in root and imports OpenAI client ─────────────────────
def check_inference_file():
    assert os.path.isfile("inference.py"), "inference.py not found in project root"
    src = open("inference.py").read()
    assert "from openai import OpenAI" in src, "inference.py must import OpenAI client"
    assert "API_BASE_URL" in src and "MODEL_NAME" in src and "HF_TOKEN" in src, \
        "inference.py must reference API_BASE_URL, MODEL_NAME, HF_TOKEN"


# ── 3. openenv.yaml has required keys and no default secrets ─────────────────
def check_yaml():
    import yaml
    cfg = yaml.safe_load(open("openenv.yaml"))
    for key in ("FLASK_ENV", "SECRET_KEY", "JWT_SECRET_KEY", "DATABASE_URL",
                "API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        assert key in cfg, f"openenv.yaml missing key: {key}"
    defaults = {
        "super-secret-defense-key-change-in-prod",
        "jwt-defense-secret-key-change-in-prod",
    }
    for k in ("SECRET_KEY", "JWT_SECRET_KEY"):
        assert cfg[k] not in defaults, \
            f"openenv.yaml: {k} is still the default placeholder — change it before pushing"


# ── 4. Env reset / step cycle ────────────────────────────────────────────────
def check_env_cycle():
    from env import SentinelEnv
    from env.models import Action

    env = SentinelEnv(max_steps=3, seed=0)
    obs = env.reset()
    assert obs.timestep == 0 and obs.sensors and obs.targets

    s = next(s for s in obs.sensors if s.available)
    t = next(t for t in obs.targets if t.active)
    obs, reward, done, info = env.step(Action(sensor_id=s.id, target_id=t.id))
    assert isinstance(reward, float)
    assert {"assignments", "missed_targets", "step_count"} <= info.keys()

    # None → idle penalty
    obs, reward, done, info = env.step(None)
    assert reward == -2.0, f"Expected idle penalty -2.0, got {reward}"

    # dict input → coerced correctly
    s2 = next((s for s in obs.sensors if s.available), None)
    t2 = next((t for t in obs.targets if t.active), None)
    if s2 and t2:
        obs, reward, done, info = env.step({"sensor_id": s2.id, "target_id": t2.id})
        assert isinstance(reward, float)


# ── 5. parse_action handles bad LLM output safely ────────────────────────────
def check_parse_action():
    from env import SentinelEnv
    from inference import parse_action
    from env.models import Action

    obs = SentinelEnv(max_steps=5, seed=1).reset()
    s = obs.sensors[0]
    t = next(t for t in obs.targets if t.active)

    # Valid JSON embedded in prose (LLM often wraps it)
    raw = f'Sure! Here is the answer: {{"sensor_id": "{s.id}", "target_id": "{t.id}"}}'
    assert isinstance(parse_action(raw, obs), Action), "Failed to parse valid embedded JSON"

    # Garbage → None, must not raise
    assert parse_action("I cannot determine the answer.", obs) is None
    assert parse_action("", obs) is None
    assert parse_action('{"sensor_id": "FAKE", "target_id": "FAKE"}', obs) is None


# ── 6. fallback_action always returns a valid action ─────────────────────────
def check_fallback():
    from env import SentinelEnv
    from inference import fallback_action

    obs = SentinelEnv(max_steps=5, seed=2).reset()
    action = fallback_action(obs)
    assert action is not None, "fallback_action returned None with available sensors/targets"

    valid_sensors = {s.id for s in obs.sensors if s.available}
    valid_targets = {t.id for t in obs.targets if t.active}
    assert action.sensor_id in valid_sensors, "fallback sensor_id not in available sensors"
    assert action.target_id in valid_targets, "fallback target_id not in active targets"

    # Highest-priority target must be chosen
    top_priority = max(t.priority for t in obs.targets if t.active)
    chosen_target = next(t for t in obs.targets if t.id == action.target_id)
    assert chosen_target.priority == top_priority, \
        f"fallback should pick highest-priority target (p={top_priority}), got p={chosen_target.priority}"


# ── 7. Full dry-run (fallback only, no LLM call) ─────────────────────────────
def check_dry_run():
    from env import SentinelEnv
    from inference import fallback_action, build_prompt

    env = SentinelEnv(max_steps=5, seed=99)
    obs = env.reset()
    total, done = 0.0, False
    while not done:
        action = fallback_action(obs)
        obs, reward, done, info = env.step(action)
        total += reward
    assert info["step_count"] > 0
    prompt = build_prompt(obs)
    assert isinstance(prompt, str) and len(prompt) > 50, "build_prompt returned empty/short string"


# ── 8. grade_episode produces score in [0.0, 1.0] ───────────────────────────
def check_grade_endpoint():
    from tasks.grader import grade_episode
    from env import SentinelEnv
    from inference import fallback_action

    env = SentinelEnv(max_steps=10, seed=42)
    obs = env.reset()
    total, done = 0.0, False
    while not done:
        action = fallback_action(obs)
        obs, reward, done, info = env.step(action)
        total += reward
    score = grade_episode(total, info["step_count"])
    assert 0.0 <= score <= 1.0, f"grade_episode out of range: {score}"
    print(f"      score={score:.4f}  total_reward={total:.1f}  steps={info['step_count']}", end="")


# ── 9. Dockerfile exists and references server:app ──────────────────────────
def check_dockerfile():
    assert os.path.isfile("Dockerfile"), "Dockerfile not found"
    src = open("Dockerfile").read()
    assert "server:app" in src, "Dockerfile must run server:app via gunicorn"
    assert "gunicorn" in src, "Dockerfile must use gunicorn, not flask dev server"


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Pre-submission Validation ===\n")
    check("Mandatory env vars (API_BASE_URL, MODEL_NAME, HF_TOKEN)", check_env_vars)
    check("inference.py in root + uses OpenAI client",               check_inference_file)
    check("openenv.yaml keys & no default secrets",                  check_yaml)
    check("Env reset / step cycle",                                  check_env_cycle)
    check("parse_action — safe on bad LLM output",                   check_parse_action)
    check("fallback_action — valid + highest-priority target",       check_fallback)
    check("Full dry-run (fallback, no LLM)",                         check_dry_run)
    check("grade_episode score in [0.0, 1.0]",                       check_grade_endpoint)
    check("Dockerfile uses gunicorn + server:app",                   check_dockerfile)
    print()
    if failures:
        print(f"❌  {len(failures)} check(s) failed:\n")
        for name, tb in failures:
            print(f"--- {name} ---\n{tb}")
        sys.exit(1)
    print("✅  All checks passed — safe to run `openenv push`.\n")
