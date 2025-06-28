# supervisor.py ── container + sleep orchestration
import os, time, subprocess, requests
from threading import Thread

MODEL_PORTS = {
    "llama.cpp-embedding": 8001,
    "vllm-agent":          8002,
    "faster-whisper":      8003,
}


# Only vllm-agent uses inactivity-based sleep/stop
MODELS_WITH_SLEEP_API = {"vllm-agent"}
last_activity = {m: 0 for m in MODEL_PORTS}
SLEEP_AFTER = int(os.getenv("INACTIVITY_SLEEP_SECONDS", 300))   # 5 min
STOP_AFTER  = int(os.getenv("INACTIVITY_STOP_SECONDS", 3600))   # 1 h

# ---------------- helpers ---------------- #

def record_activity(model):
    last_activity[model] = time.time()

def _post(url, t=3):
    try:
        requests.post(url, timeout=t)
    except requests.exceptions.RequestException:
        pass

def _get(url, t=2):
    try:
        return requests.get(url, timeout=t)
    except requests.exceptions.RequestException:
        return None

def is_model_awake(model) -> bool:
    if model == "vllm-agent":
        r = _get(f"http://localhost:{MODEL_PORTS[model]}/is_sleeping")
        return bool(r and r.status_code == 200 and not r.json().get("is_sleeping"))
    # For faster-whisper, awake means container is running
    return _container_running(model)

def wake_model(model):
    if not _container_running(model):
        subprocess.run(["docker", "start", model], stdout=subprocess.DEVNULL)
    elif model == "vllm-agent":
        _post(f"http://localhost:{MODEL_PORTS[model]}/wake_up", t=3)

def sleep_model(model):
    if model == "vllm-agent":
        _post(f"http://localhost:{MODEL_PORTS[model]}/sleep", t=3)

# -------- docker helpers -------- #

def _container_running(model):
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", model],
        capture_output=True, text=True
    ).stdout.strip()
    return out == "true"

def _stop_container(model):
    subprocess.run(["docker", "stop", model], stdout=subprocess.DEVNULL, check=False)

# ------------- main loop ------------- #

def run_supervisor():
    while True:
        now = time.time()
        # vllm-agent: sleep after inactivity, stop after longer inactivity
        vllm_idle = now - last_activity.get("vllm-agent", 0)
        if vllm_idle > SLEEP_AFTER and is_model_awake("vllm-agent"):
            sleep_model("vllm-agent")
        if vllm_idle > STOP_AFTER and _container_running("vllm-agent"):
            _stop_container("vllm-agent")

        # faster-whisper: stop after 1 hour inactivity
        fw_idle = now - last_activity.get("faster-whisper", 0)
        if fw_idle > STOP_AFTER and _container_running("faster-whisper"):
            _stop_container("faster-whisper")

        time.sleep(60)
