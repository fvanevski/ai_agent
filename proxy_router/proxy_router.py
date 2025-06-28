# proxy_router.py ─ unified GPU router / gatekeeper
import threading, json, httpx, os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from supervisor import (
    record_activity, run_supervisor, is_model_awake,
    wake_model
)

app = FastAPI()

# GPU-bound services and their local ports
MODEL_PORTS = {
    "llama.cpp-embedding": 8001,
    "vllm-agent":          8002,
    "faster-whisper":      8003,
}

# ───────────────────────────────────────────────────────────────
# Path  →  target-service map
#   (exact match is enough because we keep the endpoints short)
# ───────────────────────────────────────────────────────────────
MODEL_BY_PATH = {
    "/v1/embeddings":          "llama.cpp-embedding",
    "/v1/chat/completions":    "vllm-agent",

    # Faster-Whisper endpoints
    "/transcribe":             "faster-whisper",
    "/transcribe_url":         "faster-whisper",
    # vllm-agent wake endpoint
    "/wake_up":                "vllm-agent",
}

TIMEOUT = float(os.getenv("PROXY_TIMEOUT", 1800))

# ---------------------------------------------------------------------------
# middleware: intercept only the known API paths, forward everything else
# ---------------------------------------------------------------------------
@app.middleware("http")
async def route_request(request: Request, call_next):
    if request.url.path not in MODEL_BY_PATH:
        # static files, docs, or unknown path → let FastAPI handle
        return await call_next(request)

    model_key = MODEL_BY_PATH[request.url.path]
    port      = MODEL_PORTS[model_key]

    # If this is a transcribe request, wake up vllm-agent first
    if model_key == "faster-whisper" and request.url.path in ("/transcribe", "/transcribe_url"):
        from supervisor import wake_model
        wake_model("vllm-agent")

    # Spin container up or wake model if needed
    if not is_model_awake(model_key):
        wake_model(model_key)
        return JSONResponse(
            status_code=202,
            content={"status": f"{model_key} waking – retry in 2 s"}
        )

    record_activity(model_key)
    target_url = f"http://localhost:{port}{request.url.path}"

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            prox_req = client.build_request(
                method  = request.method,
                url     = target_url,
                headers = request.headers.raw,
                content = await request.body()
            )
            prox_resp = await client.send(prox_req, stream=True)
            response = StreamingResponse(
                prox_resp.aiter_bytes(),
                status_code = prox_resp.status_code,
                media_type  = prox_resp.headers.get("content-type"),
                headers     = {k: v for k, v in prox_resp.headers.items()
                               if k.lower().startswith("content-")}
            )
            # ...container stop logic removed...
            return response
    except httpx.ReadTimeout:
        return JSONResponse(status_code=504,
                            content={"error": f"{model_key} timed-out"})
    except Exception as exc:
        return JSONResponse(status_code=500,
                            content={"error": f"proxy error: {exc}"})

# ---------------------------------------------------------------------------
# Launch router + background supervisor
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=run_supervisor, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
