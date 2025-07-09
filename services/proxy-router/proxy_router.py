# proxy_router.py ─ A pure reverse proxy
import httpx
import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the mapping of routes to services
SERVICE_URLS = {
    "agent-service": os.getenv("AGENT_SERVICE_URL", "http://localhost:8000"),
    "asr-api": os.getenv("ASR_API_URL", "http://localhost:8003"),
    "tools-api": os.getenv("TOOLS_API_URL", "http://localhost:9000"),
    "vllm-agent": os.getenv("VLLM_AGENT_URL", "http://localhost:8002"),
    "supervisor": "http://localhost:9005",
}

client = httpx.AsyncClient()

async def is_chat_model_sleeping():
    chat_model_url = SERVICE_URLS.get("vllm-agent")
    if not chat_model_url:
        return True
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{chat_model_url}/is_sleeping", timeout=2)
            if response.status_code == 200:
                return response.json().get("is_sleeping", True)
    except httpx.RequestError as e:
        logging.error(f"Error checking if chat model is sleeping: {e}")
    return True

async def wake_and_notify(service: str):
    if service == "agent-service":
        try:
            if await is_chat_model_sleeping():
                chat_model_url = SERVICE_URLS.get("vllm-agent")
                if chat_model_url:
                    await client.post(f"{chat_model_url}/wake_up", timeout=3)
                    logging.info("Woke up chat model, resetting prefix cache...")
                    await client.post(f"{chat_model_url}/reset_prefix_cache", timeout=3)
                    logging.info("Prefix cache reset.")
            
            supervisor_url = SERVICE_URLS.get("supervisor")
            if supervisor_url:
                await client.post(f"{supervisor_url}/activity", timeout=2)
                logging.info("Notified supervisor of activity.")

        except httpx.RequestError as e:
            logging.error(f"Error during wake_and_notify: {e}")


@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(service: str, path: str, request: Request):
    if request.method != "OPTIONS":
        await wake_and_notify(service)

    if service not in SERVICE_URLS:
        return {"error": f"Service '{service}' not found"}

    service_url = SERVICE_URLS[service]
    url = f"{service_url}/{path}"

    # Prepare the request to the downstream service
    headers = dict(request.headers)
    headers.pop("host", None)

    # Make the request to the downstream service
    try:
        body = await request.body()
        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            params=request.query_params,
            timeout=300.0,
        )
    except httpx.ConnectError as e:
        return {"error": f"Could not connect to {service}: {e}"}

    # Check if the response is chunked
    if 'transfer-encoding' in response.headers and 'chunked' in response.headers['transfer-encoding']:
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=response.headers,
        )
    else:
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response.headers,
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
