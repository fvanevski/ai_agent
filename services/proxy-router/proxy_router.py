# proxy_router.py ─ A pure reverse proxy
import httpx
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

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
}

client = httpx.AsyncClient()

@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(service: str, path: str, request: Request):
    if service not in SERVICE_URLS:
        return {"error": f"Service '{service}' not found"}

    service_url = SERVICE_URLS[service]
    url = f"{service_url}/{path}"

    # Prepare the request to the downstream service
    headers = dict(request.headers)
    # httpx uses 'host' in headers to connect, which might not be desirable
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
            timeout=30.0,
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
