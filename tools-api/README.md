# LangChain Tools API

This FastAPI service exposes LangChain-based tools (such as yt_summarize) as OpenAI function-calling compatible endpoints for use with LLM chat clients.

## Endpoints
- `POST /yt_summarize` — Summarize a YouTube URL or local file using the critical summary pipeline.

## Example Request
```
POST /yt_summarize
{
  "source": "https://www.youtube.com/watch?v=..."
}
```

## Setup
- Requires Python 3.9+
- Install dependencies from the root `requirements.txt` (ensure `fastapi`, `pydantic`, and all LangChain dependencies are included)

## Running
```
uvicorn main:app --host 0.0.0.0 --port 8004 --reload
```

## OpenAPI Schema
- The OpenAPI schema is available at `/openapi.json` for tool autodiscovery.
