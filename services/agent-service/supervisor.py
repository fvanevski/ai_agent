# supervisor.py ─ A modern supervisor for AI models
import asyncio
import httpx
import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import logging
import time

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- Model Configuration ---
CHAT_MODEL_URL = os.getenv("VLLM_AGENT_URL", "http://localhost:8002")
SLEEP_AFTER_SECONDS = int(os.getenv("INACTIVITY_SLEEP_SECONDS", 300))  # 5 minutes

# --- State ---
last_activity_time = time.time()

# --- Helper Functions ---
async def is_chat_model_sleeping():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{CHAT_MODEL_URL}/is_sleeping", timeout=2)
            if response.status_code == 200:
                return response.json().get("is_sleeping", True)
    except httpx.RequestError as e:
        logging.error(f"Error checking if chat model is sleeping: {e}")
    return True # Assume sleeping on error

async def sleep_chat_model():
    if not await is_chat_model_sleeping():
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{CHAT_MODEL_URL}/sleep", timeout=3)
                logging.info("Chat model put to sleep due to inactivity.")
        except httpx.RequestError as e:
            logging.error(f"Error putting chat model to sleep: {e}")

async def wake_chat_model():
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{CHAT_MODEL_URL}/wake_up", timeout=3)
            logging.info("Waking up chat model.")
    except httpx.RequestError as e:
        logging.error(f"Error waking up chat model: {e}")

# --- Background Task ---
async def inactivity_monitor():
    while True:
        await asyncio.sleep(60)  # Check every minute
        if time.time() - last_activity_time > SLEEP_AFTER_SECONDS:
            await sleep_chat_model()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(inactivity_monitor())

# --- API Endpoints ---
@app.post("/activity")
async def record_activity():
    global last_activity_time
    last_activity_time = time.time()
    logging.info("Activity recorded.")
    return {"status": "activity recorded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9005)
