# tools-api.py – AI Agent tool micro‑service
"""
* FastAPI service that exposes:
    • **GET  /get_tools**       — returns a list of available tools
    • **POST  /run_tool**       — runs a tool and returns the result

"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from yt_tools import summarize_url, transcribe_url
import yaml
import os
import requests
import json
import importlib
from typing import Dict, Any, List

app = FastAPI(title="AI Agent Tools API")

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

# --- Pydantic Models ---

class RunToolRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any]

# --- Tool Discovery and Caching ---

TOOL_CACHE: Dict[str, Dict[str, Any]] = {}

def get_langgraph_tools() -> List[Dict[str, Any]]:
    """Loads tools defined in tools.yaml and structures them by module."""
    try:
        with open("tools.yaml", "r") as f:
            tools_data = yaml.safe_load(f)
            # Restructure the data to match the desired format
            langgraph_tools = []
            for module_name, tools in tools_data.get("LangGraph-Tools", {}).items():
                module_tools = []
                for tool in tools:
                    tool['meta'] = {'type': 'langgraph', 'module': module_name}
                    module_tools.append(tool)
                langgraph_tools.append({
                    "name": module_name,
                    "tools": module_tools
                })
            return langgraph_tools
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error reading or parsing tools.yaml: {e}")
        return []

def discover_mcpo_tools() -> List[Dict[str, Any]]:
    """Fetches tool definitions from MCPO servers and groups them by server."""
    servers = []
    try:
        with open("tools.yaml", "r") as f:
            tools_data = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error reading or parsing tools.yaml: {e}")
        return []

    for server_name, server_url in tools_data.get("MCPO Servers", {}).items():
        server_tools = []
        try:
            response = requests.get(f"{server_url}/openapi.json", timeout=5)
            response.raise_for_status()
            openapi_schema = response.json()

            for path, path_item in openapi_schema.get("paths", {}).items():
                for method, operation in path_item.items():
                    if "operationId" in operation:
                        params = {}
                        if "requestBody" in operation:
                            content = operation["requestBody"].get("content", {})
                            if "application/json" in content:
                                params = content["application/json"].get("schema", {})
                        
                        tool_def = {
                            "name": operation["operationId"],
                            "description": operation.get("summary") or operation.get("description", ""),
                            "parameters": params,
                            "meta": {
                                "type": "mcpo",
                                "server_name": server_name,
                                "server_url": server_url,
                                "path": path,
                                "method": method.lower()
                            }
                        }
                        server_tools.append(tool_def)
            
            if server_tools:
                servers.append({
                    "name": server_name,
                    "url": server_url,
                    "tools": server_tools
                })

        except requests.exceptions.RequestException as e:
            print(f"Could not fetch tools from MCPO server '{server_name}': {e}")
        except json.JSONDecodeError as e:
            print(f"Error parsing openapi.json from '{server_name}': {e}")
    return servers

def populate_tool_cache():
    """Populates the in-memory tool cache."""
    global TOOL_CACHE
    TOOL_CACHE = {"langgraph": [], "mcpo": []} # Re-initialize cache
    print("Populating tool cache...")
    
    # Process LangGraph tools
    TOOL_CACHE["langgraph"] = get_langgraph_tools()

    # Process MCPO servers
    TOOL_CACHE["mcpo"] = discover_mcpo_tools()
            
    print(f"Tool cache populated.")

@app.on_event("startup")
def on_startup():
    """On startup, populate the tool cache."""
    populate_tool_cache()

# --- API Endpoints ---

@app.get("/get_tools")
def get_tools():
    """Returns a list of all available tools, grouped by type."""
    return TOOL_CACHE


@app.post("/run_tool")
def run_tool(req: RunToolRequest):
    """Runs a tool by name with the given arguments."""
    tool_def = None
    
    # Search for the tool in LangGraph tools
    for module in TOOL_CACHE.get("langgraph", []):
        for tool in module.get("tools", []):
            if tool["name"] == req.tool_name:
                tool_def = tool
                break
        if tool_def:
            break

    # If not found, search in MCPO tools
    if not tool_def:
        for server in TOOL_CACHE.get("mcpo", []):
            for tool in server.get("tools", []):
                if tool["name"] == req.tool_name:
                    tool_def = tool
                    break
            if tool_def:
                break

    if not tool_def:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found.")

    tool_meta = tool_def.get("meta", {})
    tool_type = tool_meta.get("type")

    if tool_type == "langgraph":
        run_command = tool_def.get("run")
        if not run_command:
            raise HTTPException(status_code=500, detail="LangGraph tool has no 'run' command.")
        
        try:
            # The run command is expected to be in the format "from <module> import <function>"
            parts = run_command.split()
            module_name = parts[1]
            function_name = parts[3]
            
            module = importlib.import_module(module_name)
            function_to_run = getattr(module, function_name)
            result = function_to_run(**req.args)
            return {"result": result}
        except (ImportError, AttributeError, TypeError, Exception) as e:
            raise HTTPException(status_code=500, detail=f"Error running LangGraph tool '{req.tool_name}': {e}")

    elif tool_type == "mcpo":
        server_url = tool_meta.get("server_url")
        path = tool_meta.get("path")
        method = tool_meta.get("method", "post")
        
        endpoint_url = f"{server_url}{path}"
        
        try:
            response = requests.request(method, endpoint_url, json=req.args, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error calling MCPO tool '{req.tool_name}': {e}")

    else:
        raise HTTPException(status_code=500, detail=f"Unknown tool type for '{req.tool_name}'.")

# ---------------------------------------------------------------------------
# Run with: python tools-api.py  (or uvicorn tools-api:app --reload)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 9000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1")
    
    uvicorn.run("tools-api:app", host=host, port=port, reload=reload)