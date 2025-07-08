import os
from dotenv import load_dotenv
import logging
import json
import httpx
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import traceback

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Environment Variables ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '.env'))
TOOLS_API_URL = os.getenv("TOOLS_API_URL", "http://localhost:9000")
VLLM_AGENT_URL = os.getenv("VLLM_AGENT_URL", "http://localhost:8002/v1")
MODEL_CONTEXT_LENGTH = int(os.getenv("MODEL_CONTEXT_LENGTH", "8192"))



# 2. Define the tools
def get_tools():
    try:
        response = httpx.get(f"{TOOLS_API_URL}/get_tools")
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        logging.error(f"Error fetching tools: {e}")
        return {"langgraph": [], "mcpo": []}

tools_data = get_tools()
all_tools = []
for module in tools_data.get("langgraph", []):
    for tool in module.get("tools", []):
        # Do not prefix tool name with module name; use original tool name
        all_tools.append(tool)
for server in tools_data.get("mcpo", []):
    for tool in server.get("tools", []):
        all_tools.append(tool)



# 3. Define the agent
model = ChatOpenAI(model="chat", temperature=0, streaming=True, base_url=VLLM_AGENT_URL, api_key="dummy")

formatted_tools = []
for tool in all_tools:
    description = tool.get("description") or tool.get("summary", "")
    formatted_tools.append(
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": description,
                "parameters": tool.get("parameters", {}),
            },
        }
    )

model = model.bind_tools(formatted_tools)
logging.info(f"Formatted tools: {formatted_tools}")



def call_model(state, enabled_tool_names=None):
    logging.info("---Calling Model (OpenAI API direct)---")
    # Build OpenAI-compatible messages list
    system_prompt = (
        "You are a powerful and intelligent AI assistant. "
        "You have access to a variety of tools to help you answer user queries. "
        "Only use the provided tools if they are relevant to the user's query. "
        "If the tools are not relevant, you must answer the user's query directly without mentioning the tools."
    )
    # Convert LangChain messages to OpenAI format
    def to_openai_message(msg):
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        if hasattr(msg, "role") and hasattr(msg, "content"):
            return {"role": getattr(msg, "role", "user"), "content": msg.content}
        return msg  # fallback

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    for m in state['messages']:
        # If already OpenAI format, keep as is
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append(m)
        else:
            messages.append(to_openai_message(m))

    logging.info(f"OpenAI messages: {messages}")

    # Filter tools by enabled_tool_names if provided
    if enabled_tool_names is not None:
        filtered_tools = [tool for tool in all_tools if tool["name"] in enabled_tool_names]
    else:
        filtered_tools = all_tools

    formatted_tools_local = []
    for tool in filtered_tools:
        description = tool.get("description") or tool.get("summary", "")
        formatted_tools_local.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": description,
                    "parameters": tool.get("parameters", {}),
                },
            }
        )

    tools_payload = formatted_tools_local

    # vLLM expects POST /v1/chat/completions
    url = VLLM_AGENT_URL.replace("/v1", "") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Loop: keep sending messages until no tool_calls are returned
    MAX_TOOL_RESULT_CHARS = 4000  # Truncate tool results to avoid context overflow
    SAFETY_MARGIN = 2000  # Reserve tokens for prompt, user, and assistant messages
    while True:
        payload = {
            "model": "chat",
            "messages": messages,
            "tools": tools_payload,
            "tool_choice": "auto"
        }
        # Log the full JSON payload in a pretty format for debugging
        logging.info("=== Payload sent to vLLM ===\n%s", json.dumps(payload, indent=2, ensure_ascii=False))
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 400 and "maximum context length" in resp.text:
                logging.error(f"vLLM context overflow: {resp.text}")
                return {"messages": [{"role": "assistant", "content": "[Error: The model's context window was exceeded. The tool result was too large to process. Please ask for a more specific or summarized result, or try again with a narrower query.]"}]}
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            logging.error(f"Error calling vLLM API: {e}")
            return {"messages": [{"role": "assistant", "content": f"[Error calling vLLM API: {e}]"}]}

        logging.info(f"vLLM response: {result}")
        # vLLM returns a 'choices' list
        choice = result["choices"][0]
        message = choice["message"]
        # If tool_calls present, execute them
        if "tool_calls" in message and message["tool_calls"]:
            tool_results = []
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"].get("arguments", "{}"))
                tool_call_id = tool_call["id"]
                # Patch: If Context7_get-library-docs, set tokens param based on MODEL_CONTEXT_LENGTH
                if tool_name == "Context7_get-library-docs":
                    if "tokens" not in tool_args or not tool_args["tokens"]:
                        # Estimate available tokens: context length - safety margin
                        tool_args["tokens"] = MODEL_CONTEXT_LENGTH - SAFETY_MARGIN
                        logging.info(f"Setting tokens for Context7_get-library-docs to {tool_args['tokens']} (MODEL_CONTEXT_LENGTH={MODEL_CONTEXT_LENGTH}, SAFETY_MARGIN={SAFETY_MARGIN})")
                logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
                run_tool_payload = {"tool_name": tool_name, "args": tool_args}
                try:
                    with httpx.Client() as client:
                        tool_resp = client.post(f"{TOOLS_API_URL}/run_tool", json=run_tool_payload, timeout=120)
                        tool_resp.raise_for_status()
                        tool_result = tool_resp.json()
                        tool_result_str = json.dumps(tool_result)
                        if len(tool_result_str) > MAX_TOOL_RESULT_CHARS:
                            tool_result_str = tool_result_str[:MAX_TOOL_RESULT_CHARS] + "\n[Result truncated due to context limits. Ask for more details if needed.]"
                        logging.info(f"Tool {tool_name} executed. Result: {tool_result_str[:200]}... (truncated)" if len(tool_result_str) > 200 else f"Tool {tool_name} executed. Result: {tool_result_str}")
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_result_str
                        })
                except Exception as e:
                    logging.error(f"Error executing tool {tool_name}: {e}")
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": f"[Error executing tool: {e}]"
                    })
            # Append tool results to messages and continue loop
            messages.append(message)
            messages.extend(tool_results)
            continue
        # No tool calls: return the model's message as the final answer
        messages.append(message)
        return {"messages": [message]}



# 5. Create the FastAPI app
app = FastAPI(title="Agent Service")

origins = ["http://localhost:3000", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat_endpoint(request: Request):
    logging.info("---Request Received---")
    def safe_get_content(msg):
        if hasattr(msg, "content"):
            return msg.content
        if isinstance(msg, dict):
            return msg.get("content") or msg
        return str(msg)
    try:
        body = await request.json()
        messages = body.get("messages", [])
        enabled_tools = body.get("enabled_tools")
        # enabled_tools is expected to be a dict: {"langgraph": ["mod1", ...], "mcpo": ["server1", ...]}
        enabled_server_names = set()
        if isinstance(enabled_tools, dict):
            enabled_server_names.update(enabled_tools.get("langgraph", []))
            enabled_server_names.update(enabled_tools.get("mcpo", []))
        else:
            # fallback: treat as flat list
            enabled_server_names = set(enabled_tools or [])

        # Filter tools by enabled servers/modules
        filtered_tool_names = [
            tool["name"]
            for module in tools_data.get("langgraph", [])
            if module["name"] in enabled_server_names
            for tool in module.get("tools", [])
        ]
        filtered_tool_names += [
            tool["name"]
            for server in tools_data.get("mcpo", [])
            if server["name"] in enabled_server_names
            for tool in server.get("tools", [])
        ]

        logging.info(f"Enabled servers/modules: {enabled_server_names}")
        logging.info(f"Enabled tool names: {filtered_tool_names}")

        response = call_model({"messages": messages}, enabled_tool_names=filtered_tool_names)
        last_msg = response["messages"][-1]
        return {"response": safe_get_content(last_msg)}
    except Exception as e:
        logging.error(f"An error occurred in the chat endpoint: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Agent service error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug", reload=True)
