import os
import logging
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
import httpx
import json

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Define the state
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]

# 2. Define the tools
def get_tools():
    try:
        response = httpx.get("http://localhost:9000/get_tools")
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        logging.error(f"Error fetching tools: {e}")
        return {"langgraph": [], "mcpo": []}

tools_data = get_tools()
all_tools = []
# LangGraph tools
for module in tools_data.get("langgraph", []):
    for tool in module.get("tools", []):
        # Prefix the tool name with the module name to avoid collisions
        tool["name"] = f"{module['name']}_{tool['name']}"
        all_tools.append(tool)
# MCPO tools
for server in tools_data.get("mcpo", []):
    for tool in server.get("tools", []):
        all_tools.append(tool)

def tool_executor(state):
    """
    Executes tools by calling the tools-api microservice.
    """
    logging.info("---Executing Tools---")
    tool_messages = []
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    available_tool_names = {tool["name"] for tool in all_tools}

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        if tool_name not in available_tool_names:
            error_msg = f"Error: Tool '{tool_name}' not found. Please select from the available tools."
            logging.error(error_msg)
            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
            continue

        logging.info(f"Calling tool: {tool_name} with args: {tool_args}")

        # The tool name sent to the tools-api should not be prefixed
        original_tool_name = tool_name
        if tool_name.startswith('yt_tools_'):
            original_tool_name = tool_name.replace('yt_tools_', '')
        
        run_tool_payload = {"tool_name": original_tool_name, "args": tool_args}
        
        try:
            with httpx.Client() as client:
                response = client.post("http://localhost:9000/run_tool", json=run_tool_payload, timeout=120)
                response.raise_for_status()
                tool_result = response.json()
                logging.info(f"Tool {tool_name} executed successfully. Result: {tool_result}")
                tool_messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tool_call_id))

        except httpx.RequestError as e:
            error_msg = f"Error calling tools-api for {tool_name}: {e}"
            logging.error(error_msg)
            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
        except Exception as e:
            error_msg = f"An unexpected error occurred while running tool {tool_name}: {e}"
            logging.error(error_msg)
            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))

    return {"messages": tool_messages}

# 3. Define the agent
model = ChatOpenAI(model="chat", temperature=0, streaming=True, base_url="http://localhost:8002/v1", api_key="dummy")

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

def should_continue(state):
    logging.info("---Checking for tool calls---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        logging.info("Tool calls found, continuing.")
        return "continue"
    logging.info("No tool calls found, ending.")
    return "end"

def call_model(state):
    logging.info("---Calling Model---")
    logging.info(f"Messages sent to model: {state['messages']}")
    response = model.invoke(state["messages"])
    logging.info(f"Model response: {response}")
    return {"messages": [response]}

# 4. Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_executor)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")
app_graph = workflow.compile()

# 5. Create the FastAPI app
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI(title="Orchestrator")

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

@app.post("/chat")
async def chat_endpoint(request: Request):
    logging.info("---Request Received---")
    try:
        body = await request.json()
        messages = body.get("messages", [])
        
        inputs = {"messages": messages}
        
        logging.info("---Invoking Graph---")
        response = app_graph.invoke(inputs)
        logging.info("---Graph Invocation Complete---")
        
        return {"response": response["messages"][-1].content}
    except Exception as e:
        logging.error(f"An error occurred in the chat endpoint: {e}")
        return {"error": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
