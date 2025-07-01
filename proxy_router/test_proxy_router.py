import pytest
import requests
import requests_mock
from fastapi.testclient import TestClient
from proxy_router.proxy_router import app

client = TestClient(app)

import os

@pytest.fixture
def mock_vllm(requests_mock):
    os.environ["VLLM_URL"] = "http://localhost:8002"
    requests_mock.get("http://localhost:8002/openapi.json", json={
        "paths": {
            "/search_file_content": {
                "get": {
                    "operationId": "search_file_content",
                    "description": "Searches for a regular expression pattern within the content of files.",
                    "parameters": [
                        {
                            "name": "pattern",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "x-openai-is_tool": True
                }
            }
        }
    })
    requests_mock.post("http://localhost:8002/v1/chat/completions", json={
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1234",
                            "type": "function",
                            "function": {
                                "name": "search_file_content",
                                "arguments": '{"pattern": "vllm"}'
                            }
                        }
                    ]
                }
            }
        ]
    })
    requests_mock.get("http://localhost:8002/search_file_content?pattern=vllm", json={"result": "found"})
    requests_mock.get("http://localhost:8002/openapi.json", json={
        "paths": {
            "/search_file_content": {
                "get": {
                    "operationId": "search_file_content",
                    "description": "Searches for a regular expression pattern within the content of files.",
                    "parameters": [
                        {
                            "name": "pattern",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "x-openai-is_tool": True
                }
            }
        }
    })
    requests_mock.get("http://localhost:8002/openapi.json", json={
        "paths": {
            "/search_file_content": {
                "get": {
                    "operationId": "search_file_content",
                    "description": "Searches for a regular expression pattern within the content of files.",
                    "parameters": [
                        {
                            "name": "pattern",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "x-openai-is_tool": True
                }
            }
        }
    })
    requests_mock.post("http://localhost:8002/v1/chat/completions", json={
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1234",
                            "type": "function",
                            "function": {
                                "name": "search_file_content",
                                "arguments": '{"pattern": "vllm"}'
                            }
                        }
                    ]
                }
            }
        ]
    })
    requests_mock.get("http://localhost:8002/search_file_content?pattern=vllm", json={"result": "found"})
    requests_mock.get("http://localhost:8002/search_file_content?pattern=vllm", json={"result": "found"})
    from proxy_router.proxy_router import fetch_and_cache_tool_definitions, TOOL_DEFINITIONS_CACHE
    fetch_and_cache_tool_definitions()
    print(f"TOOL_DEFINITIONS_CACHE in test: {TOOL_DEFINITIONS_CACHE}")

def test_end_to_end(mock_vllm):
    response = client.post("/chat", json={"conversation_id": "test_conv", "input": "search for vllm"})
    assert response.status_code == 200
    assert response.json() == {"conversation_id": "test_conv", "response": {"result": "found"}}


