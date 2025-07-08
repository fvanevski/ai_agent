import unittest
from unittest.mock import MagicMock, patch
from proxy_router.proxy_router import create_graph, AgentState, call_model, should_continue
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from proxy_router.proxy_router import call_tool_node

class TestAgent(unittest.TestCase):

    def test_graph_creation(self):
        """
        Tests if the graph is created without errors.
        """
        graph = create_graph()
        self.assertIsNotNone(graph)

    @patch('proxy_router.proxy_router.httpx.Client')
    def test_call_model(self, mock_client):
        """
        Tests the call_model function to ensure it makes a request to the vLLM
        and processes the response correctly.
        """
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello, how can I help you?"}}]
        }
        mock_client.return_value.__enter__.return_value.post.return_value = mock_response

        state = AgentState(
            conversation_id="test_conv",
            input="Hello",
            chat_history=[],
            agent_outcome=[],
            intermediate_steps=[]
        )

        # Act
        result = call_model(state)

        # Assert
        self.assertIn("agent_outcome", result)
        self.assertEqual(len(result["agent_outcome"]), 1)
        self.assertEqual(result["agent_outcome"][0]["choices"][0]["message"]["content"], "Hello, how can I help you?")

    def test_should_continue(self):
        """
        Tests the should_continue function to ensure it correctly determines
        the next step in the graph.
        """
        # Test case 1: No agent_outcome
        state1 = AgentState(
            conversation_id="test_conv",
            input="Hello",
            chat_history=[],
            agent_outcome=[],
            intermediate_steps=[]
        )
        self.assertEqual(should_continue(state1), "end")

        # Test case 2: agent_outcome with no tool_calls
        state2 = AgentState(
            conversation_id="test_conv",
            input="Hello",
            chat_history=[],
            agent_outcome=[{"content": "No tools needed"}],
            intermediate_steps=[]
        )
        self.assertEqual(should_continue(state2), "end")

        # Test case 3: agent_outcome with tool_calls
        state3 = AgentState(
            conversation_id="test_conv",
            input="Hello",
            chat_history=[],
            agent_outcome=[{"tool_calls": [{"name": "search", "args": {}}]}],
            intermediate_steps=[]
        )
        self.assertEqual(should_continue(state3), "continue")

class TestToolNode(unittest.TestCase):

    def test_tool_node_execution(self):
        # Mock the search_file_content tool
        def search_file_content_mock(pattern: str):
            """Searches for a regular expression pattern within the content of files."""
            return "test content"
        search_file_content_mock.__name__ = "search_file_content"

        # Create a new graph for testing
        tools = [search_file_content_mock]
        tool_node = ToolNode(tools)

        # Mock the state
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_file_content",
                            "args": {"pattern": "test"},
                            "id": "1",
                        }
                    ],
                )
            ]
        }

        # Invoke the tool node directly
        with patch('proxy_router.proxy_router.tool_node', tool_node):
            result = call_tool_node(state)

        # Check the result
        self.assertIn("intermediate_steps", result)
        self.assertEqual(len(result["intermediate_steps"]), 1)
        self.assertEqual(result["intermediate_steps"][0].content, "test content")

    def test_tool_node_error_handling(self):
        """
        Tests that the ToolNode correctly handles exceptions raised by a tool.
        """
        # 1. Define a mock tool that always raises an exception
        def error_tool(x: int, y: int) -> int:
            """A tool that always raises a ValueError."""
            raise ValueError("This tool always fails.")
        error_tool.__name__ = "error_tool"

        # 2. Create a ToolNode with this mock tool
        tools = [error_tool]
        tool_node = ToolNode(tools)

        # 3. Define the input state for the ToolNode
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "error_tool",
                            "args": {"x": 1, "y": 2},
                            "id": "1",
                        }
                    ],
                )
            ]
        }

        # 4. Invoke the ToolNode and assert that it returns a ToolMessage with the error
        with patch('proxy_router.proxy_router.tool_node', tool_node):
            result = call_tool_node(state)
        self.assertIn("intermediate_steps", result)
        self.assertEqual(len(result["intermediate_steps"]), 1)
        tool_message = result["intermediate_steps"][0]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertIn("Error: ValueError('This tool always fails.')", tool_message.content)
        self.assertEqual(tool_message.name, "error_tool")

    def test_conversation_history(self):
        """
        Tests that the conversation history is correctly managed.
        """
        # Arrange
        state = AgentState(
            conversation_id="test_conv",
            input="Hello",
            chat_history=[],
            agent_outcome=[],
            intermediate_steps=[]
        )

        # Act
        with patch('proxy_router.proxy_router.httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Hello, how can I help you?"}}]
            }
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = call_model(state)

        # Assert
        self.assertEqual(len(result["agent_outcome"]), 1)
        self.assertEqual(result["agent_outcome"][0]["choices"][0]["message"]["content"], "Hello, how can I help you?")

        # Arrange
        state = AgentState(
            conversation_id="test_conv",
            input="What is the weather in London?",
            chat_history=result["agent_outcome"],
            agent_outcome=[],
            intermediate_steps=[]
        )

        # Act
        with patch('proxy_router.proxy_router.httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "The weather in London is sunny."}}]
            }
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = call_model(state)

        # Assert
        self.assertEqual(len(result["agent_outcome"]), 1)
        self.assertEqual(result["agent_outcome"][0]["choices"][0]["message"]["content"], "The weather in London is sunny.")

if __name__ == '__main__':
    unittest.main()