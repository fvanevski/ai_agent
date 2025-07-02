import time
import unittest
from proxy_router.proxy_router import create_graph, AgentState
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

class PerformanceTest(unittest.TestCase):

    def test_parallel_execution_performance(self):
        # Mock tools with an artificial delay
        def tool1(arg):
            """Tool 1"""
            time.sleep(1)
            return f"tool1 executed with {arg}"
        tool1.__name__ = "tool1"

        def tool2(arg):
            """Tool 2"""
            time.sleep(1)
            return f"tool2 executed with {arg}"
        tool2.__name__ = "tool2"

        tools = [tool1, tool2]
        tool_node = ToolNode(tools)

        # Mock the state
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tool1",
                            "args": {"arg": "arg1"},
                            "id": "1",
                        },
                        {
                            "name": "tool2",
                            "args": {"arg": "arg2"},
                            "id": "2",
                        },
                    ],
                )
            ]
        }

        # Measure the execution time
        start_time = time.time()
        tool_node.invoke(state)
        end_time = time.time()

        execution_time = end_time - start_time

        # Check if the execution time is less than the sum of individual delays
        self.assertLess(execution_time, 2)

if __name__ == '__main__':
    unittest.main()