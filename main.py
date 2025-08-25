# import importlib
#
#
# def main():
#     print("Available use cases:")
#     print("1 - use_case_1")
#     print("2 - use_case_2")
#     print("app1 - flask_apps.app1")
#     print("app2 - flask_apps.app2")
#     print("app3 - flask_apps.app3")
#     choice = input("Enter use case to run: ").strip()
#
#     # Mapping choices to module paths
#     use_case_map = {
#         "1": "use_cases.use_case_1",
#         "2": "use_cases.use_case_2",
#         "app1": "use_cases.flask_apps.app1.server",
#         "app2": "use_cases.flask_apps.app2.server",
#         "app3": "use_cases.flask_apps.app3.server",
#     }
#
#     module_name = use_case_map.get(choice)
#     if not module_name:
#         print("Invalid choice!")
#         return
#
#     module = importlib.import_module(module_name)
#     # Expect each module to have a run() function
#     module.run()
#
#
# if __name__ == "__main__":
#     main()


import asyncio
import logging

from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.client import Client  # Use unified client API :contentReference[oaicite:0]{index=0}

from backend.UC7.src.langgraph_fast_api import RequestData, graph

#
# # ----- 1. Define your MCP server -----
#
# mcp = FastMCP("Local Test Server")
#
# @mcp.tool()
# def add(a: int, b: int) -> int:
#     """Tool: Add two numbers."""
#     return a + b
#
# @mcp.resource("config://version")
# def version() -> str:
#     """Resource: Returns current version."""
#     return "v2.0-test"
#
# @mcp.prompt()
# def greet(name: str) -> str:
#     """Prompt: Generate greeting message."""
#     return f"Hello, {name}!"
#
# # ----- 2. Define integrated client test -----
#
# async def run_client_tests():
#     async with Client(mcp) as client:
#         # List capabilities
#         tools = await client.list_tools()
#         resources = await client.list_resources()
#         prompts = await client.list_prompts()
#         logger.info("Tools: %s", tools)
#         logger.info("Resources: %s", resources)
#         logger.info("Prompts: %s", prompts)
#
#         # Test tool
#         add_result = await client.call_tool("add", {"a": 4, "b": 6})
#         logger.info("add(4, 6) → %s", add_result)
#
#         # Test resource
#         version_str = await client.read_resource("config://version")
#         logger.info("Version resource → %s", version_str)
#
#         # Test prompt
#         greet_result = await client.get_prompt("greet", {"name": "ChatGPT"})
#         logger.info("Greet prompt → %s", greet_result)
#
# # ----- 3. Main runner to start server and client -----
#
# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#
#     # Use an in-memory transport by passing the server directly to the client
#     loop.run_until_complete(run_client_tests())


