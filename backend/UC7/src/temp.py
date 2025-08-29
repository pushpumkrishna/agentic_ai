from crewai_tools import SerperDevTool  # Web search tool
from langchain.tools import tool  # Decorator to turn functions into tools
from langgraph.prebuilt import create_react_agent  # Helper to build ReAct-style agents
from langgraph.graph import START, StateGraph, END  # Core components of LangGraph
from typing import TypedDict  # Used to define structured state
import os

from backend.config.azure_models import AzureOpenAIModels


# --- Tool Definition ---
@tool
def serper_search(user_query: str) -> str:
    """
    Perform a real-time search using the Serper API.

    This tool takes a plain-text user query, sends it to Serper (a web search API),
    and returns a string with the top relevant results. It can be used by agents
    to gather up-to-date information from the internet as part of a reasoning or
    research task.

    Args:
        user_query (str): A natural language search prompt.

    Returns:
        str: A formatted string of search results from Serper.
    """
    return SerperDevTool().run(query=user_query)


llm = AzureOpenAIModels().get_azure_model_4()


# --- Define State ---
class AgentState(TypedDict):
    user_query: str
    answer: str

    # --- Define Node ---


def search_agent(state: AgentState) -> str:
    """
    Executes a ReAct-style agent that processes a user query.

    This function takes the current state (which includes the user's question),
    creates an agent using the Gemini language model and the `serper_search` tool,
    then runs the agent to get a response. The final answer is returned as updated state.

    Args:
        state (AgentState): A dictionary with the user's query.

    Returns:
        dict: Updated state with the generated answer.
    """
    agent = create_react_agent(llm, [serper_search])
    result = agent.invoke({"messages": state["user_query"]})
    return {"answer": result["messages"][-1].content}


# --- Define Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("search_agent", search_agent)

workflow.add_edge(START, "search_agent")
workflow.add_edge("search_agent", END)

app = workflow.compile()

from IPython.display import Image, display

Image(app.get_graph().draw_mermaid_png())

# --- Run Graph ---
result = app.invoke({"user_query": "Who won the IPL 2025 final?"})
print(result["answer"])


# --- Math Agent ---
def math_agent(state: AgentState) -> str:
    """
    A math-solving agent that uses the LLM to process and solve math problems.

    Args:
        state (AgentState): Contains the user's query.

    Returns:
        dict: Updated state with the computed answer from the LLM.
    """
    print("--- Math Node ---")
    prompt = (
        f"Solve this math problem and return only the answer: {state['user_query']}"
    )
    response = llm.invoke(prompt)
    state["answer"] = response.content.strip()
    return state


# --- Router Agent ---
def router_agent(state: AgentState) -> str:
    """
    Captures a user query from the command line and updates the state.

    This function acts as an input node in the LangGraph workflow. It prompts the user
    to enter a query via the console, then stores that input in the shared state under
    the 'user_query' key, which will be used to route to the appropriate agents.

    Args:
        state (AgentState): The current state dictionary (can be empty or partially filled).

    Returns:
        dict: Updated state containing the user's query.
    """
    print("--- Input Node ---")
    state["user_query"] = input("Input user query: ")
    return state


from typing import Literal

agent_docs = {"search_agent": search_agent.__doc__, "math_agent": math_agent.__doc__}


def routing_logic(state: AgentState) -> Literal["math_agent", "search_agent"]:
    """
    Uses the LLM to choose between 'math_agent' and 'search_agent'
    based on the intent of the user query and the agents' docstrings.

    Args:
        state (AgentState): The current state containing the user query.

    Returns:
        str: The name of the next node to route to.
    """
    prompt = f"""
    You are a router agent. Your task is to choose the best agent for the job.
    Here is the user query: {state["user_query"]}

    You can choose from the following agents:
    - math_agent: {agent_docs["math_agent"]}
    - search_agent: {agent_docs["search_agent"]}

    Which agent should handle this query? Respond with just the agent name.
    """
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    return "math_agent" if "math" in decision else "search_agent"


# --- Updated Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("router_agent", router_agent)  # Adds the new router agent to the flow
workflow.add_node("search_agent", search_agent)
workflow.add_node("math_agent", math_agent)  # Adds the math agent to the flow

workflow.add_edge(START, "router_agent")
workflow.add_conditional_edges("router_agent", routing_logic)
workflow.add_edge("search_agent", END)
workflow.add_edge("math_agent", END)

app = workflow.compile()

from IPython.display import Image, display

Image(app.get_graph().draw_mermaid_png())

app.invoke({})["answer"]  # test for search node
app.invoke({})["answer"]  # test for search node
