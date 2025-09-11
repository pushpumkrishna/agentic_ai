import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

class AgentState(TypedDict):
    """
    The Agent’s Memory (state.py)
    First, we define the agent’s memory.
    This is a central Python TypedDict that will hold all the information that persists between the steps of our graph.
    We'll include a list for the messages and a special list for intermediate_steps to explicitly track the
    agent's tool usage.

    Represents the state of our agent.
    Attributes:
        messages: A list of all messages in the conversation.
        intermediate_steps: A list of action-observation pairs for logging.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    intermediate_steps: Annotated[List[tuple[AIMessage, ToolMessage]], operator.add