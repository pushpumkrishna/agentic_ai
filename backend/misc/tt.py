# LangGraph core components for graph building and execution
from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph, START, END  # For defining state and building the graph

# LangChain message types
from langchain_core.messages import HumanMessage  # For creating user messages to send to the model

# For visualizing the LangGraph (works in Jupyter/IPython environments)
from IPython.display import Image, display  # To render the graph as a Mermaid diagram

from backend.config.azure_models import AzureOpenAIModels

load_dotenv()

# Optional: Subclass MessagesState if you want to add extra keys to the state later
class MessagesState(MessagesState):
    # 'messages' key is already built-in
    pass


# Initialize the Groq Chat Model (LLaMA 3)
llm = AzureOpenAIModels().get_azure_model_4()


# Define a simple tool that the model can call
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Bind the tool to the model
llm_with_tools = llm.bind_tools([multiply])


# Define a LangGraph node to call the LLM and return updated message state
def tool_calling_llm(state: MessagesState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }


# Build the graph using LangGraph
builder = StateGraph(MessagesState)

# Add the node that handles tool + LLM logic
builder.add_node("tool_calling_llm", tool_calling_llm)

# Define the edges: START → tool_calling_llm → END
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

# Compile the graph
graph = builder.compile()

# Optional: Visualize the graph using Mermaid diagram
# display(Image(graph.get_graph().draw_mermaid_png()))

# Run the graph with a HumanMessage input
response = graph.invoke({
    "messages": HumanMessage(content="what is 2 x 3", name="Venky")
})

# Print the messages returned in final state
for m in response["messages"]:
    m.pretty_print()
