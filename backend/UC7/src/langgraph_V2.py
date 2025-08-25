import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Annotated
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages

from backend.config.azure_models import AzureOpenAIModels

llm = AzureOpenAIModels().get_azure_model_4()

"""
* Basic LangGraph concepts and setup
* Creating simple linear workflows
* Building conditional flows
* State management across nodes
* Integration with OpenAI's GPT models
"""


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    current_step: str
    user_input: str
    analysis_result: str
    summary: str
    final_response: str


# Each node is a function that updates this state:
def analyze_input(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following text and identify the main topic, sentiment, and key points:\n\n{text}\n\n"
        "Provide a structured analysis."
    )
    chain = prompt | llm
    llm_result = chain.invoke({"text": state["user_input"]})
    return {
        "analysis_result": str(llm_result.content),
        "current_step": "analysis_complete",
        "messages": [result],
    }


def summarize_analysis(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        "Create a concise summary of the following analysis:\n\n{analysis}\n\nSummary:"
    )
    chain = prompt | llm
    llm_result = chain.invoke({"analysis": state["analysis_result"]})
    return {
        "summary": str(llm_result.content),
        "current_step": "summary_complete",
        "messages": [result],
    }


def generate_response(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        "Based on the analysis and summary below, generate a helpful response to the user:\n\n"
        "Analysis: {analysis}\n\nSummary: {summary}\n\nOriginal Input: {input}\n\nResponse:"
    )
    chain = prompt | llm
    llm_result = chain.invoke(
        {
            "analysis": state["analysis_result"],
            "summary": state["summary"],
            "input": state["user_input"],
        }
    )
    return {
        "final_response": str(llm_result.content),
        "current_step": "complete",
        "messages": [llm_result],
    }


"""
Creating Your LangGraph Workflow

* Analyze input
* Summarize the analysis
* Generate a final response
"""


def create_basic_workflow():
    """Create a basic linear LangGraph workflow"""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_input)
    workflow.add_node("summarize", summarize_analysis)
    workflow.add_node("respond", generate_response)

    # Add edges (defines flow)
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


basic_app = create_basic_workflow()
print("✅ Basic workflow compiled!")


# Running the Workflow
user_input = (
    "I'm really excited about learning LangGraph! It seems like a powerful tool for building complex AI workflows. "
    "I can see many potential applications in my work."
)

print("🚀 Running workflow...\n")

try:
    result = basic_app.invoke({"user_input": user_input})

    print("=" * 50)
    print("📊 ANALYSIS RESULT")
    print("=" * 50)
    print(result["analysis_result"])

    print("\n" + "=" * 50)
    print("📝 SUMMARY")
    print("=" * 50)
    print(result["summary"])

    print("\n" + "=" * 50)
    print("💬 FINAL RESPONSE")
    print("=" * 50)
    print(result["final_response"])

    print("\n✅ Workflow completed successfully!")

except Exception as e:
    print(f"❌ Error running workflow: {e}")

"""
Building Conditional Workflows
LangGraph supports conditional logic to dynamically route the workflow based on decisions made at runtime.

In this section:

- Classify the input (technical / creative / general)
- Route to the appropriate handler node
🧾 State Update
"""


class ConditionalAgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    current_step: str
    user_input: str
    classification: str
    final_response: str


def classify_input(state: ConditionalAgentState) -> ConditionalAgentState:
    prompt = ChatPromptTemplate.from_template(
        "Classify the following text as either 'technical', 'creative', or 'general':\n\n{text}\n\nClassification "
        "(respond with just one word):"
    )
    chain = prompt | llm
    llm_result = chain.invoke({"text": state["user_input"]})
    return {
        "classification": llm_result.content.strip().lower(),
        "current_step": "classified",
        "messages": [result],
    }


"""
🧠 Handler Nodes
"""


def technical_handler(state: ConditionalAgentState) -> ConditionalAgentState:
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical analysis of the following text, including implementation details and "
        "best practices:\n\n{text}"
    )
    chain = prompt | llm
    llm_result = chain.invoke({"text": state["user_input"]})
    return {
        "final_response": str(llm_result.content),
        "current_step": "technical_complete",
        "messages": [llm_result],
    }


def creative_handler(state: ConditionalAgentState) -> ConditionalAgentState:
    prompt = ChatPromptTemplate.from_template(
        "Provide a creative interpretation of the following text, focusing on mood, theme, and "
        "literary style:\n\n{text}"
    )
    chain = prompt | llm
    llm_result = chain.invoke({"text": state["user_input"]})
    return {
        "final_response": str(llm_result.content),
        "current_step": "creative_complete",
        "messages": [result],
    }


def general_handler(state: ConditionalAgentState) -> ConditionalAgentState:
    prompt = ChatPromptTemplate.from_template(
        "Provide a general overview of the following text, summarizing the main ideas and relevance:\n\n{text}"
    )
    chain = prompt | llm
    llm_result = chain.invoke({"text": state["user_input"]})
    return {
        "final_response": str(llm_result.content),
        "current_step": "general_complete",
        "messages": [result],
    }


"""
🧭 Decision
Function
"""


def decide_path(state: ConditionalAgentState) -> str:
    classification = state.get("classification", "").lower()
    if "technical" in classification:
        return "technical"
    elif "creative" in classification:
        return "creative"
    return "general"


"""
🧩 Create Conditional Workflow
"""


def create_conditional_workflow():
    workflow = StateGraph(ConditionalAgentState)

    workflow.add_node("classify", classify_input)
    workflow.add_node("technical", technical_handler)
    workflow.add_node("creative", creative_handler)
    workflow.add_node("general", general_handler)

    workflow.add_conditional_edges(
        "classify",
        decide_path,
        {"technical": "technical", "creative": "creative", "general": "general"},
    )

    workflow.add_edge(START, "classify")
    workflow.add_edge("technical", END)
    workflow.add_edge("creative", END)
    workflow.add_edge("general", END)

    return workflow.compile()


conditional_app = create_conditional_workflow()
print("✅ Conditional workflow created!")


"""
 Testing the Conditional Workflow
We’ll now test our conditional LangGraph pipeline with different types of input:

* Technical
* Creative
* General
🧪 Test: Technical Input
"""

print("🔧 Testing TECHNICAL input...\n")

user_input = (
    "I'm building a REST API using FastAPI and PostgresSQL. "
    "How should I handle authentication and background task processing efficiently?"
)
try:
    result = conditional_app.invoke({"user_input": user_input})
    print(f"📊 Classification: {result['classification']}")
    print(f"🎯 Final Step: {result['current_step']}")
    print("\n💬 Final Response:\n")
    print(result["final_response"])
except Exception as e:
    print(f"❌ Error: {e}")

"""
🎨 Test: Creative Input
"""

print("\n🎨 Testing CREATIVE input...\n")
user_input = ("The wind whispered secrets through the trees as the moon cast shadows like forgotten "
              "dreams across the field.")

try:
    result = conditional_app.invoke({"user_input": user_input})
    print(f"📊 Classification: {result['classification']}")
    print(f"🎯 Final Step: {result['current_step']}")
    print("\n💬 Final Response:\n")
    print(result["final_response"])
except Exception as e:
    print(f"❌ Error: {e}")
