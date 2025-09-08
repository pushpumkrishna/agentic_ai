import random
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph
from backend.config.azure_models import AzureOpenAIModels
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod


class TransactionState(TypedDict):
    transaction: Dict[str, Any]
    risk_score: int
    rule_based_flag: bool
    ml_fraud_probability: float
    final_decision: str


def create_initial_state(transaction: Dict[str, Any]) -> TransactionState:
    return {
        "transaction": transaction,
        "risk_score": 0,
        "rule_based_flag": False,
        "ml_fraud_probability": 0.0,
        "final_decision": "Pending"
    }


"Transaction Monitoring Agent:: Logs incoming transactions."


def transaction_monitoring_agent(state: TransactionState) -> TransactionState:
    print("Monitoring Transaction:", state["transaction"])
    return state


"Rule-Based Detection Agent:: Flags transactions over $1000 as risky."


def rule_based_detection_agent(state: TransactionState) -> TransactionState:
    if state["transaction"]["amount"] > 1000:
        state["rule_based_flag"] = True
    print("Rule-based Fraud Detection:", state["rule_based_flag"])
    return state


"Machine Learning Agent:: Generates a fraud probability using a random number."


def machine_learning_agent(state: TransactionState) -> TransactionState:
    state["ml_fraud_probability"] = random.uniform(0, 1)
    print("ML Fraud Probability:", state["ml_fraud_probability"])
    return state


"Risk Scoring Agent:: Assigns a random risk score between 1 and 100."


def risk_scoring_agent(state: TransactionState) -> TransactionState:
    state["risk_score"] = random.randint(1, 100)
    print("Risk Score Assigned:", state["risk_score"])
    return state


"""LLM-Based Decision Agent
Uses LLM to analyze the data and decide whether to approve or decline.
"""


def alert_decision_agent(state: TransactionState) -> TransactionState:
    prompt = (
        f"Transaction Details: {state['transaction']}. "
        f"Risk Score: {state['risk_score']}, "
        f"Rule-Based Flag: {state['rule_based_flag']}, "
        f"ML Probability: {state['ml_fraud_probability']}. "
        "Should this transaction be Approved or Declined?"
    )

    llm = AzureOpenAIModels().get_azure_model_4()
    response = llm.invoke(prompt)
    print("Final Decision (LLM-Based):", response)
    return state


"Adding Nodes:: We define each step in our fraud detection graph:"

graph = StateGraph(TransactionState)
graph.add_node("monitor", transaction_monitoring_agent)
graph.add_node("risk_score1", risk_scoring_agent)
graph.add_node("rule_based", rule_based_detection_agent)
graph.add_node("ml_model", machine_learning_agent)
graph.add_node("decision", alert_decision_agent)

"""Connecting Nodes: We define the execution sequence"""

graph.set_entry_point("monitor")
graph.add_edge("monitor", "risk_score1")
graph.add_edge("risk_score1", "rule_based")
graph.add_edge("rule_based", "ml_model")
graph.add_edge("ml_model", "decision")

"""Visualizing the Graph: LangGraph allows us to visualize the pipeline using Mermaid.js."""

fraud_detection_pipeline = graph.compile()
display(
    Image(
        fraud_detection_pipeline.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
    )
)

"""Running a Sample Transaction: Now, we can test our fraud detection system with a sample transaction."""

transaction_ = {"amount": 1200, "location": "New York", "device_id": "XYZ123"}
state = create_initial_state(transaction_)
final_state = fraud_detection_pipeline.invoke(state)
print("Fraud Detection Completed: Final Decision -", final_state["final_decision"])
