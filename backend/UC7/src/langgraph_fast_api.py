from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from tenacity import retry, wait_exponential, stop_after_attempt
from pydantic import BaseModel, constr
from fastapi import FastAPI
from backend.config.azure_models import AzureOpenAIModels
from backend.config.logging_lib import logger

# load_dotenv()
llm = AzureOpenAIModels().get_azure_model_4()


# Define state
def answer_question(state: dict) -> dict:
    user_input = state["user_input"]
    response = llm.invoke([HumanMessage(content=user_input)])
    return {"answer": response.content}


# Build the graph
workflow = StateGraph(dict)
workflow.add_node("answer", answer_question)
workflow.add_edge(START, "answer")
workflow.add_edge("answer", END)
graph = workflow.compile()


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def safe_invoke_llm(message):
    return llm.invoke([HumanMessage(content=message)])


def answer_question(state: dict) -> dict:
    user_input = state["user_input"]
    try:
        response = safe_invoke_llm(user_input)
        return {"answer": response.content}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


class RequestData(BaseModel):
    user_input: constr(min_length=1, max_length=500)  # limit input size


def answer_questions(state: dict) -> dict:
    logger.info(f"Received input: {state['user_input']}")
    response = safe_invoke_llm(state["user_input"])
    logger.info("LLM response generated")
    return {"answer": response.content}


app = FastAPI()


@app.post("/run")
async def run_workflow(data: RequestData):
    result = graph.invoke({"user_input": data.user_input})
    return {"result": result["answer"]}


"""
uvicorn backend.UC7.src.langgraph_fast_api:app --reload
"""
