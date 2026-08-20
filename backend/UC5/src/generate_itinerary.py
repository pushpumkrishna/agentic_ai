from langchain_core.messages import HumanMessage
import langchain_ollama
import json


def generate_itinerary(state):
    llm = langchain_ollama.ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    prompt = f"""
    Using the following preferences, create a detailed itinerary:
    {json.dumps(state["preferences"], indent=2)}

    Include sections for each day, dining options, and downtime.
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"itinerary": result.strip()}
    except Exception as e:
        return {"itinerary": "", "warning": str(e)}
