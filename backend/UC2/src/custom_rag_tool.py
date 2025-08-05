# Create custom RAG tool
from crewai.tools import tool
from backend.UC2.src.vector_database import VectorDatabase
from backend.config.azure_models import AzureOpenAIModels
from backend.config.logging_lib import logger


@tool("RAG Tool")
def rag_tool(question: str) -> str:
    """Tool to search for relevant information from a vector database."""

    model, company_db = VectorDatabase().populate()

    # Encode the question
    query_vec = model.encode(question)

    # Get top 5 similar vector
    results = company_db.search(query_vec, top_k=5)

    # Build context from the results
    context = "\n".join([f"- {res['metadata']['sentence']}" for res in results])

    # Create the prompt
    prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.

            Context:
            {context}

            Question: {question}

            Answer:
            """

    # Generate an answer using the context
    client = AzureOpenAIModels().get_azure_model_4()

    response = client.invoke(model="gpt-4o-mini", input=prompt)

    answer = response.content
    total_tokens = response.response_metadata["token_usage"].get("token_usage", 0)

    logger.info(f"\n{'*' * 100}\nTotal tokens used : {total_tokens}\n{'*' * 100}")
    logger.info(f"\n{'*' * 100}\nLLM Output : {answer}\n{'*' * 100}")

    # Return the answer
    return answer


rag_tool.run("What is Quantum Horizons?")
