# --- Standard Library Imports ---
from dotenv import load_dotenv
import os
import asyncio
from backend.config.logging_lib import logger
from backend.rag_optimization.build_graph import GraphRetrieval
from backend.rag_optimization.encoding import EncodeEmbeddings
from backend.rag_optimization.final_graph import execute_plan_and_print_steps
from backend.rag_optimization.retrieve_data import RetrieveData
from backend.rag_optimization.second_retreival import SecondRetrieval
from backend.rag_optimization.step_1_preprocessing import ProcessDocument
from backend.utils.measure_time import measure_time

# --- Load environment variables (e.g., API keys) ---
load_dotenv(override=True)

# --- Set environment variable for debugging (optional) ---
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100000"

# Retrieve the Groq API key from environment variable (for use by Groq LLMs)
groq_api_key = os.getenv("GROQ_API_KEY")


@measure_time
async def main():
    hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"
    handler = ProcessDocument(hp_pdf_path)

    # ✅ Await async pipeline
    chapter_summaries, book_quotes_list = await handler.preprocess_pipeline()

    # ✅ Both values available
    for index, value in enumerate(chapter_summaries):
        print(value.metadata, value.page_content)

    print("book_quotes_list:: ", book_quotes_list)

    # ✅ Do not overwrite book_quotes_list
    encoding_handler = EncodeEmbeddings()
    (
        chunks_vector_store,
        chapter_summaries_vector_store,
        book_quotes_vectorstore,
    ) = await encoding_handler.create_vector_db(
        chapter_summaries, book_quotes_list, hp_pdf_path
    )
    logger.info("Vector stores created successfully")

    retriever_handler = RetrieveData(
        chunks_vector_store=chunks_vector_store,
        chapter_summaries_vector_store=chapter_summaries_vector_store,
        book_quotes_vectorstore=book_quotes_vectorstore,
    )

    result = await retriever_handler.run_retriever_pipeline()
    logger.info(f"Retriever completed... ")

    graph_handler = GraphRetrieval(
        chunks_vector_store, chapter_summaries_vector_store, book_quotes_vectorstore
    )
    await graph_handler.graph_pipeline()

    second_handler = SecondRetrieval(
        chunks_vector_store, chapter_summaries_vector_store, book_quotes_vectorstore
    )

    await second_handler.test_answer_workflow_graph()

    logger.info("test_answer_workflow_graph done")

    """An example we want the model to fail"""
    # -----------------------------------------------------------
    # Example: Run the Plan-and-Execute Agent for a Sample Question
    # -----------------------------------------------------------

    # Define the input question for the agent
    input = {"question": "what did professor lupin teach?"}

    # Execute the plan-and-execute workflow and print each step
    final_answer, final_state = await execute_plan_and_print_steps(chunks_vector_store,
                                                                   chapter_summaries_vector_store,
                                                                   book_quotes_vectorstore,
                                                                   input)
    print((final_answer, final_state))

    """An example we want the model to succeed"""
    # -----------------------------------------------------------
    # Example: Run the Plan-and-Execute Agent for a Complex Question
    # -----------------------------------------------------------

    # Define the input question for the agent.
    # This question requires reasoning about the professor who helped the villain and what class they teach.
    input = {
        "question": "what is the class that the professor who helped the villain is teaching?"
    }

    # Execute the plan-and-execute workflow and print each step.
    # The function will print the reasoning process and the final answer.
    final_answer, final_state = await execute_plan_and_print_steps(chunks_vector_store,
                                                                   chapter_summaries_vector_store,
                                                                   book_quotes_vectorstore,
                                                                   input)
    print((final_answer, final_state))


if __name__ == "__main__":
    asyncio.run(main())
