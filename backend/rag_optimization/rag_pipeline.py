# --- Standard Library Imports ---
import os
from dotenv import load_dotenv
# import os
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config.logging_lib import logger
from backend.rag_optimization.step_6_build_graph import GraphRetrieval
from backend.rag_optimization.step_2_encoding import EncodeEmbeddings
from backend.rag_optimization.final_graph import execute_plan_and_print_steps
from backend.rag_optimization.step_3_retrieve_data import RetrieveData
from backend.rag_optimization.step_7_second_retrieval import SecondRetrieval
from backend.rag_optimization.step_1_preprocessing import ProcessDocument
from backend.rag_optimization.step_4_rewrite_question import RewriteMultipleQuestion
from backend.rag_optimization.step_5_relevancy import RelevanceCheck
from backend.utils.measure_time import measure_time

# --- Load environment variables (e.g., API keys) ---
load_dotenv(override=True)


# # --- Set environment variable for debugging (optional) ---
# os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100000"

# Retrieve the Groq API key from environment variable (for use by Groq LLMs)
# groq_api_key = os.getenv("GROQ_API_KEY")


@measure_time
async def main(input_question):
    if (
            os.path.exists("../rag_optimization/chunks_vector_store")
            and os.path.exists("../rag_optimization/chapter_summaries_vector_store")
            and os.path.exists("../rag_optimization/book_quotes_vectorstore")
    ):
        logger.info("Found existing vector stores; loading from disk")
        embeddings = HuggingFaceEmbeddings(
            model_name="C:/Users/703395858/PycharmProjects/agentic_ai/backend/models/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        def load_faiss(folder: str, emb):
            return FAISS.load_local(
                folder, emb, allow_dangerous_deserialization=True
            )

        chunks_vector_store = await asyncio.to_thread(
            load_faiss, "../rag_optimization/chunks_vector_store", embeddings
        )
        chapter_summaries_vector_store = await asyncio.to_thread(
            load_faiss, "../rag_optimization/chapter_summaries_vector_store", embeddings
        )
        book_quotes_vectorstore = await asyncio.to_thread(
            load_faiss, "../rag_optimization/book_quotes_vectorstore", embeddings
        )

        logger.info("Loaded vector stores from disk successfully")

    else:
        hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"
        handler = ProcessDocument(hp_pdf_path)

        # ✅ Await async pipeline
        chapter_summaries, book_quotes_list = await handler.preprocess_pipeline()
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

    init_state = {"question": input_question}

    retriever_handler = RetrieveData(
        chunks_vector_store=chunks_vector_store,
        chapter_summaries_vector_store=chapter_summaries_vector_store,
        book_quotes_vectorstore=book_quotes_vectorstore,
        init_state=init_state
    )

    relevant_content_state = await retriever_handler.run_retriever_pipeline()
    logger.info(f"Retriever completed... ")
    print(f"relevant_content_state: {relevant_content_state.keys()}")

    rewrite_handler = RewriteMultipleQuestion()
    state = await rewrite_handler.rewrite_question(relevant_content_state)
    print(f"relevant_content_state1: {state.keys()}")

    relevancy_handler = RelevanceCheck()
    state = await relevancy_handler.run_relevancy_pipeline(state=state)
    print(f"relevant_content_state2: {state.keys()}")

    graph_handler = GraphRetrieval(
        chunks_vector_store, chapter_summaries_vector_store, book_quotes_vectorstore, state
    )
    await graph_handler.graph_pipeline()

    second_handler = SecondRetrieval(
        chunks_vector_store, chapter_summaries_vector_store, book_quotes_vectorstore,
        state)

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
                                                                   input,
                                                                   state)
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
                                                                   input,
                                                                   state)
    print((final_answer, final_state))


if __name__ == "__main__":
    # input_question_ = input("Please ask the question")
    input_question_ = "who is fluffy"
    asyncio.run(main(input_question=input_question_))
