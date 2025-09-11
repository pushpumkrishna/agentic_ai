from typing import TypedDict, Any
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, StateGraph
from IPython.display import display, Image
from langgraph.graph.state import CompiledStateGraph

from backend.config.logging_lib import logger
from backend.rag_optimization.step_3_retrieve_data import RetrieveData
from backend.rag_optimization.step_4_rewrite_question import RewriteMultipleQuestion
from backend.rag_optimization.step_5_relevancy import RelevanceCheck
from backend.utils.measure_time import measure_time


# Define the state for the workflow graph
class QualitativeRetrievalAnswerGraphState(TypedDict):
    question: str
    context: str
    answer: str


class GraphRetrieval(RetrieveData, RewriteMultipleQuestion, RelevanceCheck):
    """
    # -----------------------------------------------
    # Qualitative Retrieval Answer Graph Construction
    # -----------------------------------------------

    Description:
        Graph pipeline that builds a StateGraph workflow for qualitative retrieval and
        answer generation. It wires together retrieval, filtering, rewriting, answering,
        and grading nodes into a reusable workflow and renders a Mermaid diagram.

    Params:
        chunks_vector_store (FAISS): FAISS vector store for book chunks.
        chapter_summaries_vector_store (FAISS): FAISS vector store for chapter summaries.
        book_quotes_vectorstore (FAISS): FAISS vector store for book quotes.
        **data: Additional data passed to parent class.

    Return:
        None

    Exceptions:
        TypeError: If inputs are not FAISS instances.
        RuntimeError: If graph compilation or rendering fails.
    """

    def __init__(
        self,
        chunks_vector_store: FAISS,
        chapter_summaries_vector_store: FAISS,
        book_quotes_vectorstore: FAISS,
        init_state,
        **data: Any,
    ):
        # Validate types and initialize parent
        if not isinstance(chunks_vector_store, FAISS):
            raise TypeError("chunks_vector_store must be a FAISS instance")
        if not isinstance(chapter_summaries_vector_store, FAISS):
            raise TypeError("chapter_summaries_vector_store must be a FAISS instance")
        if not isinstance(book_quotes_vectorstore, FAISS):
            raise TypeError("book_quotes_vectorstore must be a FAISS instance")

        super().__init__(
            chunks_vector_store,
            chapter_summaries_vector_store,
            book_quotes_vectorstore,
            init_state,
            **data,
        )
        # Retriever for book chunks (returns the top 1 most relevant chunk)
        self._chunks_query_retriever = (
            chunks_vector_store.as_retriever(search_kwargs={"k": 1})
            if isinstance(chunks_vector_store, FAISS)
            else None
        )

        # Retriever for chapter summaries (returns the top 1 most relevant summary)
        self._chapter_summaries_query_retriever = (
            chapter_summaries_vector_store.as_retriever(search_kwargs={"k": 1})
            if isinstance(chapter_summaries_vector_store, FAISS)
            else None
        )

        # Retriever for book quotes (returns the top 10 most relevant quotes)
        self._book_quotes_query_retriever = (
            book_quotes_vectorstore.as_retriever(search_kwargs={"k": 10})
            if isinstance(book_quotes_vectorstore, FAISS)
            else None
        )

        self.init_state = init_state
        logger.info("Initialized GraphRetrieval with provided FAISS vectorstores")

    @measure_time
    async def graph_pipeline(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """
        Description:
            Build and compile a StateGraph workflow connecting retrieval, filtering,
            rewriting, answering, and grading nodes. Render and display the resulting
            workflow as a Mermaid diagram.

        Params:
            None

        Return:
            None

        Exceptions:
            RuntimeError: If graph compilation or rendering fails.
        """
        logger.info("Starting graph_pipeline")

        try:
            # retrieve_context_input_question = RetrieveData(self._chunks_vector_store,
            # self._chapter_summaries_vector_store,
            # self._book_quotes_vectorstore,
            # self.init_state)
            # -------------------------
            # Create the workflow graph object
            # -------------------------
            qualitative_retrieval_answer_workflow = StateGraph(
                QualitativeRetrievalAnswerGraphState
            )

            # Node: Retrieve context for the question from vector stores
            qualitative_retrieval_answer_workflow.add_node(
                "retrieve_context_per_question", self.retrieve_context_input_question
            )

            # Node: Use LLM to keep only relevant content from the retrieved context
            qualitative_retrieval_answer_workflow.add_node(
                "keep_only_relevant_content", self.keep_only_relevant_content
            )

            # Node: Rewrite the question for better retrieval if needed
            qualitative_retrieval_answer_workflow.add_node(
                "rewrite_question", self.rewrite_question
            )

            # Node: Answer the question from the relevant context using LLM
            qualitative_retrieval_answer_workflow.add_node(
                "answer_question_from_context", self.answer_question_from_context
            )

            # -------------------------
            # Build the Workflow Edges
            # -------------------------

            # Set the entry point of the workflow
            qualitative_retrieval_answer_workflow.set_entry_point(
                "retrieve_context_per_question"
            )

            # Edge: After retrieving context, filter to keep only relevant content
            qualitative_retrieval_answer_workflow.add_edge(
                "retrieve_context_per_question", "keep_only_relevant_content"
            )

            # Conditional Edge: After filtering, check if content is relevant
            # If relevant, answer the question; if not, rewrite the question
            qualitative_retrieval_answer_workflow.add_conditional_edges(
                "keep_only_relevant_content",
                self.is_relevant_content,
                {
                    "relevant": "answer_question_from_context",
                    "not relevant": "rewrite_question",
                },
            )

            # Edge: After rewriting the question, retrieve context again
            qualitative_retrieval_answer_workflow.add_edge(
                "rewrite_question", "retrieve_context_per_question"
            )

            # Conditional Edge: After answering, grade the answer
            # If hallucination, try answering again; if not useful, rewrite question; if useful, end
            qualitative_retrieval_answer_workflow.add_conditional_edges(
                "answer_question_from_context",
                self.grade_generation_v_documents_and_question,
                {
                    "hallucination": "answer_question_from_context",
                    "not_useful": "rewrite_question",
                    "useful": END,
                },
            )

            logger.info("Graph nodes and edges added successfully")

            # Compile the workflow graph into an executable app (potentially blocking)
            logger.info("Compiling the StateGraph workflow")
            qualitative_retrieval_answer_retrival_app = await asyncio.to_thread(
                qualitative_retrieval_answer_workflow.compile
            )
            logger.info("Compiled the StateGraph workflow successfully")

            # Draw the Mermaid PNG using the app's graph; run in a thread to avoid blocking
            logger.info("Rendering Mermaid diagram to PNG")
            # draw_mermaid_png is expected to accept draw_method=MermaidDrawMethod.API
            draw_png_callable = (
                lambda: qualitative_retrieval_answer_retrival_app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API
                )
            )
            png_bytes = await asyncio.to_thread(draw_png_callable)

            # Validate png_bytes and display
            if png_bytes is None:
                logger.error("Graph drawing returned no bytes")
                raise RuntimeError("Failed to draw Mermaid PNG (no bytes returned)")

            # Display the workflow graph as a Mermaid diagram in Jupyter / IPython
            logger.info("Displaying Mermaid PNG in environment")
            display(Image(png_bytes))

            logger.info("Finished graph_pipeline")

        except Exception as e:
            logger.exception("Error in graph_pipeline")
            raise RuntimeError("Failed to build or render graph pipeline") from e

        return qualitative_retrieval_answer_retrival_app
