from pprint import pprint
from typing import Any, Optional, Dict, List
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from backend.config.azure_models import AzureOpenAIModels
from backend.rag_optimization.research import RewriteQuestion
from backend.config.logging_lib import logger
import asyncio


# Output schema for the filtered relevant content
class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(
        description="The relevant content from the retrieved documents that is relevant to the query."
    )


class RetrieveData(RewriteQuestion):
    """--- LLM-based Function to Filter Only Relevant Retrieved Content ---"""

    # Output schema for the filtered relevant content
    relevant_content: Optional[str] = Field(
        default=None,
        description="The relevant content from the retrieved documents that is relevant to the query.",
    )
    rewritten_question: Optional[str] = Field(
        default=None, description="The rewritten version of the original user question."
    )

    explanation: Optional[str] = Field(
        default=None,
        description="A brief explanation of why the retrieved content is relevant to the rewritten question.",
    )

    chunks_query_retriever: Optional[Any] = None
    chapter_summaries_query_retriever: Optional[Any] = None
    book_quotes_query_retriever: Optional[Any] = None

    def __init__(
        self,
        chunks_vector_store: FAISS,
        chapter_summaries_vector_store: FAISS,
        book_quotes_vectorstore: FAISS,
        **data: Any,
    ):
        """--- Create Query Retrievers from Vector Stores ---"""
        super().__init__(**data)

        # Retriever for book chunks (returns the top 1 most relevant chunk)
        self.chunks_query_retriever = (
            chunks_vector_store.as_retriever(search_kwargs={"k": 1})
            if isinstance(chunks_vector_store, FAISS)
            else None
        )

        # Retriever for chapter summaries (returns the top 1 most relevant summary)
        self.chapter_summaries_query_retriever = (
            chapter_summaries_vector_store.as_retriever(search_kwargs={"k": 1})
            if isinstance(chapter_summaries_vector_store, FAISS)
            else None
        )

        # Retriever for book quotes (returns the top 10 most relevant quotes)
        self.book_quotes_query_retriever = (
            book_quotes_vectorstore.as_retriever(search_kwargs={"k": 10})
            if isinstance(book_quotes_vectorstore, FAISS)
            else None
        )

    @staticmethod
    async def escape_quotes(text: str) -> str:
        """
        Description:
            Escapes both single and double quotes in a string.

        Params:
            text (str): The string to escape.

        Return:
            str: The string with single and double quotes escaped.

        Exceptions:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # lightweight operation — run inline
        return text.replace('"', '\\"').replace("'", "\\'")

    async def retrieve_context_per_question(
        self, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Description:
            Retrieves relevant context for a given question by aggregating content from:
            - Book chunks
            - Chapter summaries
            - Book quotes

        Params:
            state (dict): A dictionary containing the question to answer, with key "question".

        Return:
            dict: A dictionary with keys:
                - "context": Aggregated context string from all sources.
                - "question": The original question.

        Exceptions:
            TypeError: If state is not a dict or missing 'question'.
            RuntimeError: If retriever calls fail.
        """
        logger.info("Starting retrieve_context_per_question")
        if not isinstance(state, dict) or "question" not in state:
            raise TypeError("state must be a dict containing a 'question' key")
        question = state["question"]

        context, context_summaries, book_quotes = "", "", ""

        try:
            # Retrieve relevant book chunks
            if self.chunks_query_retriever is not None:
                logger.info("Retrieving relevant chunks...")
                # Some retrievers expose .invoke, some expose .get_relevant_documents. Use whatever is present.
                if hasattr(self.chunks_query_retriever, "invoke"):
                    docs = await asyncio.to_thread(
                        self.chunks_query_retriever.invoke, question
                    )
                else:
                    docs = await asyncio.to_thread(
                        self.chunks_query_retriever.get_relevant_documents, question
                    )
                # join page_content safely
                context = " ".join(getattr(doc, "page_content", "") for doc in docs)

            # Retrieve relevant chapter summaries
            if self.chapter_summaries_query_retriever is not None:
                logger.info("Retrieving relevant chapter summaries...")
                docs_summaries = await asyncio.to_thread(
                    self.chapter_summaries_query_retriever.get_relevant_documents,
                    question,
                )
                context_summaries = " ".join(
                    f"{getattr(doc, 'page_content', '')} (Chapter {doc.metadata.get('chapter')})"
                    for doc in docs_summaries
                )

            # Retrieve relevant book quotes
            if self.book_quotes_query_retriever is not None:
                logger.info("Retrieving relevant book quotes...")
                docs_book_quotes = await asyncio.to_thread(
                    self.book_quotes_query_retriever.get_relevant_documents, question
                )
                book_quotes = " ".join(
                    getattr(doc, "page_content", "") for doc in docs_book_quotes
                )

            # Aggregate all contexts and escape problematic characters
            all_contexts = context + " " + context_summaries + " " + book_quotes
            all_contexts = await self.escape_quotes(all_contexts)
            logger.info("Finished retrieve_context_per_question")
            return {"context": all_contexts, "question": question}

        except Exception as e:
            logger.exception("Error while retrieving context for question")
            raise RuntimeError("Failed to retrieve context from vector stores") from e

    async def run_retriever_pipeline(self) -> Any:
        """
        Description:
            Run the full retriever -> filter -> relevance check -> answer -> grade pipeline.

        Params:
            None

        Return:
            Any: The final graded answer/result from the pipeline.

        Exceptions:
            RuntimeError: If any step of the pipeline fails.
        """
        logger.info("Starting run_retriever_pipeline")
        try:
            # 1. Define the initial state with the question to answer
            init_state = {"question": "who is fluffy?"}

            # 2. Retrieve relevant context for the question from the vector stores (chunks, summaries, quotes)
            context_state = await self.retrieve_context_per_question(init_state)

            # 3. Use an LLM to filter and keep only the content relevant to the question from the retrieved context
            relevant_content_state = await self.keep_only_relevant_content(
                context_state
            )

            # 4. Check if the filtered content is relevant to the question using an LLM-based relevance check
            # These methods are inherited from RewriteQuestion; they may be sync or async.
            # We attempt to call them appropriately (prefer async if available).
            if asyncio.iscoroutinefunction(self.is_relevant_content):
                is_relevant_content_state = await self.is_relevant_content(
                    relevant_content_state
                )
            else:
                is_relevant_content_state = await asyncio.to_thread(
                    self.is_relevant_content, relevant_content_state
                )

            # 5. Use an LLM to answer the question based on the relevant context
            if asyncio.iscoroutinefunction(self.answer_question_from_context):
                answer_state = await self.answer_question_from_context(
                    relevant_content_state
                )
            else:
                answer_state = await asyncio.to_thread(
                    self.answer_question_from_context, relevant_content_state
                )

            # 6. Grade the generated answer:
            #    - Check if the answer is grounded in the provided context (fact-checking)
            #    - Check if the question can be fully answered from the context
            if asyncio.iscoroutinefunction(
                self.grade_generation_v_documents_and_question
            ):
                final_answer = await self.grade_generation_v_documents_and_question(
                    answer_state
                )
            else:
                final_answer = await asyncio.to_thread(
                    self.grade_generation_v_documents_and_question, answer_state
                )

            # 7. Print the final answer (preserve original behavior)
            print(
                answer_state.get("answer")
                if isinstance(answer_state, dict)
                else answer_state
            )
            logger.info("Finished run_retriever_pipeline")
            return final_answer

        except Exception as e:
            logger.exception("Error in run_retriever_pipeline")
            raise RuntimeError("Retriever pipeline failed") from e

    async def keep_only_relevant_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Description:
            Filters and keeps only the relevant content from the retrieved documents that is relevant to the query.

        Params:
            state (dict): A dictionary containing:
                - "question": The query question.
                - "context": The retrieved documents as a string.

        Return:
            dict: A dictionary containing:
                - "relevant_context": The filtered relevant content.
                - "context": The original context.
                - "question": The original question.

        Exceptions:
            TypeError: If state is not a dict or missing keys.
            RuntimeError: If the LLM filtering fails.
        """
        logger.info("Starting keep_only_relevant_content")
        if (
            not isinstance(state, dict)
            or "question" not in state
            or "context" not in state
        ):
            raise TypeError("state must be a dict containing 'question' and 'context'")

        # Prompt template for filtering relevant content from retrieved documents
        keep_only_relevant_content_prompt_template = """
             You receive a query: {query} and retrieved documents: {retrieved_documents} from a vector store.
             You need to filter out all the non relevant information that doesn't supply important information 
             regarding the {query}.
             Your goal is just to filter out the non relevant information.
             You can remove parts of sentences that are not relevant to the query or remove whole sentences that are 
             not relevant to the query.
             DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
             Output the filtered relevant content.
             """

        # Create the prompt for the LLM
        keep_only_relevant_content_prompt = PromptTemplate(
            template=keep_only_relevant_content_prompt_template,
            input_variables=["query", "retrieved_documents"],
        )

        # Initialize the LLM for filtering relevant content
        keep_only_relevant_content_llm = AzureOpenAIModels().get_azure_model_4()

        # Create the LLM chain for filtering relevant content
        keep_only_relevant_content_chain = (
            keep_only_relevant_content_prompt
            | keep_only_relevant_content_llm.with_structured_output(KeepRelevantContent)
        )
        question = state["question"]
        context = state["context"]

        input_data = {"query": question, "retrieved_documents": context}

        try:
            logger.info("Keeping only the relevant content...")
            pprint("--------------------")
            # chain.invoke may be blocking — run in a thread
            output = await asyncio.to_thread(
                keep_only_relevant_content_chain.invoke, input_data
            )

            # handle output with structured model
            relevant_content = getattr(output, "relevant_content", None)
            if relevant_content is None:
                # try dict-style output
                relevant_content = (
                    output.get("relevant_content") if isinstance(output, dict) else ""
                )

            # Some LLM structured outputs may be lists of strings — ensure it's a single string
            if isinstance(relevant_content, list):
                relevant_content = "".join(str(x) for x in relevant_content)
            relevant_content = str(relevant_content)
            relevant_content = await self.escape_quotes(relevant_content)

            logger.info("Finished keep_only_relevant_content")
            return {
                "relevant_context": relevant_content,
                "context": context,
                "question": question,
            }

        except Exception as e:
            logger.exception("Error while filtering relevant content using LLM")
            raise RuntimeError("LLM filtering failed") from e


if __name__ == "__main__":
    hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"
    handler = RetrieveData()
    summaries, book_quotes_list = handler.preprocess_pipeline()
    for index, value in enumerate(summaries):
        print((value.metadata, value.page_content))
