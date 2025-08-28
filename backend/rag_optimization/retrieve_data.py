from pprint import pprint
from typing import Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from backend.config.azure_models import AzureOpenAIModels
from backend.rag_optimization.research import RewriteQuestion


# Output schema for the filtered relevant content
class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(
        description="The relevant content from the retrieved documents that is relevant to the query."
    )


class RetrieveData(RewriteQuestion):
    """ --- LLM-based Function to Filter Only Relevant Retrieved Content ---"""

    # Output schema for the filtered relevant content
    relevant_content: Optional[str] = Field(
        default=None,
        description="The relevant content from the retrieved documents that is relevant to the query."
    )
    rewritten_question: Optional[str] = Field(
        default=None,
        description="The rewritten version of the original user question."
    )

    explanation: Optional[str] = Field(
        default=None,
        description="A brief explanation of why the retrieved content is relevant to the rewritten question."
    )

    chunks_query_retriever: Optional[Any] = None
    chapter_summaries_query_retriever: Optional[Any] = None
    book_quotes_query_retriever: Optional[Any] = None

    def __init__(
            self,
            chunks_vector_store: FAISS,
            chapter_summaries_vector_store: FAISS,
            book_quotes_vectorstore: FAISS,
            **data: Any
    ):
        """--- Create Query Retrievers from Vector Stores ---"""

        # The following retrievers are used to fetch relevant documents from the vector stores
        # based on a query. The number of results returned can be controlled via the 'k' parameter.

        # Retriever for book chunks (returns the top 1 most relevant chunk)
        super().__init__(**data)
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
    def escape_quotes(text):
        """
        Escapes both single and double quotes in a string.

        Args:
            text (str): The string to escape.

        Returns:
            str: The string with single and double quotes escaped.
        """
        return text.replace('"', '\\"').replace("'", "\\'")

    def retrieve_context_per_question(self, state):
        """
        Retrieves relevant context for a given question by aggregating content from:
        - Book chunks
        - Chapter summaries
        - Book quotes

        Args:
            state (dict): A dictionary containing the question to answer, with key "question".

        Returns:
            dict: A dictionary with keys:
                - "context": Aggregated context string from all sources.
                - "question": The original question.
        """

        question = state["question"]
        context, context_summaries, book_quotes = "", "", ""

        # Retrieve relevant book chunks
        if self.chunks_query_retriever is not None:
            print("Retrieving relevant chunks...")
            docs = self.chunks_query_retriever.invoke(question)
            context = " ".join(doc.page_content for doc in docs)

        # Retrieve relevant chapter summaries
        if self.chapter_summaries_query_retriever is not None:
            print("Retrieving relevant chapter summaries...")
            docs_summaries = self.chapter_summaries_query_retriever.get_relevant_documents(question)
            context_summaries = " ".join(
                f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
            )

        # Retrieve relevant book quotes
        if self.book_quotes_query_retriever is not None:
            print("Retrieving relevant book quotes...")
            docs_book_quotes = self.book_quotes_query_retriever.get_relevant_documents(question)
            book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)

        # Aggregate all contexts and escape problematic characters
        all_contexts = context + context_summaries + book_quotes
        all_contexts = self.escape_quotes(all_contexts)

        return {"context": all_contexts, "question": question}

    def run_retriever_pipeline(self):

        # 1. Define the initial state with the question to answer
        init_state = {"question": "who is fluffy?"}

        # 2. Retrieve relevant context for the question from the vector stores (chunks, summaries, quotes)
        context_state = self.retrieve_context_per_question(init_state)

        # 3. Use an LLM to filter and keep only the content relevant to the question from the retrieved context
        relevant_content_state = self.keep_only_relevant_content(context_state)

        # 4. Check if the filtered content is relevant to the question using an LLM-based relevance check
        is_relevant_content_state = self.is_relevant_content(relevant_content_state)

        # 5. Use an LLM to answer the question based on the relevant context
        answer_state = self.answer_question_from_context(relevant_content_state)

        # 6. Grade the generated answer:
        #    - Check if the answer is grounded in the provided context (fact-checking)
        #    - Check if the question can be fully answered from the context
        final_answer = self.grade_generation_v_documents_and_question(answer_state)

        # 7. Print the final answer
        print(answer_state["answer"])

        return final_answer

    def keep_only_relevant_content(self, state):
        """
        Filters and keeps only the relevant content from the retrieved documents that is relevant to the query.

        Args:
            state (dict): A dictionary containing:
                - "question": The query question.
                - "context": The retrieved documents as a string.

        Returns:
            dict: A dictionary containing:
                - "relevant_context": The filtered relevant content.
                - "context": The original context.
                - "question": The original question.
        """

        # Prompt template for filtering relevant content from retrieved documents
        keep_only_relevant_content_prompt_template = """
             You receive a query: {query} and retrieved documents: {retrieved_documents} from a vector store.
             You need to filter out all the non relevant information that doesn't supply important information regarding the {query}.
             Your goal is just to filter out the non relevant information.
             You can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
             DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
             Output the filtered relevant content.
             """

        # Create the prompt for the LLM
        keep_only_relevant_content_prompt = PromptTemplate(
            template=keep_only_relevant_content_prompt_template,
            input_variables=["query", "retrieved_documents"],
        )

        # # Initialize the LLM for filtering relevant content
        keep_only_relevant_content_llm = AzureOpenAIModels().get_azure_model_4()

        # Create the LLM chain for filtering relevant content
        keep_only_relevant_content_chain = (
                keep_only_relevant_content_prompt
                | keep_only_relevant_content_llm.with_structured_output(KeepRelevantContent)
        )
        question = state["question"]
        context = state["context"]

        input_data = {
            "query": question,
            "retrieved_documents": context
        }

        print("Keeping only the relevant content...")
        pprint("--------------------")
        output = keep_only_relevant_content_chain.invoke(input_data)
        relevant_content = output.relevant_content
        relevant_content = "".join(relevant_content)
        relevant_content = self.escape_quotes(relevant_content)

        return {
            "relevant_context": relevant_content,
            "context": context,
            "question": question
        }


if __name__ == "__main__":

    hp_pdf_path = "Harry_Potter_Book_1_The_Sorcerers_Stone.pdf"
    handler = RetrieveData()
    summaries, book_quotes_list = handler.preprocess_pipeline()
    for index, value in enumerate(summaries):
        print((value.metadata, value.page_content))
