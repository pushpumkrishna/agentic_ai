import datetime
from typing import Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from backend.config.azure_models import AzureOpenAIModels


class TutorialSearch(BaseModel):
    """A data model for searching over a database of tutorial videos."""

    # The main query for a similarity search over the video's transcript.
    content_search: str = Field(None, description="Similarity search query applied to video transcripts.")

    # A more succinct query for searching just the video's title.
    title_search: str = Field(None,
                              description="Alternate version of the content search query to apply to video titles.")

    # Optional metadata filters
    min_view_count: Optional[int] = Field(None, description="Minimum view count filter, inclusive.")
    max_view_count: Optional[int] = Field(None, description="Maximum view count filter, exclusive.")
    earliest_publish_date: Optional[datetime.date] = Field(None, description="Earliest publish date filter, inclusive.")
    latest_publish_date: Optional[datetime.date] = Field(None, description="Latest publish date filter, exclusive.")
    min_length_sec: Optional[int] = Field(None, description="Minimum video length in seconds, inclusive.")
    max_length_sec: Optional[int] = Field(None, description="Maximum video length in seconds, exclusive.")

    def pretty_print(self) -> None:
        """A helper function to print the populated fields of the model."""
        # noinspection PyTypeChecker
        for field in self.__class__.model_fields:
            if getattr(self, field) is not None:
                print(f"{field}: {getattr(self, field)}")

    def run(self):

        # System prompt for the query analyzer
        system = """You are an expert at converting user questions into database queries. \
        You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
        Given a question, return a database query optimized to retrieve the most relevant results.

        If there are acronyms or words you are not familiar with, do not try to rephrase them."""

        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
        structured_llm = AzureOpenAIModels().get_azure_model_4().with_structured_output(TutorialSearch)

        # The final query analyzer chain
        query_analyzer = prompt | structured_llm

        # Test 1: A simple query
        query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()

        # Test 2: A query with a date filter
        query_analyzer.invoke(
            {"question": "videos on chat langchain published in 2023"}
        ).pretty_print()

        # Test 3: A query with a length filter
        query_analyzer.invoke(
            {
                "question": "how to use multi-modal models in an agent, only videos under 5 minutes"
            }
        ).pretty_print()


if __name__ == "__main__":
    TutorialSearch().run()



