import asyncio
from typing import Any, Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from backend.config.azure_models import AzureOpenAIModels
from backend.config.logging_lib import logger

"""--- LLM-based Function to Rewrite a Question for Better Vectorstore Retrieval ---"""


class RewriteQuestion(BaseModel):
    """
    Output schema for the rewritten question.
    """
    rewritten_question: str = Field(
        description="The improved question optimized for vectorstore retrieval."
    )
    explanation: str = Field(
        description="The explanation of the rewritten question."
    )


class RewriteMultipleQuestion:

    def __init__(self):
        self.llm_model = AzureOpenAIModels().get_azure_model_4()

    async def rewrite_question(self, state: Dict[str, Any]) -> Dict[str, str]:
        """
        Description:
            Rewrites the given question using the LLM to optimize it for vectorstore retrieval.

        Params:
            state (dict): A dictionary containing the question to rewrite, with key "question".

        Return:
            dict: A dictionary with the rewritten question under the key "question".

        Exceptions:
            TypeError: If state is not a dict or missing 'question'.
            RuntimeError: If LLM call fails.
        """
        logger.info("Starting rewrite_question")
        if not isinstance(state, dict) or "question" not in state:
            raise TypeError("state must be a dict containing a 'question' key")

        # Create a JSON parser for the output schema
        rewrite_question_string_parser = JsonOutputParser(
            pydantic_object=RewriteQuestion
        )

        # Initialize the LLM for rewriting questions
        rewrite_llm = self.llm_model

        # Define the prompt template for question rewriting
        rewrite_prompt_template = """
        You are a question re-writer that converts an input question to a better version optimized for vectorstore 
        retrieval.
        Analyze the input question {question} and try to reason about the underlying semantic intent / meaning.
        {format_instructions}
        """

        # Create the prompt object
        rewrite_prompt = PromptTemplate(
            template=rewrite_prompt_template,
            input_variables=["question"],
            partial_variables={
                "format_instructions": rewrite_question_string_parser.get_format_instructions()
            },
        )

        # Combine prompt, LLM, and parser into a chain
        question_rewriter = (
                rewrite_prompt | rewrite_llm | rewrite_question_string_parser
        )

        question = state["question"]
        logger.info(f"Rewriting the question: {question}")

        try:
            result = await asyncio.to_thread(
                lambda: question_rewriter.invoke({"question": question})
            )

            # result may be dict-like or object — normalize
            if isinstance(result, dict):
                new_question = result.get("rewritten_question") or result.get(
                    "question"
                )
            else:
                # try attribute access
                new_question = getattr(result, "rewritten_question", None) or getattr(
                    result, "question", None
                )

            if not new_question:
                raise RuntimeError("LLM did not return a rewritten question")

            logger.info(f"Finished rewrite_question: {new_question}")

            state["new_question"] = new_question
            return state

        except Exception as e:
            logger.exception("Error in rewrite_question")
            raise RuntimeError("Failed to rewrite question using LLM") from e
