import asyncio
from typing import ClassVar, Any, Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from backend.config.azure_models import AzureOpenAIModels
from backend.utils.constants import QUESTION_ANSWER_COT_PROMPT_TEMPLATE
from backend.config.logging_lib import logger


# Define the output schema for the answer
class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(
        description="Generates an answer to a query based on a given context."
    )


# Output schema for the relevance check
class Relevance(BaseModel):
    is_relevant: bool = Field(
        description="Whether the document is relevant to the query."
    )
    explanation: str = Field(
        description="An explanation of why the document is relevant or not."
    )


# Define the output schema for the grounding check
class IsGroundedOnFacts(BaseModel):
    """
    Output schema for checking if the answer is grounded in the provided context.
    """

    grounded_on_facts: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Define the output schema for the LLM's response
class QuestionAnswer(BaseModel):
    can_be_answered: bool = Field(
        description="binary result of whether the question can be fully answered or not"
    )
    explanation: str = Field(
        description="An explanation of why the question can be fully answered or not."
    )


"""--- LLM-based Function to Rewrite a Question for Better Vectorstore Retrieval ---"""


class RewriteQuestion(BaseModel):
    """
    Output schema for the rewritten question.
    """

    llm_model: ClassVar = AzureOpenAIModels().get_azure_model_4()

    @staticmethod
    async def rewrite_question(state: Dict[str, Any]) -> Dict[str, str]:
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
        rewrite_llm = RewriteQuestion.llm_model

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
        logger.info("Rewriting the question: %s", question)

        try:
            # chain.invoke is blocking — run in a thread
            result = await asyncio.to_thread(
                question_rewriter.invoke, {"question": question}
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

            logger.info("Finished rewrite_question: %s", new_question)
            return {"question": new_question}

        except Exception as e:
            logger.exception("Error in rewrite_question")
            raise RuntimeError("Failed to rewrite question using LLM") from e

    """--- LLM-based Function to Answer a Question from Context Using Chain-of-Thought Reasoning ---"""

    async def answer_question_from_context(
        self, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Description:
            Answers a question from a given context using chain-of-thought reasoning.

        Params:
            state (dict): A dictionary containing:
                - "question": The query question.
                - "context" or "aggregated_context": The context to answer the question from.

        Return:
            dict: A dictionary containing:
                - "answer": The answer to the question from the context.
                - "context": The context used.
                - "question": The original question.

        Exceptions:
            TypeError: If state is not a dict or missing keys.
            RuntimeError: If LLM invocation fails.
        """
        logger.info("Starting answer_question_from_context")
        if (
            not isinstance(state, dict)
            or "question" not in state
            or not ("context" in state or "aggregated_context" in state)
        ):
            raise TypeError(
                "state must be a dict containing 'question' and 'context' or 'aggregated_context'"
            )

        # Initialize the LLM for answering questions with chain-of-thought reasoning
        question_answer_from_context_llm = self.llm_model

        # Create the prompt object
        question_answer_from_context_cot_prompt = PromptTemplate(
            template=QUESTION_ANSWER_COT_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        # Combine the prompt and LLM into a chain with structured output
        question_answer_from_context_cot_chain = (
            question_answer_from_context_cot_prompt
            | question_answer_from_context_llm.with_structured_output(
                QuestionAnswerFromContext
            )
        )

        # Use 'aggregated_context' if available, otherwise fall back to 'context'
        question = state["question"]
        context = state.get("aggregated_context", state.get("context", ""))

        input_data = {"question": question, "context": context}

        logger.info("Invoking LLM to answer the question from context")
        try:
            output = await asyncio.to_thread(
                question_answer_from_context_cot_chain.invoke, input_data
            )

            # Normalize output
            if isinstance(output, dict):
                answer = output.get("answer_based_on_content") or output.get("answer")
            else:
                answer = getattr(output, "answer_based_on_content", None) or getattr(
                    output, "answer", None
                )

            if answer is None:
                raise RuntimeError("LLM did not return an answer")

            logger.info("Finished answer_question_from_context")
            print(f"answer before checking hallucination: {answer}")
            return {"answer": answer, "context": context, "question": question}

        except Exception as e:
            logger.exception("Error in answer_question_from_context")
            raise RuntimeError(
                "Failed to answer question from context using LLM"
            ) from e

    async def is_relevant_content(self, state: Dict[str, Any]) -> str:
        """
        Description:
            Determines if the retrieved context is relevant to the query.

        Params:
            state (dict): A dictionary containing:
                - "question": The query question.
                - "context": The retrieved context to check for relevance.

        Return:
            str: "relevant" if the context is relevant, "not relevant" otherwise.

        Exceptions:
            TypeError: If state is not a dict or missing keys.
            RuntimeError: If LLM relevance check fails.
        """
        logger.info("Starting is_relevant_content")
        if (
            not isinstance(state, dict)
            or "question" not in state
            or "context" not in state
        ):
            raise TypeError("state must be a dict containing 'question' and 'context'")

        # Prompt template for checking if the retrieved context is relevant to the query
        is_relevant_content_prompt_template = """
        You receive a query: {query} and a context: {context} retrieved from a vector store. 
        You need to determine if the document is relevant to the query. 
        {format_instructions}
        """

        # JSON parser for the output schema
        is_relevant_json_parser = JsonOutputParser(pydantic_object=Relevance)

        # Initialize the LLM for relevance checking
        is_relevant_llm = self.llm_model

        # Create the prompt object for the LLM
        is_relevant_content_prompt = PromptTemplate(
            template=is_relevant_content_prompt_template,
            input_variables=["query", "context"],
            partial_variables={
                "format_instructions": is_relevant_json_parser.get_format_instructions()
            },
        )

        # Combine prompt, LLM, and parser into a chain
        is_relevant_content_chain = (
            is_relevant_content_prompt | is_relevant_llm | is_relevant_json_parser
        )

        question = state["question"]
        context = state["context"]

        input_data = {"query": question, "context": context}

        logger.info("Invoking LLM to check relevance")
        try:
            output = await asyncio.to_thread(
                is_relevant_content_chain.invoke, input_data
            )
            # Normalize output
            if isinstance(output, dict):
                is_rel = output.get("is_relevant")
            else:
                is_rel = getattr(output, "is_relevant", None)

            logger.info(f"Finished is_relevant_content: {is_rel}")
            if is_rel:
                print("The document is relevant.")
                return "relevant"
            else:
                print("The document is not relevant.")
                return "not relevant"

        except Exception as e:
            logger.exception("Error in is_relevant_content")
            raise RuntimeError("Failed to determine relevance using LLM") from e

    @staticmethod
    async def grade_generation_v_documents_and_question(state: Dict[str, Any]) -> str:
        """
        Description:
            Grades the generated answer to a question based on:
            - Whether the answer is grounded in the provided context (fact-checking)
            - Whether the question can be fully answered from the context

        Params:
            state (dict): A dictionary containing:
                - "context": The context used to answer the question
                - "question": The original question
                - "answer": The generated answer

        Return:
            str: One of "hallucination", "useful", or "not_useful"

        Exceptions:
            TypeError: If state is not a dict or missing keys.
            RuntimeError: If LLM checks fail.
        """
        logger.info("Starting grade_generation_v_documents_and_question")
        if (
            not isinstance(state, dict)
            or "context" not in state
            or "answer" not in state
            or "question" not in state
        ):
            raise TypeError(
                "state must be a dict containing 'context', 'answer', and 'question'"
            )

        # Initialize the LLM for fact-checking (using same model)
        is_grounded_on_facts_llm = RewriteQuestion.llm_model

        # Define the prompt template for fact-checking
        is_grounded_on_facts_prompt_template = """
        You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
        You don't mind if it doesn't make sense, as long as it is grounded in the context.
        Output a JSON containing the answer to the question, and apart from the JSON format don't output any 
        additional text.
        """

        # Create the prompt object
        is_grounded_on_facts_prompt = PromptTemplate(
            template=is_grounded_on_facts_prompt_template,
            input_variables=["context", "answer"],
        )

        # Create the LLM chain for fact-checking
        is_grounded_on_facts_chain = (
            is_grounded_on_facts_prompt
            | is_grounded_on_facts_llm.with_structured_output(IsGroundedOnFacts)
        )

        """--- LLM Chain to Determine if a Question Can Be Fully Answered from Context ---"""

        # Define the prompt template for the LLM
        can_be_answered_prompt_template = """
        You receive a query: {question} and a context: {context}. 
        You need to determine if the question can be fully answered based on the context.
        {format_instructions}
        """

        # Create a JSON parser for the output schema
        can_be_answered_json_parser = JsonOutputParser(pydantic_object=QuestionAnswer)

        # Create the prompt object for the LLM
        answer_question_prompt = PromptTemplate(
            template=can_be_answered_prompt_template,
            input_variables=["question", "context"],
            partial_variables={
                "format_instructions": can_be_answered_json_parser.get_format_instructions()
            },
        )

        # Initialize the LLM for this task
        can_be_answered_llm = RewriteQuestion.llm_model

        # Compose the chain: prompt -> LLM -> output parser
        can_be_answered_chain = (
            answer_question_prompt | can_be_answered_llm | can_be_answered_json_parser
        )

        # Extract relevant fields from state
        context = state["context"]
        answer = state["answer"]
        question = state["question"]

        try:
            # 1. Check if the answer is grounded in the provided context (fact-checking)
            logger.info("Invoking LLM to check grounding in facts")
            result = await asyncio.to_thread(
                is_grounded_on_facts_chain.invoke,
                {"context": context, "answer": answer},
            )

            if isinstance(result, dict):
                grounded_on_facts = result.get("grounded_on_facts")
            else:
                grounded_on_facts = getattr(result, "grounded_on_facts", None)

            logger.info(f"Grounded on facts: {grounded_on_facts}")
            print("Checking if the answer is grounded in the facts...")

            if not grounded_on_facts:
                # If not grounded, label as hallucination
                print("The answer is hallucination.")
                return "hallucination"
            else:
                print("The answer is grounded in the facts.")

                # 2. Check if the question can be fully answered from the context
                input_data = {"question": question, "context": context}
                logger.info(
                    "Invoking LLM to determine if question can be fully answered"
                )
                output = await asyncio.to_thread(
                    can_be_answered_chain.invoke, input_data
                )

                if isinstance(output, dict):
                    can_be_answered = output.get("can_be_answered")
                else:
                    can_be_answered = getattr(output, "can_be_answered", None)

                if can_be_answered:
                    print("The question can be fully answered.")
                    return "useful"
                else:
                    print("The question cannot be fully answered.")
                    return "not_useful"

        except Exception as e:
            logger.exception("Error in grade_generation_v_documents_and_question")
            raise RuntimeError(
                "Failed to grade generation vs documents and question"
            ) from e
