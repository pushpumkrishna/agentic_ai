import asyncio
from typing import Dict, Any, ClassVar, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from backend.config.azure_models import AzureOpenAIModels
from backend.config.logging_lib import logger
from backend.utils.constants import QUESTION_ANSWER_COT_PROMPT_TEMPLATE


# Define the output schema for the answer
class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(
        description="Generates an answer to a query based on a given context."
    )


# Define the output schema for the LLM response
class QuestionAnswer(BaseModel):
    can_be_answered: bool = Field(
        description="binary result of whether the question can be fully answered or not"
    )
    explanation: str = Field(
        description="An explanation of why the question can be fully answered or not."
    )


# Output schema for the relevance check
class Relevance(BaseModel):
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
    is_relevant: bool = Field(
        description="Whether the document is relevant to the query."
    )


# Define the output schema for the grounding check
class IsGroundedOnFacts(BaseModel):
    """
    Output schema for checking if the answer is grounded in the provided context.
    """
    grounded_on_facts: bool = Field(
        description="Answer is grounded in the facts, 'True' or 'False'"
    )


class RelevanceCheck:
    llm_model: ClassVar = AzureOpenAIModels().get_azure_model_4()

    """--- LLM-based Function to Answer a Question from Context Using Chain-of-Thought Reasoning ---"""

    async def answer_question_from_context(
            self,
            state: Dict[str, Any]
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
                - "new_answer": The answer to the question from the context.
                - "context": The context used.
                - "answer": Original Answer
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
                | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext)
        )

        # Use 'aggregated_context' if available, otherwise fall back to 'context'
        question = state["question"]
        context = state.get("aggregated_context", state.get("context", ""))

        input_data = {"question": question, "context": context}

        logger.info("Invoking LLM to answer the question from context")
        try:
            output = await asyncio.to_thread(
                lambda: question_answer_from_context_cot_chain.invoke(input_data)
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
            state["answer"] = answer
            return state

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
                lambda: is_relevant_content_chain.invoke(input_data)
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

    async def grade_generation_v_documents_and_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
            str: One of "hallucinations", "useful", or "not_useful"

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

        # Initialize the LLM for fact-checking (using the same model)
        is_grounded_on_facts_llm = self.llm_model

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
        can_be_answered_llm = self.llm_model

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
                lambda: is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
            )

            if isinstance(result, dict):
                grounded_on_facts = result.get("grounded_on_facts")
            else:
                grounded_on_facts = getattr(result, "grounded_on_facts", None)

            logger.info(f"Grounded on facts: {grounded_on_facts}")
            print("Checking if the answer is grounded in the facts...")

            if not grounded_on_facts:
                # If not grounded, label as hallucination
                state["hallucination"] = True
                print("The answer is hallucination.")
            else:
                state["hallucination"] = False
                print("The answer is grounded in the facts.")

            # 2. Check if the question can be fully answered from the context
            input_data = {"question": question, "context": context}
            logger.info(
                "Invoking LLM to determine if question can be fully answered"
            )
            output = await asyncio.to_thread(
                lambda: can_be_answered_chain.invoke(input_data)
            )

            if isinstance(output, dict):
                can_be_answered = output.get("can_be_answered")
            else:
                can_be_answered = getattr(output, "can_be_answered", None)

            if can_be_answered:
                print("The question can be fully answered.")
                state["can_be_answered"] = True
                state["explanation"] = output.get("explanation", "")
            else:
                print("The question cannot be fully answered.")
                state["can_be_answered"] = False
                state["explanation"] = output.get("explanation", "")
            return state

        except Exception as e:
            logger.exception("Error in grade_generation_v_documents_and_question")
            raise RuntimeError(
                "Failed to grade generation vs documents and question"
            ) from e

    async def run_relevancy_pipeline(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 4. Check if the filtered content is relevant to the question using an LLM-based relevance check

            # Use an LLM to answer the question based on the relevant context
            if asyncio.iscoroutinefunction(self.answer_question_from_context):
                answer_state = await self.answer_question_from_context(state=state)
            else:
                answer_state = await asyncio.to_thread(
                    lambda: self.answer_question_from_context(state)
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
                    lambda: self.grade_generation_v_documents_and_question(answer_state)
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
