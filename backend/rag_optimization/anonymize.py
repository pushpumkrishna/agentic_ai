from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from backend.config.azure_models import AzureOpenAIModels
from typing import List
from backend.config.logging_lib import logger
from backend.rag_optimization.helper_functions import text_wrap
from backend.rag_optimization.step_7_second_retrieval import Plan


# Define a Pydantic model for the anonymized question output
class AnonymizeQuestion(BaseModel):
    """
    Output schema for the anonymized question.
    Attributes:
      anonymized_question (str): The question with named entities replaced by variables.
      mapping (dict): Mapping of variables to original named entities.
      explanation (str): Explanation of the anonymization process.
    """

    anonymized_question: str = Field(description="Anonymized question.")
    mapping: dict = Field(description="Mapping of original name entities to variables.")
    explanation: str = Field(description="Explanation of the action.")


class DeAnonymizePlan(BaseModel):
    """
    Output schema for the de-anonymized plan.
    Attributes:
        plan (List): Plan to follow in the future, with all variables replaced by the mapped words.
    """

    plan: List[str] = Field(
        description="Plan to follow in future. with all the variables replaced with the mapped words."
    )


class AnonymizeDeAnonymize:
    """
    In order to generate a general plan, without any biases based on prior knowledge of any LLM,
    we anonymize the input question, first, and map the name entities into variables
    """

    # -----------------------------------------------------------
    # Anonymize Question Chain: Replace Named Entities with Variables
    # -----------------------------------------------------------

    async def anonymize(self):
        # Create a JSON output parser for the AnonymizeQuestion schema
        anonymize_question_parser = JsonOutputParser(pydantic_object=AnonymizeQuestion)

        # Prompt template for the LLM to anonymize questions
        anonymize_question_prompt_template = """
        You are a question anonymizer. The input you receive is a string containing several words that
        construct a question {question}. Your goal is to change all name entities in the input to variables, 
        and remember the mapping of the original name entities to the variables.
        
        Example 1:
          if the input is "who is harry potter?" the output should be "who is X?" and the mapping should be 
          {{"X": "harry potter"}}
        
        Example 2:
          if the input is "how did the bad guy played with the alex and rony?"
          the output should be "how did the X played with the Y and Z?" and the mapping should be 
          {{"X": "bad guy", "Y": "alex", "Z": "rony"}}
        
        You must replace all name entities in the input with variables, and remember the mapping of the original 
        name entities to the variables.
        
        Output the anonymized question and the mapping in a JSON format.
        {format_instructions}
        """

        # Create the PromptTemplate object for the anonymization task
        anonymize_question_prompt = PromptTemplate(
            template=anonymize_question_prompt_template,
            input_variables=["question"],
            partial_variables={
                "format_instructions": anonymize_question_parser.get_format_instructions()
            },
        )

        # Initialize the LLM for anonymization (using GPT-4o)
        anonymize_question_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the anonymization chain: prompt -> LLM -> output parser
        anonymize_question_chain = (
            anonymize_question_prompt
            | anonymize_question_llm
            | anonymize_question_parser
        )

        return anonymize_question_chain

        # -----------------------------------------------------------
        # De-Anonymize Plan Chain: Replace Variables in Plan with Mapped Words
        # -----------------------------------------------------------

    async def de_anonymize(self):
        # Prompt template for de-anonymizing a plan
        de_anonymize_plan_prompt_template = (
            "You receive a list of tasks: {plan}, where some of the words are replaced with mapped variables. "
            "You also receive the mapping for those variables to words {mapping}. "
            "Replace all the variables in the list of tasks with the mapped words. "
            "If no variables are present, return the original list of tasks. "
            "In any case, just output the updated list of tasks in a JSON format as described here, "
            "without any additional text apart from the JSON."
        )

        # Create the PromptTemplate object for the de-anonymization task
        de_anonymize_plan_prompt = PromptTemplate(
            template=de_anonymize_plan_prompt_template,
            input_variables=["plan", "mapping"],
        )

        # Initialize the LLM for de-anonymization (using GPT-4o)
        de_anonymize_plan_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the de-anonymization chain: prompt -> LLM -> structured output
        de_anonymize_plan_chain = (
            de_anonymize_plan_prompt
            | de_anonymize_plan_llm.with_structured_output(DeAnonymizePlan)
        )

        return de_anonymize_plan_chain

    async def plan(self):
        # -----------------------------------------------------------
        # Example: Anonymize, Plan, and De-anonymize a Question
        # -----------------------------------------------------------

        # 1. Define the question to answer
        state1 = {"question": "how did the harry beat quirrell? \n"}
        print(f"question: {state1['question']}")

        anonymize_question_chain = await self.anonymize()
        de_anonymize_plan_chain = await self.de_anonymize()

        # 2. Anonymize the question (replace named entities with variables)
        anonymized_question_output = await anonymize_question_chain.ainvoke(state1)
        anonymized_question = anonymized_question_output[
            "anonymized_question"
        ]  # The anonymized question
        mapping = anonymized_question_output[
            "mapping"
        ]  # Mapping of variables to original entities

        print(f"anonymized_query: {anonymized_question} \n")
        print(f"mapping: {mapping} \n")

        # Prompt template for generating a plan from a question
        planner_prompt = """
        For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

        This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
        Do not add any superfluous steps. 
        The result of the final step should be the final answer. 
        Make sure that each step has all the information needed - do not skip steps.
                    """

        planner_prompt = PromptTemplate(
            template=planner_prompt,
            input_variables=["question"],
        )

        # Initialize the LLM for planning (using GPT-4o)
        planner_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the planning chain: prompt -> LLM -> structured output
        planner = planner_prompt | planner_llm.with_structured_output(Plan)

        # 3. Generate a step-by-step plan for the anonymized question
        plan = planner.invoke({"question": anonymized_question})
        logger.info(text_wrap(f"plan: {plan.steps}"))
        print("")

        # 4. De-anonymize the plan (replace variables back with original entities)
        de_anonymized_plan = await de_anonymize_plan_chain.ainvoke(
            {"plan": plan.steps, "mapping": mapping}
        )
        print(text_wrap(f"de_anonymized_plan: {de_anonymized_plan.plan}"))

    async def sample_run(self):
        # -----------------------------------------------------------
        # Example: Anonymize, Plan, and De-anonymize a Question
        # -----------------------------------------------------------

        # 1. Define the question to answer
        state1 = {"question": "how did the harry beat quirrell? \n"}
        print(f"question: {state1['question']}")

        anonymize_question_chain = await self.anonymize()
        de_anonymize_plan_chain = await self.de_anonymize()

        # 2. Anonymize the question (replace named entities with variables)
        anonymized_question_output = await anonymize_question_chain.ainvoke(state1)
        anonymized_question = anonymized_question_output[
            "anonymized_question"
        ]  # The anonymized question
        mapping = anonymized_question_output[
            "mapping"
        ]  # Mapping of variables to original entities

        print(f"anonymized_query: {anonymized_question} \n")
        print(f"mapping: {mapping} \n")

        # Prompt template for generating a plan from a question
        planner_prompt = """
                For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

                This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
                Do not add any superfluous steps. 
                The result of the final step should be the final answer. 
                Make sure that each step has all the information needed - do not skip steps.
                            """

        planner_prompt = PromptTemplate(
            template=planner_prompt,
            input_variables=["question"],
        )

        # Initialize the LLM for planning (using GPT-4o)
        planner_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the planning chain: prompt -> LLM -> structured output
        planner = planner_prompt | planner_llm.with_structured_output(Plan)

        # 3. Generate a step-by-step plan for the anonymized question
        plan = await planner.ainvoke({"question": anonymized_question})
        print(text_wrap(f"plan: {plan.steps}"))
        print("")

        # 4. De-anonymize the plan (replace variables back with original entities)
        de_anonymized_plan = de_anonymize_plan_chain.invoke(
            {"plan": plan.steps, "mapping": mapping}
        )
        print(text_wrap(f"de_anonymized_plan: {de_anonymized_plan.plan}"))
