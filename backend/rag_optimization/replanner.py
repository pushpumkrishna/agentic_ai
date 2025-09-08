from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from backend.config.azure_models import AzureOpenAIModels
from backend.rag_optimization.second_retreival import Plan


# -----------------------------------------------------------
# Replanner: Update Plan Based on Progress and Aggregated Context
# -----------------------------------------------------------


# Define a Pydantic model for the possible results of the replanning action
class ActPossibleResults(BaseModel):
    """
    Represents the possible results of the replanning action.

    Attributes:
        plan (Plan): The updated plan to follow in the future.
        explanation (str): Explanation of the action taken or the reasoning behind the plan update.
    """

    plan: Plan = Field(description="Plan to follow in future.")
    explanation: str = Field(description="Explanation of the action.")


# Output schema for the task handler
class TaskHandlerOutput(BaseModel):
    """
    Output schema for the task handler.
    - query: The query to be either retrieved from the vector store, or the question that should be answered
    from context.
    - curr_context: The context to be based on in order to answer the query.
    - tool: The tool to be used; should be one of 'retrieve_chunks', 'retrieve_summaries', 'retrieve_quotes',
    or 'answer_from_context'.
    """

    query: str = Field(
        description="The query to be either retrieved from the vector store, "
        "or the question that should be answered from context."
    )
    curr_context: str = Field(
        description="The context to be based on in order to answer the query."
    )
    tool: str = Field(
        description="The tool to be used should be either retrieve_chunks, retrieve_summaries, "
        "retrieve_quotes, or answer_from_context."
    )


class RePlannerPipeline:
    """Given the original question, the current plan, the past steps, and the so far aggregated information,
    update the plan"""

    def replanner_pipeline(self):
        # Create a JSON output parser for the ActPossibleResults schema
        act_possible_results_parser = JsonOutputParser(
            pydantic_object=ActPossibleResults
        )

        # Prompt template for re-planning, instructing the LLM to update the plan based on the current state
        replanner_prompt_template = """
        - For the given objective, come up with a simple step by step plan of how to figure out the answer. 
        - This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
        - Do not add any superfluous steps. 
        - The result of the final step should be the final answer. 
        - Make sure that each step has all the information needed - do not skip steps.

        - Assume that the answer was not found yet and you need to update the plan accordingly, 
        so the plan should never be empty.
        
        Your objective was this:
        {question}
        
        Your original plan was this:
        {plan}
        
        You have currently done the follow steps:
        {past_steps}
        
        You already have the following context:
        {aggregated_context}
        
        Update your plan accordingly. If further steps are needed, fill out the plan with only those steps.
        Do not return previously done steps as part of the plan.
        
        The format is JSON so escape quotes and new lines.
        
        {format_instructions}
        """

        # Create a PromptTemplate object for the replanner
        replanner_prompt = PromptTemplate(
            template=replanner_prompt_template,
            input_variables=["question", "plan", "past_steps", "aggregated_context"],
            partial_variables={
                "format_instructions": act_possible_results_parser.get_format_instructions()
            },
        )

        # Initialize the LLM for replanning (using GPT-4o)
        replanner_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the re-planner chain: prompt -> LLM -> output parser
        replanner = replanner_prompt | replanner_llm | act_possible_results_parser

        return replanner

    def task_handler(self):
        # -----------------------------------------------------------
        # Task Handler: Decide Which Tool to Use for Each Task
        # -----------------------------------------------------------

        # Prompt template for the task handler LLM
        tasks_handler_prompt_template = """
        You are a task handler that receives a task {curr_task} and have to decide with tool to use to execute the task.
        You have the following tools at your disposal:
        Tool A: a tool that retrieves relevant information from a vector store of book chunks based on a given query.
        - use Tool A when you think the current task should search for information in the book chunks.
        Tool B: a tool that retrieves relevant information from a vector store of chapter summaries based on a given query.
        - use Tool B when you think the current task should search for information in the chapter summaries.
        Tool C: a tool that retrieves relevant information from a vector store of quotes from the book based on a given query.
        - use Tool C when you think the current task should search for information in the book quotes.
        Tool D: a tool that answers a question from a given context.
        - use Tool D ONLY when you the current task can be answered by the aggregated context {aggregated_context}

        You also receive the last tool used {last_tool}
        if {last_tool} was retrieve_chunks, use other tools than Tool A.

        You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
        You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
        if you decide to use Tools A,B or C, output the query to be used for the tool and also output the relevant tool.
        if you decide to use Tool D, output the question to be used for the tool, the context, and also that the tool to be used is Tool D.
        """

        # Create the prompt object for the task handler
        task_handler_prompt = PromptTemplate(
            template=tasks_handler_prompt_template,
            input_variables=[
                "curr_task",
                "aggregated_context",
                "last_tool",
                "past_steps",
                "question",
            ],
        )

        # Initialize the LLM for the task handler (using GPT-4o)
        task_handler_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the task handler chain: prompt -> LLM -> structured output
        task_handler_chain = (
            task_handler_prompt
            | task_handler_llm.with_structured_output(TaskHandlerOutput)
        )

        return task_handler_chain
