from pprint import pprint
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from backend.config.azure_models import AzureOpenAIModels
from backend.rag_optimization.anonymize import AnonymizeDeAnonymize
from backend.rag_optimization.helper_functions import text_wrap
from backend.rag_optimization.replanner import RePlannerPipeline
from backend.rag_optimization.step_7_second_retrieval import PlanExecute, SecondRetrieval, Plan
from langgraph.graph import StateGraph, END
from IPython.display import display, Image


class CanBeAnsweredAlready(BaseModel):
    """
    Output schema for checking if the question can be fully answered from the given context.
    Attributes:
        can_be_answered (bool): Whether the question can be fully answered or not based on the given context.
    """

    can_be_answered: bool = Field(
        description="Whether the question can be fully answered or not based on the given context."
    )


class TaskHandlerChainRun(SecondRetrieval):
    """Define the sophisticated pipeline graph functions"""

    def __init__(
            self,
            chunks_vector_store: FAISS,
            chapter_summaries_vector_store: FAISS,
            book_quotes_vectorstore: FAISS,
            init_state,
            **data: Any
    ):

        super().__init__(chunks_vector_store, chapter_summaries_vector_store, book_quotes_vectorstore, init_state,
                         **data)

    @staticmethod
    async def run_task_handler_chain(state: PlanExecute):
        """
        Run the task handler chain to decide which tool to use to execute the task.

        Args:
            state: The current state of the plan execution.

        Returns:
            The updated state of the plan execution.
        """
        state["curr_state"] = "task_handler"
        print("the current plan is:")
        print(state["plan"])
        pprint("--------------------")

        # Initialize past_steps if not present
        if not state.get("past_steps", ""):
            state["past_steps"] = []

        # Get the current task from the plan
        curr_task = state["plan"][0]

        # Prepare inputs for the task handler chain
        inputs = {
            "curr_task": curr_task,
            "aggregated_context": state.get("aggregated_context", ""),
            "last_tool": state.get("tool", ""),
            "past_steps": state.get("past_steps", ""),
            "question": state.get("question", ""),
        }

        # Invoke the task handler chain
        task_handler_chain = RePlannerPipeline().task_handler()
        output = task_handler_chain.invoke(inputs)

        # Update state with the completed task
        state["past_steps"].append(curr_task)
        state["plan"].pop(0)

        # Decide which tool to use based on output
        if output.tool == "retrieve_chunks":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"] = "retrieve_chunks"
        elif output.tool == "retrieve_summaries":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"] = "retrieve_summaries"
        elif output.tool == "retrieve_quotes":
            state["query_to_retrieve_or_answer"] = output.query
            state["tool"] = "retrieve_quotes"
        elif output.tool == "answer_from_context":
            state["query_to_retrieve_or_answer"] = output.query
            state["curr_context"] = output.curr_context
            state["tool"] = "answer"
        else:
            raise ValueError(
                "Invalid tool was outputted. Must be either 'retrieve' or 'answer_from_context'"
            )
        return state

    @staticmethod
    def retrieve_or_answer(state: PlanExecute):
        """
        Decide whether to retrieve or answer the question based on the current state.

        Args:
            state: The current state of the plan execution.

        Returns:
            String indicating the chosen tool.
        """
        state["curr_state"] = "decide_tool"
        print("deciding whether to retrieve or answer")
        if state["tool"] == "retrieve_chunks":
            return "chosen_tool_is_retrieve_chunks"
        elif state["tool"] == "retrieve_summaries":
            return "chosen_tool_is_retrieve_summaries"
        elif state["tool"] == "retrieve_quotes":
            return "chosen_tool_is_retrieve_quotes"
        elif state["tool"] == "answer":
            return "chosen_tool_is_answer"
        else:
            raise ValueError(
                f"Invalid tool was outputted: {output.tool}. "
                "Must be one of 'retrieve_chunks', 'retrieve_summaries', "
                "'retrieve_quotes', or 'answer_from_context'."
            )

    async def run_qualitative_chunks_retrieval_workflow(self, state):
        """
            Run the qualitative chunks retrieval workflow.

            Args:
                state: The current state of the plan execution.

            Returns:
                The state with the updated aggregated context.
            """
        output = {}
        state["curr_state"] = "retrieve_chunks"
        print("Running the qualitative chunks retrieval workflow...")
        question = state["query_to_retrieve_or_answer"]
        inputs = {"question": question}

        # Stream outputs from the workflow app
        qualitative_chunks_retrieval_workflow_app = await (
            self.chunks_retrieval_workflow_graph_construction()
        )
        async for output in qualitative_chunks_retrieval_workflow_app.astream(inputs):
            for _, _ in output.items():
                pass
            pprint("--------------------")
        # Aggregate the retrieved context
        if not state.get("aggregated_context", ""):
            state["aggregated_context"] = ""

        output["relevant_context"] = ""
        state["aggregated_context"] += output["relevant_context"]
        return state

    async def run_qualitative_summaries_retrieval_workflow(self, state):
        """
        Run the qualitative summaries retrieval workflow.

        Args:
            state: The current state of the plan execution.

        Returns:
            The state with the updated aggregated context.
        """
        output = {}
        state["curr_state"] = "retrieve_summaries"
        print("Running the qualitative summaries retrieval workflow...")
        question = state["query_to_retrieve_or_answer"]
        inputs = {"question": question}
        qualitative_summaries_retrieval_workflow_app = await (
            self.summaries_retrieval_workflow_graph_construction()
        )
        async for output in qualitative_summaries_retrieval_workflow_app.astream(inputs):
            for _, _ in output.items():
                pass
            pprint("--------------------")
        if not state.get("aggregated_context", ""):
            state["aggregated_context"] = ""
        output["relevant_context"] = ""
        state["aggregated_context"] += output["relevant_context"]
        return state

    async def run_qualitative_book_quotes_retrieval_workflow(self, state):
        """
        Run the qualitative book quotes retrieval workflow.

        Args:
            state: The current state of the plan execution.

        Returns:
            The state with the updated aggregated context.
        """
        output = {}
        state["curr_state"] = "retrieve_book_quotes"
        print("Running the qualitative book quotes retrieval workflow...")
        question = state["query_to_retrieve_or_answer"]
        inputs = {"question": question}

        qualitative_book_quotes_retrieval_workflow_app = await (
            self.book_quotes_retrieval_workflow_graph_construction()
        )
        async for output in qualitative_book_quotes_retrieval_workflow_app.astream(inputs):
            for _, _ in output.items():
                pass
            pprint("--------------------")
        if not state["aggregated_context"]:
            state["aggregated_context"] = ""
        output["relevant_context"] = ""
        state["aggregated_context"] += output["relevant_context"]
        return state

    async def run_qualitative_answer_workflow(self, state):
        """
        Run the qualitative answer workflow.

        Args:
            state: The current state of the plan execution.

        Returns:
            The state with the updated aggregated context.
        """
        global output

        state["curr_state"] = "answer"
        print("Running the qualitative answer workflow...")
        question = state["query_to_retrieve_or_answer"]
        context = state["curr_context"]
        inputs = {"question": question, "context": context}

        qualitative_answer_workflow_app = await self.answer_workflow_graph_construction()
        async for output in qualitative_answer_workflow_app.astream(inputs):
            for _, _ in output.items():
                pass
            pprint("--------------------")
        if not state["aggregated_context"]:
            state["aggregated_context"] = ""
        output["answer"] = ""
        state["aggregated_context"] += output["answer"]
        return state

    async def run_qualitative_answer_workflow_for_final_answer(self, state):
        """
        Run the qualitative answer workflow for the final answer.

        Args:
            state: The current state of the plan execution.

        Returns:
            The state with the updated response.
        """
        state["curr_state"] = "get_final_answer"
        print("Running the qualitative answer workflow for final answer...")
        question = state["question"]
        context = state["aggregated_context"]
        inputs = {"question": question, "context": context}

        qualitative_answer_workflow_app = await self.answer_workflow_graph_construction()
        async for output in qualitative_answer_workflow_app.astream(inputs):
            for _, value in output.items():
                pass
            pprint("--------------------")
        state["response"] = value
        return state

    @staticmethod
    async def anonymize_queries(state: PlanExecute):
        """
        Anonymizes the question.

        Args:
            state: The current state of the plan execution.

        Returns:
            The updated state with the anonymized question and mapping.
        """
        state["curr_state"] = "anonymize_question"
        print("Anonymizing question")
        pprint("--------------------")

        anonymize_question_chain = await AnonymizeDeAnonymize().anonymize()
        anonymized_question_output = await anonymize_question_chain.ainvoke(state["question"])
        anonymized_question = anonymized_question_output["anonymized_question"]
        print(f"anonimized_querry: {anonymized_question}")
        pprint("--------------------")
        mapping = anonymized_question_output["mapping"]
        state["anonymized_question"] = anonymized_question
        state["mapping"] = mapping
        return state

    @staticmethod
    async def de_anonymize_queries(state: PlanExecute):
        """
        De-anonymizes the plan.

        Args:
            state: The current state of the plan execution.

        Returns:
            The updated state with the de-anonymized plan.
        """
        state["curr_state"] = "de_anonymize_plan"
        print("De-anonymizing plan")
        pprint("--------------------")

        de_anonymize_plan_chain = await AnonymizeDeAnonymize().de_anonymize()
        deanonimzed_plan = await de_anonymize_plan_chain.ainvoke(
            {"plan": state["plan"], "mapping": state["mapping"]}
        )
        state["plan"] = deanonimzed_plan.plan
        print(f"de-anonimized_plan: {deanonimzed_plan.plan}")
        return state

    @staticmethod
    async def plan_step(state: PlanExecute):
        """
        Plans the next step.

        Args:
            state: The current state of the plan execution.

        Returns:
            The updated state with the plan.
        """
        state["curr_state"] = "planner"
        print("Planning step")
        pprint("--------------------")

        # Prompt template for generating a plan from a question
        planner_prompt = """
                    For the given query {question}, come up with a simple step by step plan of how to figure out the answer. 

                    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
                    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
                    """

        planner_prompt = PromptTemplate(
            template=planner_prompt,
            input_variables=["question"],
        )

        # Initialize the LLM for planning (using GPT-4o)
        planner_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the planning chain: prompt -> LLM -> structured output
        planner = planner_prompt | planner_llm.with_structured_output(Plan)
        plan = await planner.ainvoke({"question": state["anonymized_question"]})
        state["plan"] = plan.steps
        print(f"plan: {state['plan']}")
        return state

    async def break_down_plan_step(self, state: PlanExecute):
        """
        Breaks down the plan steps into retrievable or answerable tasks.

        Args:
            state: The current state of the plan execution.

        Returns:
            The updated state with the refined plan.
        """
        state["curr_state"] = "break_down_plan"
        print("Breaking down plan steps into retrievable or answerable tasks")
        pprint("--------------------")

        # Prompt template for refining a plan so that each step is executable by a retrieval or answer operation
        break_down_plan_prompt_template = """
        You receive a plan {plan} which contains a series of steps to follow in order to answer a query. 
        You need to go through the plan and refine it according to these rules:
        1. Every step must be executable by one of the following:
            i. Retrieving relevant information from a vector store of book chunks
            ii. Retrieving relevant information from a vector store of chapter summaries
            iii. Retrieving relevant information from a vector store of book quotes
            iv. Answering a question from a given context.
        2. Every step should contain all the information needed to execute it.

        Output the refined plan.
        """

        # Create a PromptTemplate for the LLM
        break_down_plan_prompt = PromptTemplate(
            template=break_down_plan_prompt_template,
            input_variables=["plan"],
        )

        # Initialize the LLM for plan breakdown (using GPT-4o)
        break_down_plan_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the chain: prompt -> LLM -> structured output (Plan)
        break_down_plan_chain = (
                break_down_plan_prompt | break_down_plan_llm.with_structured_output(Plan)
        )

        refined_plan = await break_down_plan_chain.ainvoke({"plan": state["plan"]})
        state["plan"] = refined_plan.steps
        return state

    async def replan_step(self, state: PlanExecute):
        """
        Replans the next step.

        Args:
            state: The current state of the plan execution.

        Returns:
            The updated state with the plan.
        """
        state["curr_state"] = "replan"
        print("Replanning step")
        pprint("--------------------")
        inputs = {
            "question": state["question"],
            "plan": state["plan"],
            "past_steps": state["past_steps"],
            "aggregated_context": state["aggregated_context"],
        }
        replanner = RePlannerPipeline().replanner_pipeline()
        output = await replanner.ainvoke(inputs)
        state["plan"] = output["plan"]["steps"]
        return state

    async def can_be_answered(self, state: PlanExecute):
        """
        Determines if the question can be answered.

        Args:
            state: The current state of the plan execution.

        Returns:
            String indicating whether the original question can be answered or not.
        """
        state["curr_state"] = "can_be_answered_already"
        print("Checking if the ORIGINAL QUESTION can be answered already")
        pprint("--------------------")
        question = state["question"]
        context = state["aggregated_context"]
        inputs = {"question": question, "context": context}

        # Prompt template for the LLM to determine answerability
        can_be_answered_already_prompt_template = """
        You receive a query: {question} and a context: {context}.
        You need to determine if the question can be fully answered relying only on the given context.
        The only information you have and can rely on is the context you received. 
        You have no prior knowledge of the question or the context.
        If you think the question can be answered based on the context, output 'true', otherwise output 'false'.
        """

        # Create the PromptTemplate object
        can_be_answered_already_prompt = PromptTemplate(
            template=can_be_answered_already_prompt_template,
            input_variables=["question", "context"],
        )

        # Initialize the LLM for this task (using GPT-4o)
        can_be_answered_already_llm = AzureOpenAIModels().get_azure_model_4()

        # Compose the chain: prompt -> LLM -> structured output
        can_be_answered_already_chain = (
                can_be_answered_already_prompt
                | can_be_answered_already_llm.with_structured_output(CanBeAnsweredAlready)
        )
        output = await can_be_answered_already_chain.ainvoke(inputs)

        if output.can_be_answered:
            print("The ORIGINAL QUESTION can be fully answered already.")
            pprint("--------------------")
            print("the aggregated context is:")
            print(text_wrap(state["aggregated_context"]))
            print("--------------------")
            return "can_be_answered_already"
        else:
            print("The ORIGINAL QUESTION cannot be fully answered yet.")
            pprint("--------------------")
            return "cannot_be_answered_yet"

    def make_final_graph(self):
        """
        # -----------------------------------------------------------
        # Define the Plan-and-Execute Agent Workflow Graph
        # -----------------------------------------------------------
        """

        # Initialize the workflow graph with the PlanExecute state
        agent_workflow = StateGraph(PlanExecute)

        # -------------------------
        # Add Nodes (Steps/Functions)
        # -------------------------

        # 1. Anonymize the question (replace named entities with variables)
        agent_workflow.add_node("anonymize_question", self.anonymize_queries)

        # 2. Generate a step-by-step plan for the anonymized question
        agent_workflow.add_node("planner", self.plan_step)

        # 3. De-anonymize the plan (replace variables back with original entities)
        agent_workflow.add_node("de_anonymize_plan", self.de_anonymize_queries)

        # 4. Break down the plan into retrievable/answerable tasks
        agent_workflow.add_node("break_down_plan", self.break_down_plan_step)

        # 5. Decide which tool to use for the current task
        agent_workflow.add_node("task_handler", self.run_task_handler_chain)

        # 6. Retrieve relevant book chunks
        agent_workflow.add_node(
            "retrieve_chunks", self.run_qualitative_chunks_retrieval_workflow
        )

        # 7. Retrieve relevant chapter summaries
        agent_workflow.add_node(
            "retrieve_summaries", self.run_qualitative_summaries_retrieval_workflow
        )

        # 8. Retrieve relevant book quotes
        agent_workflow.add_node(
            "retrieve_book_quotes", self.run_qualitative_book_quotes_retrieval_workflow
        )

        # 9. Answer the question from the aggregated context
        agent_workflow.add_node("answer", self.run_qualitative_answer_workflow)

        # 10. Replan if needed (update plan based on progress/context)
        agent_workflow.add_node("replan", self.replan_step)

        # 11. Get the final answer from the aggregated context
        agent_workflow.add_node(
            "get_final_answer", self.run_qualitative_answer_workflow_for_final_answer
        )

        # -------------------------
        # Define Workflow Edges (Transitions)
        # -------------------------

        # Set the entry point of the workflow
        agent_workflow.set_entry_point("anonymize_question")

        # Anonymize -> Plan
        agent_workflow.add_edge("anonymize_question", "planner")

        # Plan -> De-anonymize
        agent_workflow.add_edge("planner", "de_anonymize_plan")

        # De-anonymize -> Break down plan
        agent_workflow.add_edge("de_anonymize_plan", "break_down_plan")

        # Break down plan -> Task handler
        agent_workflow.add_edge("break_down_plan", "task_handler")

        # Task handler -> (conditional) Retrieve or Answer
        agent_workflow.add_conditional_edges(
            "task_handler",
            self.retrieve_or_answer,
            {
                "chosen_tool_is_retrieve_chunks": "retrieve_chunks",
                "chosen_tool_is_retrieve_summaries": "retrieve_summaries",
                "chosen_tool_is_retrieve_quotes": "retrieve_book_quotes",
                "chosen_tool_is_answer": "answer",
            },
        )

        # Retrieval/Answer nodes -> Replan
        agent_workflow.add_edge("retrieve_chunks", "replan")
        agent_workflow.add_edge("retrieve_summaries", "replan")
        agent_workflow.add_edge("retrieve_book_quotes", "replan")
        agent_workflow.add_edge("answer", "replan")

        # Replan -> (conditional) Get final answer or continue
        agent_workflow.add_conditional_edges(
            "replan",
            self.can_be_answered,
            {
                "can_be_answered_already": "get_final_answer",
                "cannot_be_answered_yet": "break_down_plan",
            },
        )

        # Get final answer -> End
        agent_workflow.add_edge("get_final_answer", END)

        # -------------------------
        # Compile and Visualize the Workflow
        # -------------------------

        plan_and_execute_app = agent_workflow.compile()

        # Display the workflow graph as a Mermaid diagram
        display(Image(plan_and_execute_app.get_graph(xray=True).draw_mermaid_png()))

        return plan_and_execute_app
