from typing import TypedDict, List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image
from backend.config.azure_models import AzureOpenAIModels
from backend.rag_optimization.retrieve_data import RetrieveData
from pprint import pprint


# -----------------------------------------------------------
# Retrieval Functions for Different Context Types
# -----------------------------------------------------------


class QualitativeRetrievalGraphState(TypedDict):
    """
    Represents the state of the qualitative retrieval graph.

    Attributes:
        question (str): The input question to be answered.
        context (str): The context retrieved from the source (e.g., book chunks, summaries, or quotes).
        relevant_context (str): The distilled or filtered context that is most relevant to the question.
    """

    question: str
    context: str
    relevant_context: str


# Define the output schema for the plan
class Plan(BaseModel):
    """
    Represents a step-by-step plan to answer a given question.
    Attributes:
        steps (List[str]): Ordered list of steps to follow.
    """

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class PlanExecute(TypedDict):
    """
    Represents the state at each step of the plan execution pipeline.

    Attributes:
        curr_state (str): The current state or status of the execution.
        question (str): The original user question.
        anonymized_question (str): The anonymized version of the question (entities replaced with variables).
        query_to_retrieve_or_answer (str): The query to be used for retrieval or answering.
        plan (List[str]): The current plan as a list of steps to execute.
        past_steps (List[str]): List of steps that have already been executed.
        mapping (dict): Mapping of anonymized variables to original named entities.
        curr_context (str): The current context used for answering or retrieval.
        aggregated_context (str): The accumulated context from previous steps.
        tool (str): The tool or method used for the current step (e.g., retrieval, answer).
        response (str): The response or output generated at this step.
    """

    curr_state: str
    question: str
    anonymized_question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    mapping: Dict[str, str]
    curr_context: str
    aggregated_context: str
    tool: str
    response: str


# Define the state for the answer workflow graph
class QualitativeAnswerGraphState(TypedDict):
    """
    Represents the state of the qualitative answer graph.

    Attributes:
        question (str): The input question to be answered.
        context (str): The context used to answer the question.
        answer (str): The generated answer to the question.
    """

    question: str
    context: str
    answer: str


class IsGroundedOnFacts(BaseModel):
    """
    Output schema for checking if the answer is grounded in the provided context.
    """

    grounded_on_facts: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class SecondRetrieval(RetrieveData):
    def __init__(
        self,
        chunks_vector_store: FAISS,
        chapter_summaries_vector_store: FAISS,
        book_quotes_vectorstore: FAISS,
        **data: Any,
    ):
        super().__init__(
            chunks_vector_store,
            chapter_summaries_vector_store,
            book_quotes_vectorstore,
            **data,
        )

    async def retrieve_chunks_context_per_question(self, state):
        """
        Retrieves relevant context for a given question from the book chunks.

        Args:
            state (dict): A dictionary containing the question to answer, with key "question".

        Returns:
            dict: A dictionary with keys:
                - "context": Aggregated context string from relevant book chunks.
                - "question": The original question.
        """
        print("Retrieving relevant chunks...")
        question = state["question"]
        # Retrieve relevant book chunks using the retriever
        docs = self.chunks_query_retriever.get_relevant_documents(question)
        # Concatenate the content of the retrieved documents
        context = " ".join(doc.page_content for doc in docs)
        context = self.escape_quotes(context)
        return {"context": context, "question": question}

    def escape_quotes(self, text):
        """
        Escapes both single and double quotes in a string.

        Args:
            text (str): The string to escape.

        Returns:
            str: The string with single and double quotes escaped.
        """
        return text.replace('"', '\\"').replace("'", "\\'")

    def retrieve_summaries_context_per_question(self, state):
        """
        Retrieves relevant context for a given question from chapter summaries.

        Args:
            state (dict): A dictionary containing the question to answer, with key "question".

        Returns:
            dict: A dictionary with keys:
                - "context": Aggregated context string from relevant chapter summaries.
                - "question": The original question.
        """
        print("Retrieving relevant chapter summaries...")
        question = state["question"]
        # Retrieve relevant chapter summaries using the retriever
        docs_summaries = self.chapter_summaries_query_retriever.get_relevant_documents(
            question
        )
        # Concatenate the content of the retrieved summaries, including chapter citation
        context_summaries = " ".join(
            f"{doc.page_content} (Chapter {doc.metadata['chapter']})"
            for doc in docs_summaries
        )
        context_summaries = self.escape_quotes(context_summaries)
        return {"context": context_summaries, "question": question}

    def retrieve_book_quotes_context_per_question(self, state):
        """
        Retrieves relevant context for a given question from book quotes.

        Args:
            state (dict): A dictionary containing the question to answer, with key "question".

        Returns:
            dict: A dictionary with keys:
                - "context": Aggregated context string from relevant book quotes.
                - "question": The original question.
        """
        question = state["question"]
        print("Retrieving relevant book quotes...")
        # Retrieve relevant book quotes using the retriever
        docs_book_quotes = self.book_quotes_query_retriever.get_relevant_documents(
            question
        )
        # Concatenate the content of the retrieved quotes
        book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)
        book_quotes_context = self.escape_quotes(book_quotes)
        return {"context": book_quotes_context, "question": question}

    # -----------------------------------------------------------
    # Qualitative Chunks Retrieval Workflow Graph Construction
    # -----------------------------------------------------------

    def chunks_retrieval_workflow_graph_construction(self):
        # 1. Create the workflow graph object with the appropriate state type
        qualitative_chunks_retrieval_workflow = StateGraph(
            QualitativeRetrievalGraphState
        )

        # 2. Define and add nodes to the graph
        # Node: Retrieve relevant context from book chunks for the question
        qualitative_chunks_retrieval_workflow.add_node(
            "retrieve_chunks_context_per_question",
            self.retrieve_chunks_context_per_question,
        )
        # Node: Use LLM to keep only the relevant content from the retrieved context
        qualitative_chunks_retrieval_workflow.add_node(
            "keep_only_relevant_content", self.keep_only_relevant_content
        )

        # 3. Set the entry point of the workflow
        qualitative_chunks_retrieval_workflow.set_entry_point(
            "retrieve_chunks_context_per_question"
        )

        # 4. Add edges to define the workflow
        # After retrieving context, filter to keep only relevant content
        qualitative_chunks_retrieval_workflow.add_edge(
            "retrieve_chunks_context_per_question", "keep_only_relevant_content"
        )

        # Conditional edge: After filtering, check if distilled content is grounded in the original context
        # If grounded, end; if not, repeat filtering
        qualitative_chunks_retrieval_workflow.add_conditional_edges(
            "keep_only_relevant_content",
            self.is_distilled_content_grounded_on_content,
            {
                "grounded on the original context": END,
                "not grounded on the original context": "keep_only_relevant_content",
            },
        )

        # 5. Compile the workflow graph into an executable app
        qualitative_chunks_retrieval_workflow_app = (
            qualitative_chunks_retrieval_workflow.compile()
        )

        # 6. Display the workflow graph as a Mermaid diagram
        display(
            Image(
                qualitative_chunks_retrieval_workflow_app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )

        return qualitative_chunks_retrieval_workflow_app

    # -----------------------------------------------------------
    # Qualitative Summaries Retrieval Workflow Graph Construction
    # -----------------------------------------------------------

    def summaries_retrieval_workflow_graph_construction(self):
        # 1. Create the workflow graph object with the appropriate state type
        qualitative_summaries_retrieval_workflow = StateGraph(
            QualitativeRetrievalGraphState
        )

        # 2. Define and add nodes to the graph
        # Node: Retrieve relevant context from chapter summaries for the question
        qualitative_summaries_retrieval_workflow.add_node(
            "retrieve_summaries_context_per_question",
            self.retrieve_summaries_context_per_question,
        )
        # Node: Use LLM to keep only the relevant content from the retrieved context
        qualitative_summaries_retrieval_workflow.add_node(
            "keep_only_relevant_content", self.keep_only_relevant_content
        )

        # 3. Set the entry point of the workflow
        qualitative_summaries_retrieval_workflow.set_entry_point(
            "retrieve_summaries_context_per_question"
        )

        # 4. Add edges to define the workflow
        # After retrieving context, filter to keep only relevant content
        qualitative_summaries_retrieval_workflow.add_edge(
            "retrieve_summaries_context_per_question", "keep_only_relevant_content"
        )

        # Conditional edge: After filtering, check if distilled content is grounded in the original context
        # If grounded, end; if not, repeat filtering
        qualitative_summaries_retrieval_workflow.add_conditional_edges(
            "keep_only_relevant_content",
            self.is_distilled_content_grounded_on_content,
            {
                "grounded on the original context": END,
                "not grounded on the original context": "keep_only_relevant_content",
            },
        )

        # 5. Compile the workflow graph into an executable app
        qualitative_summaries_retrieval_workflow_app = (
            qualitative_summaries_retrieval_workflow.compile()
        )

        # 6. Display the workflow graph as a Mermaid diagram
        display(
            Image(
                qualitative_summaries_retrieval_workflow_app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )

        return qualitative_summaries_retrieval_workflow_app

    # -----------------------------------------------------------
    # Qualitative Book Quotes Retrieval Workflow Graph Construction
    # -----------------------------------------------------------

    def book_quotes_retrieval_workflow_graph_construction(self):
        # 1. Create the workflow graph object with the appropriate state type
        qualitative_book_quotes_retrieval_workflow = StateGraph(
            QualitativeRetrievalGraphState
        )

        # 2. Define and add nodes to the graph
        # Node: Retrieve relevant context from book quotes for the question
        qualitative_book_quotes_retrieval_workflow.add_node(
            "retrieve_book_quotes_context_per_question",
            self.retrieve_book_quotes_context_per_question,
        )
        # Node: Use LLM to keep only the relevant content from the retrieved context
        qualitative_book_quotes_retrieval_workflow.add_node(
            "keep_only_relevant_content", self.keep_only_relevant_content
        )

        # 3. Set the entry point of the workflow
        qualitative_book_quotes_retrieval_workflow.set_entry_point(
            "retrieve_book_quotes_context_per_question"
        )

        # 4. Add edges to define the workflow
        # After retrieving context, filter to keep only relevant content
        qualitative_book_quotes_retrieval_workflow.add_edge(
            "retrieve_book_quotes_context_per_question", "keep_only_relevant_content"
        )

        # Conditional edge: After filtering, check if distilled content is grounded in the original context
        # If grounded, end; if not, repeat filtering
        qualitative_book_quotes_retrieval_workflow.add_conditional_edges(
            "keep_only_relevant_content",
            self.is_distilled_content_grounded_on_content,
            {
                "grounded on the original context": END,
                "not grounded on the original context": "keep_only_relevant_content",
            },
        )

        # 5. Compile the workflow graph into an executable app
        qualitative_book_quotes_retrieval_workflow_app = (
            qualitative_book_quotes_retrieval_workflow.compile()
        )

        # 6. Display the workflow graph as a Mermaid diagram
        display(
            Image(
                qualitative_book_quotes_retrieval_workflow_app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )

        return qualitative_book_quotes_retrieval_workflow_app

    # -----------------------------------------------------------
    # Initialize the state for the retrieval/answering pipeline
    # -----------------------------------------------------------

    def second_retrieval_pipeline(self):
        # This dictionary defines the initial state for the workflow,
        # specifying the question to be answered.
        # You can modify the "question" value to test different queries.

        qualitative_chunks_retrieval_workflow_app = (
            self.chunks_retrieval_workflow_graph_construction()
        )
        qualitative_summaries_retrieval_workflow_app = (
            self.summaries_retrieval_workflow_graph_construction()
        )
        qualitative_book_quotes_retrieval_workflow_app = (
            self.book_quotes_retrieval_workflow_graph_construction()
        )

        init_state = {
            "question": "worse than getting killed"  # The question to answer
        }
        # -----------------------------------------------------------
        # Test the book chunks retrieval workflow
        # -----------------------------------------------------------

        # Stream the outputs from the qualitative_chunks_retrieval_workflow_app
        # for the given initial state (init_state contains the question to answer).
        for output in qualitative_chunks_retrieval_workflow_app.stream(init_state):
            # Iterate through the output items (node name, value)
            for _, value in output.items():
                pass  # The value variable will hold the latest state after each node execution
            pprint(
                "--------------------"
            )  # Print a separator for clarity between steps

        # After the workflow completes, print the final relevant context extracted
        print(f"relevant context: {value['relevant_context']}")
        # -----------------------------------------------------------
        # Test the chapter summaries retrieval workflow
        # -----------------------------------------------------------

        # Stream the outputs from the qualitative_summaries_retrieval_workflow_app
        # for the given initial state (init_state contains the question to answer).
        for output in qualitative_summaries_retrieval_workflow_app.stream(init_state):
            # Iterate through the output items (node name, value)
            for _, value in output.items():
                pass  # The value variable will hold the latest state after each node execution
            pprint(
                "--------------------"
            )  # Print a separator for clarity between steps

        # After the workflow completes, print the final relevant context extracted
        print(f"relevant context: {value['relevant_context']}")
        # -----------------------------------------------------------
        # Test the book quotes retrieval workflow
        # -----------------------------------------------------------

        # Stream the outputs from the qualitative_book_quotes_retrieval_workflow_app
        # for the given initial state (init_state contains the question to answer).
        for output in qualitative_book_quotes_retrieval_workflow_app.stream(init_state):
            # Iterate through the output items (node name, value)
            for _, value in output.items():
                pass  # The value variable will hold the latest state after each node execution
            pprint(
                "--------------------"
            )  # Print a separator for clarity between steps

        # After the workflow completes, print the final relevant context extracted
        print(f"relevant context: {value['relevant_context']}")

    def is_answer_grounded_on_context(self, state):
        """
        Determines if the answer to the question is grounded in the provided context (i.e., not a hallucination).

        Args:
            state (dict): A dictionary containing:
                - "context": The context used to answer the question.
                - "answer": The generated answer to the question.

        Returns:
            str: "hallucination" if the answer is not grounded in the context,
                 "grounded on context" if the answer is supported by the context.
        """

        is_grounded_on_facts_llm = self.llm_model
        # Define the prompt template for fact-checking
        is_grounded_on_facts_prompt_template = """
               You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
               You don't mind if it doesn't make sense, as long as it is grounded in the context.
               Output a JSON containing the answer to the question, and apart from the JSON format don't output any additional text.
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
        print("Checking if the answer is grounded in the facts...")

        # Extract context and answer from the state
        context = state["context"]
        answer = state["answer"]

        # Use the LLM chain to check if the answer is grounded in the context
        result = is_grounded_on_facts_chain.invoke(
            {"context": context, "answer": answer}
        )
        grounded_on_facts = result.grounded_on_facts

        # Return the result based on grounding
        if not grounded_on_facts:
            print("The answer is hallucination.")
            return "hallucination"
        else:
            print("The answer is grounded in the facts.")
            return "grounded on context"

    # -----------------------------------------------
    # Qualitative Answer Workflow Graph Construction
    # -----------------------------------------------

    def answer_workflow_graph_construction(self):
        # Create the workflow graph object
        qualitative_answer_workflow = StateGraph(QualitativeAnswerGraphState)

        # -------------------------
        # Define and Add Graph Nodes
        # -------------------------

        # Node: Answer the question from the provided context using LLM
        qualitative_answer_workflow.add_node(
            "answer_question_from_context", self.answer_question_from_context
        )

        # return qualitative_answer_workflow

        # -------------------------
        # Build the Workflow Edges
        # -------------------------

        # def build_the_workflow_edges(self):
        #     """
        #
        #     :return:
        #     """
        #     qualitative_answer_workflow = self.answer_workflow_graph_construction()
        # Set the entry point of the workflow
        qualitative_answer_workflow.set_entry_point("answer_question_from_context")

        # Conditional Edge: After answering, check if the answer is grounded in the context
        # If hallucination, try answering again; if grounded, end
        qualitative_answer_workflow.add_conditional_edges(
            "answer_question_from_context",
            self.is_answer_grounded_on_context,
            {
                "hallucination": "answer_question_from_context",
                "grounded on context": END,
            },
        )

        # Compile the workflow graph into an executable app
        qualitative_answer_workflow_app = qualitative_answer_workflow.compile()

        # Display the workflow graph as a Mermaid diagram
        display(
            Image(
                qualitative_answer_workflow_app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )

        return qualitative_answer_workflow_app

    # -----------------------------------------------
    # Test the qualitative answer workflow graph
    # -----------------------------------------------

    def test_answer_workflow_graph(self):
        """

        :return:
        """
        # Define the question and context for the test
        qualitative_answer_workflow_app = self.answer_workflow_graph_construction()

        question = "who is harry?"  # The question to answer
        context = "Harry Potter is a cat."  # The context to answer the question from

        # Initialize the state for the workflow
        init_state = {"question": question, "context": context}

        # Stream the outputs from the qualitative_answer_workflow_app
        # This will execute each node in the workflow step by step
        for output in qualitative_answer_workflow_app.stream(init_state):
            # Iterate through the output items (node name, value)
            for _, value in output.items():
                pass  # The value variable holds the latest state after each node execution
            pprint(
                "--------------------"
            )  # Print a separator for clarity between steps

        # After the workflow completes, print the final answer generated by the workflow
        print(f"answer: {value['answer']}")

        # -----------------------------------------------
        # Planning Component for Multi-Step Question Answering
        # -----------------------------------------------

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

        # -----------------------------------------------------------
        # Chain to Refine a Plan into Executable Steps for Retrieval/Answering
        # -----------------------------------------------------------

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
        # -----------------------------------------------------------
        # Example: How to Use the Planner and Refine the Plan
        # -----------------------------------------------------------

        # 1. Define the question to answer
        question = {"question": "how did the main character beat the villain?"}

        # 2. Generate a step-by-step plan to answer the question using the planner chain
        my_plan = planner.invoke(question)
        print("Initial Plan:")
        print(my_plan)

        # 3. Refine the plan so that each step is executable by a retrieval or answer operation
        refined_plan = break_down_plan_chain.invoke([my_plan.steps])
        print("\nRefined Plan:")
        print(refined_plan)
        steps = [
            "Identify the main character and the villain in the story.",
            "Locate the climax or the final confrontation between the main character and the villain.",
            "Analyze the actions taken by the main character during this confrontation.",
            "Determine the specific action or strategy that led to the defeat of the villain.",
            "Summarize the findings to answer how the main character beat the villain.",
        ]
        steps = [
            "Identify the main character and the villain in the story by retrieving relevant information from a "
            "vector store of book chunks, chapter summaries, or book quotes.",
            "Locate the climax or the final confrontation between the main character and the villain by retrieving "
            "relevant information from a vector store of book chunks, chapter summaries, or book quotes.",
            "Analyze the actions taken by the main character during this confrontation by retrieving relevant "
            "information from a vector store of book chunks, chapter summaries, or book quotes.",
            "Determine the specific action or strategy that led to the defeat of the villain by retrieving relevant "
            "information from a vector store of book chunks, chapter summaries, or book quotes.",
            "Summarize the findings to answer how the main character beat the villain by answering a question from a "
            "given context.",
        ]
