from typing import TypedDict, Any
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, StateGraph
from IPython.display import display, Image
from backend.rag_optimization.research import RewriteQuestion


# -----------------------------------------------
# Qualitative Retrieval Answer Graph Construction
# -----------------------------------------------

# Define the state for the workflow graph
class QualitativeRetrievalAnswerGraphState(TypedDict):
    question: str
    context: str
    answer: str


class GraphRetrieval(RewriteQuestion):

    def __init__(self, **data: Any):
        super().__init__(**data)

    def graph_pipeline(self):

        # -------------------------
        # Define and Add Graph Nodes
        # -------------------------
        # Each node represents a function in the pipeline

        # Create the workflow graph object
        qualitative_retrieval_answer_workflow = StateGraph(QualitativeRetievalAnswerGraphState)

        # Node: Retrieve context for the question from vector stores
        qualitative_retrieval_answer_workflow.add_node(
            "retrieve_context_per_question", self.retrieve_context_per_question
        )

        # Node: Use LLM to keep only relevant content from the retrieved context
        qualitative_retrieval_answer_workflow.add_node(
            "keep_only_relevant_content", self.keep_only_relevant_content
        )

        # Node: Rewrite the question for better retrieval if needed
        qualitative_retrieval_answer_workflow.add_node(
            "rewrite_question", self.rewrite_question
        )

        # Node: Answer the question from the relevant context using LLM
        qualitative_retrieval_answer_workflow.add_node(
            "answer_question_from_context", self.answer_question_from_context
        )

        # -------------------------
        # Build the Workflow Edges
        # -------------------------

        # Set the entry point of the workflow
        qualitative_retrieval_answer_workflow.set_entry_point("retrieve_context_per_question")

        # Edge: After retrieving context, filter to keep only relevant content
        qualitative_retrieval_answer_workflow.add_edge(
            "retrieve_context_per_question", "keep_only_relevant_content"
        )

        # Conditional Edge: After filtering, check if content is relevant
        # If relevant, answer the question; if not, rewrite the question
        qualitative_retrieval_answer_workflow.add_conditional_edges(
            "keep_only_relevant_content",
            self.is_relevant_content,
            {
                "relevant": "answer_question_from_context",
                "not relevant": "rewrite_question"
            },
        )

        # Edge: After rewriting the question, retrieve context again
        qualitative_retrieval_answer_workflow.add_edge(
            "rewrite_question", "retrieve_context_per_question"
        )

        # Conditional Edge: After answering, grade the answer
        # If hallucination, try answering again; if not useful, rewrite question; if useful, end
        qualitative_retrieval_answer_workflow.add_conditional_edges(
            "answer_question_from_context",
            self.grade_generation_v_documents_and_question,
            {
                "hallucination": "answer_question_from_context",
                "not_useful": "rewrite_question",
                "useful": END
            },
        )

        # Compile the workflow graph into an executable app
        qualitative_retrieval_answer_retrival_app = qualitative_retrieval_answer_workflow.compile()

        # Display the workflow graph as a Mermaid diagram
        display(
            Image(
                qualitative_retrieval_answer_retrival_app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )
