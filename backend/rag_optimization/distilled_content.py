import os
from pprint import pprint

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


"""--- Function: Check if Distilled Content is Grounded in the Original Context ---"""

groq_api_key = os.getenv("GROQ_API_KEY")


# Output schema for the LLM's response
class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(
        description="Whether the distilled content is grounded on the original context."
    )
    explanation: str = Field(
        description="An explanation of why the distilled content is or is not grounded on the original context."
    )


class DistilledContent:

    def is_distilled_content_grounded_on_content(self, state):
        """
        Determines if the distilled content is grounded on the original context.

        Args:
            state (dict): A dictionary containing:
                - "relevant_context": The distilled content.
                - "context": The original context.

        Returns:
            str: "grounded on the original context" if grounded, otherwise "not grounded on the original context".
        """
        # Prompt template for the LLM to determine grounding
        is_distilled_content_grounded_on_content_prompt_template = """
        You receive some distilled content: {distilled_content} and the original context: {original_context}.
        You need to determine if the distilled content is grounded on the original context.
        If the distilled content is grounded on the original context, set the grounded field to true.
        If the distilled content is not grounded on the original context, set the grounded field to false.
        {format_instructions}
        """

        pprint("--------------------")
        print("Determining if the distilled content is grounded on the original context...")

        # JSON parser for the output schema
        is_distilled_content_grounded_on_content_json_parser = JsonOutputParser(
            pydantic_object=IsDistilledContentGroundedOnContent
        )

        # Create the prompt object for the LLM
        is_distilled_content_grounded_on_content_prompt = PromptTemplate(
            template=is_distilled_content_grounded_on_content_prompt_template,
            input_variables=["distilled_content", "original_context"],
            partial_variables={
                "format_instructions": is_distilled_content_grounded_on_content_json_parser.get_format_instructions()
            },
        )

        # Initialize the LLM for this task
        is_distilled_content_grounded_on_content_llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            groq_api_key=groq_api_key,
            max_tokens=4000
        )

        # Compose the chain: prompt -> LLM -> output parser
        is_distilled_content_grounded_on_content_chain = (
                is_distilled_content_grounded_on_content_prompt
                | is_distilled_content_grounded_on_content_llm
                | is_distilled_content_grounded_on_content_json_parser
        )

        distilled_content = state["relevant_context"]
        original_context = state["context"]

        input_data = {
            "distilled_content": distilled_content,
            "original_context": original_context
        }

        # Invoke the LLM chain to check grounding
        output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
        grounded = output["grounded"]

        if grounded:
            print("The distilled content is grounded on the original context.")
            return "grounded on the original context"
        else:
            print("The distilled content is not grounded on the original context.")
            return "not grounded on the original context"
