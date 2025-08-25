from langchain_core.prompts import PromptTemplate

from backend.config.azure_models import AzureOpenAIModels
from pydantic import Field, BaseModel

def evaluate_faithfulness(question, context, generated_answer):

    """A helper function to run our custom faithfulness evaluation chain."""

    class ResultScore(BaseModel):
        score: float = Field(...,
                             description="The score of the result, ranging from 0 to 1 where 1 is the best possible score.")

    # The prompt template for faithfulness includes several examples (few-shot prompting)
    # to make the instructions to the judge LLM crystal clear.
    faithfulness_prompt = PromptTemplate(
        input_variables=["question", "context", "generated_answer"],
        template="""
        Question: {question}
        Context: {context}
        Generated Answer: {generated_answer}

        Evaluate if the generated answer to the question can be deduced from the context.
        Score of 0 or 1, where 1 is perfectly faithful *AND CAN BE DERIVED FROM THE CONTEXT* and 0 otherwise.
        You don't mind if the answer is correct; all you care about is if the answer can be deduced from the context.

        [... a few examples from the notebook to guide the LLM ...]

        Example:
        Question: What is 2+2?
        Context: 4.
        Generated Answer: 4.
        In this case, the context states '4', but it does not provide information to deduce the answer to 'What is 2+2?', so the score should be 0.
        """
    )

    # Build the faithfulness chain using the same structured LLM.
    faithfulness_chain = (faithfulness_prompt |
                          AzureOpenAIModels().get_azure_model_4().with_structured_output(ResultScore))
    result = faithfulness_chain.invoke({
        "question": question,
        "context": context,
        "generated_answer": generated_answer
    })
    return result.score

# Test the faithfulness chain. The answer is correct, but is it faithful?
question = "what is 3+3?"
context = "6"
generated_answer = "6"
score = evaluate_faithfulness(question, context, generated_answer)

print(f"Faithfulness Score: {score}")
