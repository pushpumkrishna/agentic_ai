from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from backend.config.azure_models import AzureOpenAIModels
from grouse import EvaluationSample, GroundedQAEvaluator
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_correctness,
)


class ResultScore(BaseModel):
    score: float = Field(
        None,
        description="The score of the result, ranging from 0 to 1 where 1 is the best possible score.",
    )

    def run_process_evaluation(self):
        # This prompt template clearly instructs the LLM on how to score the answer's correctness.
        correctness_prompt = PromptTemplate(
            input_variables=["question", "ground_truth", "generated_answer"],
            template="""
            Question: {question}
            Ground Truth: {ground_truth}
            Generated Answer: {generated_answer}
        
            Evaluate the correctness of the generated answer compared to the ground truth.
            Score from 0 to 1, where 1 is perfectly correct and 0 is completely incorrect.
        
            Score:
            """,
        )

        # We build the evaluation chain by piping the prompt to the LLM with structured output.
        correctness_chain = (
            correctness_prompt
            | AzureOpenAIModels()
            .get_azure_model_4()
            .with_structured_output(ResultScore)
        )

        def evaluate_correctness(_question, _ground_truth, _generated_answer):
            """A helper function to run our custom correctness evaluation chain."""
            result = correctness_chain.invoke(
                {
                    "question": _question,
                    "ground_truth": _ground_truth,
                    "generated_answer": _generated_answer,
                }
            )
            return result.score

        # Test the correctness chain with a partially correct answer.
        question = "What is the capital of France and Spain?"
        ground_truth = "Paris and Madrid"
        generated_answer = "Paris"
        score = evaluate_correctness(question, ground_truth, generated_answer)

        print(f"Correctness Score: {score}")

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
            """,
        )

        # Build the faithfulness chain using the same structured LLM.
        faithfulness_chain = faithfulness_prompt | self.llm.with_structured_output(
            ResultScore
        )

        def evaluate_faithfulness(question, context, generated_answer):
            """A helper function to run our custom faithfulness evaluation chain."""
            result = faithfulness_chain.invoke(
                {
                    "question": question,
                    "context": context,
                    "generated_answer": generated_answer,
                }
            )
            return result.score

        # Test the faithfulness chain. The answer is correct, but is it faithful?
        question = "what is 3+3?"
        context = "6"
        generated_answer = "6"
        score = evaluate_faithfulness(question, context, generated_answer)

        print(f"Faithfulness Score: {score}")

        def evaluate_faithfulness(question, context, generated_answer):
            """A helper function to run our custom faithfulness evaluation chain."""
            result = faithfulness_chain.invoke(
                {
                    "question": question,
                    "context": context,
                    "generated_answer": generated_answer,
                }
            )
            return result.score

        # Test the faithfulness chain. The answer is correct, but is it faithful?
        question = "what is 3+3?"
        context = "6"
        generated_answer = "6"
        score = evaluate_faithfulness(question, context, generated_answer)

        # You will need to install deepeval: pip install deepeval

        # Create test cases
        test_case_correctness = LLMTestCase(
            input="What is the capital of Spain?",
            expected_output="Madrid is the capital of Spain.",
            actual_output="MadriD.",
        )

        test_case_faithfulness = LLMTestCase(
            input="what is 3+3?", actual_output="6", retrieval_context=["6"]
        )

        # The evaluate() function runs all test cases against all specified metrics
        evaluation_results = evaluate(
            test_cases=[test_case_correctness, test_case_faithfulness],
            metrics=[GEval(name="Correctness", model="gpt-4o"), FaithfulnessMetric()],
        )

        print(evaluation_results)

    print(f"Faithfulness Score: {score}")

    def process_(self):
        # You will need to install grouse: pip install grouse-eval

        evaluator = GroundedQAEvaluator()
        unfaithful_sample = EvaluationSample(
            input="Where is the Eiffel Tower located?",
            actual_output="The Eiffel Tower is located at Rue Rabelais in Paris.",
            references=[
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France",
                "Gustave Eiffel died in his appartment at Rue Rabelais in Paris.",
            ],
        )

        result = evaluator.evaluate(eval_samples=[unfaithful_sample]).evaluations[0]
        print(f"Grouse Faithfulness Score (0 or 1): {result.faithfulness.faithfulness}")

        # 1. Prepare the evaluation data
        questions = [
            "What is the name of the three-headed dog guarding the Sorcerer's Stone?",
            "Who gave Harry Potter his first broomstick?",
            "Which house did the Sorting Hat initially consider for Harry?",
        ]

        # These would be the answers generated by our RAG pipeline
        generated_answers = [
            "The three-headed dog is named Fluffy.",
            "Professor McGonagall gave Harry his first broomstick, a Nimbus 2000.",
            "The Sorting Hat strongly considered putting Harry in Slytherin.",
        ]

        # The ground truth, or "perfect" answers
        ground_truth_answers = [
            "Fluffy",
            "Professor McGonagall",
            "Slytherin",
        ]

        # The context retrieved by our RAG system for each question
        retrieved_documents = [
            [
                "A massive, three-headed dog was guarding a trapdoor. Hagrid mentioned its name was Fluffy."
            ],
            [
                "First years are not allowed brooms, but Professor McGonagall, head of Gryffindor, made an exception for Harry."
            ],
            [
                "The Sorting Hat muttered in Harry's ear, 'You could be great, you know, it's all here in your head, and Slytherin will help you on the way to greatness...'"
            ],
        ]

        # You will need to install ragas and datasets: pip install ragas datasets

        # 2. Structure the data into a Hugging Face Dataset object
        data_samples = {
            "question": questions,
            "answer": generated_answers,
            "contexts": retrieved_documents,
            "ground_truth": ground_truth_answers,
        }

        dataset = Dataset.from_dict(data_samples)
        # 3. Define the metrics we want to use for evaluation
        metrics = [
            faithfulness,  # How factually consistent is the answer with the context? (Prevents hallucination)
            answer_relevancy,  # How relevant is the answer to the question?
            context_recall,  # Did we retrieve all the necessary context to answer the question?
            answer_correctness,  # How accurate is the answer compared to the ground truth?
        ]

        # 4. Run the evaluation
        result = evaluate(dataset=dataset, metrics=metrics)

        # 5. Display the results in a clean table format
        results_df = result.to_pandas()
        print(results_df)
