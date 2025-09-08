from backend.rag_optimization.helper_functions import text_wrap
from backend.rag_optimization.sample_1 import TaskHandlerChainRun
from backend.utils.measure_time import measure_time

"""Run the sophisticated graph function"""


@measure_time
async def execute_plan_and_print_steps(chunks_vector_store,
                                       chapter_summaries_vector_store,
                                       book_quotes_vectorstore,
                                       inputs,
                                       recursion_limit=45):
    """
    Executes the plan-and-execute agent workflow and prints each step.

    Args:
        :param inputs: The initial input state for the plan-and-execute agent.
        :param recursion_limit: Maximum number of steps to prevent infinite loops.
        :param book_quotes_vectorstore:
        :param chapter_summaries_vector_store:
        :param chunks_vector_store:

        :return: tuple: (response, final_state)
                response (str): The final answer or message if not found.
                final_state (dict): The final state after execution.

    """
    # Configuration for the workflow (limits recursion to avoid infinite loops)
    config = {"recursion_limit": recursion_limit}
    agent_state_value = {}
    try:
        plan_and_execute_app = TaskHandlerChainRun(
            chunks_vector_store,
            chapter_summaries_vector_store,
            book_quotes_vectorstore).make_final_graph()
        # Stream the outputs from the plan_and_execute_app workflow
        async for plan_output in plan_and_execute_app.astream(inputs, config=config):
            # Iterate through each step's output and print the current state
            for _, agent_state_value in plan_output.items():
                pass  # agent_state_value holds the latest state after each node execution
                print(f" curr step: {agent_state_value}")
        # Extract the final response from the last state
        response = agent_state_value["response"]

    except RecursionError:
        # Handle the case where the recursion limit is reached
        response = "The answer wasn't found in the data."
    # Save the final state for further inspection or evaluation
    final_state = agent_state_value
    # Print the final answer in a wrapped format for readability
    print(text_wrap(f" the final answer is: {response}"))
    return response, final_state


if __name__ == "__main__":
    """An example we want the model to fail"""
    # -----------------------------------------------------------
    # Example: Run the Plan-and-Execute Agent for a Sample Question
    # -----------------------------------------------------------

    # Define the input question for the agent
    input = {"question": "what did professor lupin teach?"}

    # Execute the plan-and-execute workflow and print each step
    final_answer, final_state_ = execute_plan_and_print_steps(input)
    print((final_answer, final_state_))

    """An example we want the model to succeed"""
    # -----------------------------------------------------------
    # Example: Run the Plan-and-Execute Agent for a Complex Question
    # -----------------------------------------------------------

    # Define the input question for the agent.
    # This question requires reasoning about the professor who helped the villain and what class they teach.
    input = {
        "question": "what is the class that the professor who helped the villain is teaching?"
    }

    # Execute the plan-and-execute workflow and print each step.
    # The function will print the reasoning process and the final answer.
    final_answer, final_state_ = execute_plan_and_print_steps(input)
    print((final_answer, final_state_))
