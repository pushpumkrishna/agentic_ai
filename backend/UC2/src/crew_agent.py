from crewai import Agent
from crewai import Crew
from crewai.process import Process
from crewai import Task
from backend.UC2.src.custom_rag_tool import rag_tool
from backend.UC2.src.web_tool import web_search_tool


class CreateAgents:
    def __init__(self):
        pass

    @staticmethod
    def create_agents():
        retriever_agent = Agent(
            role="Retriever Agent",
            goal="Retrieve the most relevant information to answer the user's query: {user_query}",
            backstory=(
                "You're a helpful agent. "
                "You're an expert at finding the right information to answer a user's query. "
                "You are great at following instructions and sequentially picking tools for information retrieval. "
                "You have decades of experience doing this."
            ),
            tools=[rag_tool, web_search_tool],
            llm="azure/gpt-4o-mini",
            # llm=self.llm,
            verbose=False,
        )

        customer_support_agent = Agent(
            role="Senior Customer Support Agent",
            goal=(
                "Accurately and concisely answer the user's query: {user_query} using the retrieved information. "
                "If you are unable to answer the query, apologise and tell that you do not have all the information you need to answer the query."
            ),
            backstory=(
                "You are a helpful senior customer support agent. "
                "You have decades of experience in answering user queries grounded to accurate information."
            ),
            llm="azure/gpt-4o-mini",
            # llm=self.llm,
            verbose=False,
        )
        return retriever_agent, customer_support_agent

    def create_tasks(self):
        # Task 1: Retrieval Task

        retriever_agent, customer_support_agent = self.create_agents()
        retrieval_task = Task(
            description=(
                "Retrieve the most relevant information from the given sources to answer the user's query: {user_query}. "
                "ALWAYS use the RAG Tool first. "
                "If you cannot find the required information, ONLY THEN use the Web Search Tool. "
                "DO NOT USE the Web Search Tool if you have sufficient information to accurately answer the user's query."
            ),
            expected_output="The most relevant information from the given sources to answer the user's query in a text format.",
            agent=retriever_agent,
        )

        # Task 2: Customer Support Task
        customer_support_task = Task(
            description=(
                "Using the retrieved information, accurately and concisely answer the user's query: {user_query}."
            ),
            expected_output=(
                "Concise and accurate response based on the retrieved information given the user query: {user_query}. "
                "If you are unable to answer the query, apologise and inform the user that you do not have all the necessary information."
            ),
            agent=customer_support_agent,
            context=[
                retrieval_task
            ],  # This task will use the output from the previous task as its context
        )
        return retrieval_task, customer_support_task

    def create_crew(self):
        retriever_agent, customer_support_agent = self.create_agents()
        retrieval_task, customer_support_task = self.create_tasks()
        customer_support_crew = Crew(
            agents=[retriever_agent, customer_support_agent],
            tasks=[retrieval_task, customer_support_task],
            verbose=False,
            process=Process.sequential,
        )
        return customer_support_crew

    def run_pipeline(self, user_query):
        # Crew inputs
        crew_inputs = {
            "user_query": user_query,
        }

        # Run the crew
        result = self.create_crew().kickoff(inputs=crew_inputs)
        print(result.raw)
