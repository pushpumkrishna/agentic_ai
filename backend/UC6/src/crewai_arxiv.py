from crewai import Agent, Task, Crew
from typing import Type, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import arxiv
import ssl
import datetime
import time
import warnings
warnings.filterwarnings("ignore")
# Disable SSL verification globally (for dev only)
#ssl._create_default_https_context = ssl._create_unverified_context      # ignore


class FetchArxivPapersInput(BaseModel):
    """Input schema for FetchArxivPapersTool."""

    target_date: datetime.date = Field(
        ...,
        description="Target date to fetch papers for.",
    )


class FetchArxivPapersTool(BaseTool):
    name: str = "fetch_arxiv_papers"
    description: str = "Fetches all arXiv papers from selected categories submitted on the target date."
    args_schema: Type[BaseModel] = FetchArxivPapersInput

    def _run(self, target_date: datetime.date) -> List[dict]:
        # List of AI-related categories
        AI_CATEGORIES = ["Hindu"]

        # Define the date range for the target date
        start_date = target_date.strftime("%Y%m%d%H%M")
        end_date = (target_date + datetime.timedelta(days=1)).strftime("%Y%m%d%H%M")

        # Initialize the arXiv client
        client = arxiv.Client(
            page_size=100,  # Fetch 100 results per page
            delay_seconds=3,  # Delay between requests to respect rate limits
        )

        all_papers = []

        for category in AI_CATEGORIES:
            print(f"Fetching papers for category: {category}")

            search_query = (
                f"cat:{category} AND submittedDate:[{start_date} TO {end_date}]"
            )

            search = arxiv.Search(
                query=search_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                max_results=None,  # Fetch all results
            )

            # Collect results for the category
            category_papers = []
            for result in client.results(search):
                category_papers.append(
                    {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "summary": result.summary,
                        "published": result.published,
                        "url": result.entry_id,
                    }
                )

                # Delay between requests to respect rate limits
                time.sleep(3)

            print(f"Fetched {len(category_papers)} papers from {category}")
            all_papers.extend(category_papers)

        return all_papers


arxiv_search_tool = FetchArxivPapersTool()

# Agent 1: Arxiv Researcher
researcher = Agent(
    role="Senior Researcher",
    goal="Find the top 10 papers from the search results from arXiv on {date}."
    "Rank them appropriately.",
    backstory="You are a senior researcher with a deep understanding of all topics in AI and AI research."
    "You are able to identify the best research papers based on the title and abstract.",
    verbose=True,
    llm="azure/gpt-4o-mini",
    tools=[arxiv_search_tool],
)

# Agent 2: Frontend Engineer
frontend_engineer = Agent(
    role="Senior Frontend & AI Engineer",
    goal="Compile the results into a HTML file.",
    backstory="You are a competent frontend engineer writing HTML and CSS with decades of experience."
    "You have also been working with AI for decades and understand it well",
    verbose=True,
    llm="azure/gpt-4o-mini",
)


# Task for Arxiv Researcher
research_task = Task(
    description=" Find the top 10 research papers from the search results from arXiv on {date}.",
    expected_output=(
        "A list of top 10 research papers with the following information in the following format:"
        "- Title"
        "- Authors"
        "- Abstract"
        "- Link to the paper"
    ),
    agent=researcher,
    human_input=True,
)

# Task for Frontend Engineer
reporting_task = Task(
    description="Compile the results into a detailed report in a HTML file.",
    expected_output=(
        "An HTML file with the results in the following format:"
        "Top 10 AI Research Papers published on {date}"
        "Use the tabular format for the following:"
        "- Title (which on clicking opens the paper in a new tab)"
        "- Authors"
        "- Short summary of the abstract (2-4 sentences)"
        "Please do not add '''html''' to the top and bottom of the final file."
    ),
    agent=frontend_engineer,
    context=[research_task],
    output_file="./ai_research_report.html",
    human_input=True,
)


arxiv_research_crew = Crew(
    agents=[researcher, frontend_engineer],
    tasks=[research_task, reporting_task],
    verbose=True,
)

crew_inputs = {"date": "2025-03-13"}

final_result = arxiv_research_crew.kickoff(inputs=crew_inputs)
print(final_result)
