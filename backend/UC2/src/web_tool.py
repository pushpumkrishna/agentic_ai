# Create custom Web search tool
import os
import requests
from crewai.tools import tool
import urllib3
from backend.config.logging_lib import logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Tool to search the web for relevant information."""

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": os.environ.get("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    payload = {"q": query}

    response = requests.post(url, headers=headers, json=payload, verify=False)

    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}")

    data = response.json()

    # Get value associated with key "organic" else return []
    search_results = data.get("organic", [])

    if not search_results:
        return "No search results found."

    context = ""

    for result in search_results:
        title = result.get("title", "")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        context += f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n\n"
        logger.info(f"Web Search Results:\n{context}")
