from langchain_community.utilities import GoogleSerperAPIWrapper


def fetch_useful_links(state):
    search = GoogleSerperAPIWrapper()
    destination = state["preferences"].get("destination", "")
    month = state["preferences"].get("month", "")
    query = f"Travel tips and guides for {destination} in {month}"
    try:
        search_results = search.results(query)
        organic_results = search_results.get("organic", [])
        links = [
            {"title": result.get("title", "No title"), "link": result.get("link", "")}
            for result in organic_results[:5]
        ]
        return {"useful_links": links}
    except Exception as e:
        return {"useful_links": [], "warning": f"Failed to fetch links: {str(e)}"}
