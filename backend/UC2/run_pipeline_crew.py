from backend.UC2.src.crew_agent import CreateAgents

input_query = "Create a detailed company profile of Genpact India"

# web_search_tool.run("Important AI innovations of 2025")
CreateAgents().run_pipeline(input_query)
