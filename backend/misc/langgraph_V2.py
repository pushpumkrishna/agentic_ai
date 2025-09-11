class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]