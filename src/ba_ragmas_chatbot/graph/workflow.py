from langgraph.graph import StateGraph, START, END
from ba_ragmas_chatbot.graph.state import AgentState
from ba_ragmas_chatbot.graph.nodes import (
    research_node,
    editor_node,
    writer_node,
    fact_check_node,
    polisher_node,
)


def create_graph():
    """
    Constructs the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", research_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("fact_checker", fact_check_node)
    workflow.add_node("polisher", polisher_node)

    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "editor")
    workflow.add_edge("editor", "writer")
    workflow.add_edge("writer", "fact_checker")
    workflow.add_edge("fact_checker", "polisher")
    workflow.add_edge("polisher", END)

    app = workflow.compile()
    return app
