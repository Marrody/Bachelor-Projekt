from langgraph.graph import StateGraph, START, END
from ba_ragmas_chatbot.graph.state import AgentState
from ba_ragmas_chatbot.graph.nodes import (
    research_node,
    editor_node,
    writer_node,
    fact_check_node,
    polisher_node,
)


def route_after_fact_check(state: AgentState) -> str:
    """
    Decides if text goes back to writer or forwards it to polisher.
    """

    if state.get("revision_count", 0) >= 2:
        return "polisher"

    critique = state.get("critique", "").strip().upper()

    if critique == "PASS" or critique.startswith("PASS") or critique == "":
        return "polisher"
    else:
        return "writer"


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

    workflow.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        {"writer": "writer", "polisher": "polisher"},
    )

    workflow.add_edge("polisher", END)

    app = workflow.compile()
    return app
