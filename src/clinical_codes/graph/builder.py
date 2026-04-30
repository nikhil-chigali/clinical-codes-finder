from langgraph.graph import END, StateGraph

from clinical_codes.config import MAX_ITERATIONS, NODE_CONSOLIDATOR, NODE_PLANNER
from clinical_codes.graph.nodes import (
    consolidator,
    evaluator,
    executor,
    planner,
    summarizer,
)
from clinical_codes.graph.state import GraphState


def route_after_evaluator(state: GraphState) -> str:
    # `iteration` is post-increment (planner writes iteration + 1 at the start of each pass).
    # At MAX_ITERATIONS the cap fires regardless of the evaluator's decision.
    if state["iteration"] >= MAX_ITERATIONS:
        return NODE_CONSOLIDATOR
    if state["evaluator_output"].decision == "refine":
        return NODE_PLANNER
    return NODE_CONSOLIDATOR


def make_initial_state(query: str) -> GraphState:
    return {
        "query": query,
        "iteration": 0,
        "planner_output": None,
        "raw_results": {},
        "evaluator_output": None,
        "attempt_history": [],
        "consolidated": {},
        "summary": "",
    }


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node(NODE_PLANNER, planner)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator)
    graph.add_node(NODE_CONSOLIDATOR, consolidator)
    graph.add_node("summarizer", summarizer)

    graph.set_entry_point(NODE_PLANNER)

    graph.add_edge(NODE_PLANNER, "executor")
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {NODE_PLANNER: NODE_PLANNER, NODE_CONSOLIDATOR: NODE_CONSOLIDATOR},
    )
    graph.add_edge(NODE_CONSOLIDATOR, "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
