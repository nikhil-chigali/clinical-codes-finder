from langgraph.graph import END, StateGraph

from clinical_codes.config import MAX_ITERATIONS, NODE_RE_RANKER, NODE_PLANNER
from clinical_codes.graph.nodes import (
    evaluator,
    executor,
    planner,
    re_ranker,
    summarizer,
)
from clinical_codes.graph.state import GraphState


def route_after_planner(state: GraphState) -> str:
    if not state["planner_output"].selected_systems:
        return NODE_RE_RANKER
    return "executor"


def route_after_evaluator(state: GraphState) -> str:
    # `iteration` is post-increment (planner writes iteration + 1 at the start of each pass).
    # At MAX_ITERATIONS the cap fires regardless of the evaluator's decision.
    if state["iteration"] >= MAX_ITERATIONS:
        return NODE_RE_RANKER
    if state["evaluator_output"].decision == "refine":
        return NODE_PLANNER
    return NODE_RE_RANKER


def make_initial_state(query: str) -> GraphState:
    return {
        "query": query,
        "iteration": 0,
        "planner_output": None,
        "raw_results": {},
        "evaluator_output": None,
        "attempt_history": [],
        "consolidated": [],
        "summary": "",
    }


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node(NODE_PLANNER, planner)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator)
    graph.add_node(NODE_RE_RANKER, re_ranker)
    graph.add_node("summarizer", summarizer)

    graph.set_entry_point(NODE_PLANNER)

    graph.add_conditional_edges(
        NODE_PLANNER,
        route_after_planner,
        {"executor": "executor", NODE_RE_RANKER: NODE_RE_RANKER},
    )
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {NODE_PLANNER: NODE_PLANNER, NODE_RE_RANKER: NODE_RE_RANKER},
    )
    graph.add_edge(NODE_RE_RANKER, "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
