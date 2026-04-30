from clinical_codes.config import MAX_ITERATIONS, NODE_CONSOLIDATOR, NODE_PLANNER
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
