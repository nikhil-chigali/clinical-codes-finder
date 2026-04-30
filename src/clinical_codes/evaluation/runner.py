import time

from clinical_codes.evaluation.schema import GoldQuery, RunResult
from clinical_codes.graph.builder import build_graph, make_initial_state

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(gold_query: GoldQuery) -> RunResult:
    start = time.monotonic()
    try:
        state = _get_graph().invoke(make_initial_state(gold_query.query))
        latency_s = time.monotonic() - start
        planner_out = state["planner_output"]
        predicted_systems = planner_out.selected_systems if planner_out else []
        predicted_codes = {
            sys: [r.code for r in results]
            for sys, results in state["consolidated"].items()
        }
        api_calls = sum(
            len(a.planner_output.selected_systems)
            for a in state["attempt_history"]
        )
        return RunResult(
            query_id=gold_query.id,
            query=gold_query.query,
            query_type=gold_query.query_type,
            predicted_systems=predicted_systems,
            predicted_codes=predicted_codes,
            iterations=state["iteration"],
            api_calls=api_calls,
            latency_s=latency_s,
            error=None,
            summary=state["summary"],
        )
    except Exception as exc:
        return RunResult(
            query_id=gold_query.id,
            query=gold_query.query,
            query_type=gold_query.query_type,
            predicted_systems=[],
            predicted_codes={},
            iterations=0,
            api_calls=0,
            latency_s=time.monotonic() - start,
            error=str(exc),
            summary="",
        )
