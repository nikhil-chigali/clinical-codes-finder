import json
import time
from pathlib import Path

from clinical_codes.evaluation.schema import GoldQuery, GoldSet, RunResult
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
        predicted_systems = list(state["consolidated"].keys())
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


def run_gold_set(path: Path | str) -> list[RunResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    gold_set = GoldSet.model_validate(data)
    results = []
    for gq in gold_set.queries:
        result = run_query(gq)
        status = f"ERROR: {result.error}" if result.error else f"{result.latency_s:.1f}s"
        print(f"  {gq.id} ({gq.query_type}): {status}")
        results.append(result)
    return results
