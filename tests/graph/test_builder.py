from clinical_codes.config import MAX_ITERATIONS, NODE_CONSOLIDATOR, NODE_PLANNER
from clinical_codes.schemas import SystemName


# ── make_initial_state ─────────────────────────────────────────────────────────

def test_make_initial_state_sets_query() -> None:
    from clinical_codes.graph.builder import make_initial_state

    state = make_initial_state("hypertension")
    assert state["query"] == "hypertension"


def test_make_initial_state_defaults() -> None:
    from clinical_codes.graph.builder import make_initial_state

    state = make_initial_state("hypertension")
    assert state["iteration"] == 0
    assert state["planner_output"] is None
    assert state["raw_results"] == {}
    assert state["evaluator_output"] is None
    assert state["attempt_history"] == []
    assert state["consolidated"] == {}
    assert state["summary"] == ""


# ── route_after_evaluator ──────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    from clinical_codes.graph.state import EvaluatorOutput, PlannerOutput
    state = {
        "query": "diabetes",
        "iteration": 1,
        "planner_output": PlannerOutput(
            selected_systems=[SystemName.ICD10CM],
            search_terms={SystemName.ICD10CM: "diabetes"},
            rationale="ICD-10-CM for condition.",
        ),
        "raw_results": {},
        "evaluator_output": EvaluatorOutput(
            decision="sufficient",
            weak_systems=[],
            feedback="Good.",
        ),
        "attempt_history": [],
        "consolidated": {},
        "summary": "",
    }
    state.update(overrides)
    return state


def test_route_sufficient_under_cap() -> None:
    from clinical_codes.graph.builder import route_after_evaluator
    from clinical_codes.graph.state import EvaluatorOutput

    state = _base_state(
        iteration=1,
        evaluator_output=EvaluatorOutput(
            decision="sufficient", weak_systems=[], feedback="Good."
        ),
    )
    assert route_after_evaluator(state) == NODE_CONSOLIDATOR


def test_route_refine_under_cap() -> None:
    from clinical_codes.graph.builder import route_after_evaluator
    from clinical_codes.graph.state import EvaluatorOutput

    state = _base_state(
        iteration=1,
        evaluator_output=EvaluatorOutput(
            decision="refine",
            weak_systems=[SystemName.ICD10CM],
            feedback="ICD-10-CM returned no results.",
        ),
    )
    assert route_after_evaluator(state) == NODE_PLANNER


def test_route_cap_forces_consolidate() -> None:
    from clinical_codes.graph.builder import route_after_evaluator
    from clinical_codes.graph.state import EvaluatorOutput

    state = _base_state(
        iteration=MAX_ITERATIONS,
        evaluator_output=EvaluatorOutput(
            decision="refine",
            weak_systems=[SystemName.ICD10CM],
            feedback="Still weak.",
        ),
    )
    assert route_after_evaluator(state) == NODE_CONSOLIDATOR


def test_route_one_below_cap_still_refines() -> None:
    from clinical_codes.graph.builder import route_after_evaluator
    from clinical_codes.graph.state import EvaluatorOutput

    state = _base_state(
        iteration=MAX_ITERATIONS - 1,
        evaluator_output=EvaluatorOutput(
            decision="refine",
            weak_systems=[SystemName.LOINC],
            feedback="LOINC empty.",
        ),
    )
    assert route_after_evaluator(state) == NODE_PLANNER
