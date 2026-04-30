import operator
from typing import get_args, get_type_hints

import pytest
from pydantic import ValidationError

from clinical_codes.config import MAX_ITERATIONS
from clinical_codes.schemas import SystemName


# ── MAX_ITERATIONS ────────────────────────────────────────────────────────────

def test_max_iterations_value() -> None:
    assert MAX_ITERATIONS == 2


def test_max_iterations_type() -> None:
    assert isinstance(MAX_ITERATIONS, int)


# ── PlannerOutput ─────────────────────────────────────────────────────────────

def test_planner_output_valid() -> None:
    from clinical_codes.graph.state import PlannerOutput
    po = PlannerOutput(
        selected_systems=[SystemName.ICD10CM, SystemName.LOINC],
        search_terms={SystemName.ICD10CM: "diabetes", SystemName.LOINC: "glucose"},
        rationale="ICD-10-CM for condition, LOINC for lab test.",
    )
    assert po.selected_systems == [SystemName.ICD10CM, SystemName.LOINC]
    assert po.search_terms[SystemName.ICD10CM] == "diabetes"
    assert po.rationale == "ICD-10-CM for condition, LOINC for lab test."


def test_planner_output_search_terms_can_be_subset() -> None:
    from clinical_codes.graph.state import PlannerOutput
    # On refinement, search_terms contains only weak systems — not all selected_systems
    po = PlannerOutput(
        selected_systems=[SystemName.ICD10CM, SystemName.LOINC],
        search_terms={SystemName.LOINC: "glucose panel"},
        rationale="Re-querying LOINC only.",
    )
    assert SystemName.ICD10CM not in po.search_terms
    assert SystemName.LOINC in po.search_terms


def test_planner_output_missing_rationale_raises() -> None:
    from clinical_codes.graph.state import PlannerOutput
    with pytest.raises(ValidationError):
        PlannerOutput(
            selected_systems=[SystemName.ICD10CM],
            search_terms={SystemName.ICD10CM: "diabetes"},
        )


# ── EvaluatorOutput ───────────────────────────────────────────────────────────

def test_evaluator_output_sufficient() -> None:
    from clinical_codes.graph.state import EvaluatorOutput
    eo = EvaluatorOutput(
        decision="sufficient",
        weak_systems=[],
        feedback="All systems returned strong results.",
    )
    assert eo.decision == "sufficient"
    assert eo.weak_systems == []


def test_evaluator_output_refine() -> None:
    from clinical_codes.graph.state import EvaluatorOutput
    eo = EvaluatorOutput(
        decision="refine",
        weak_systems=[SystemName.LOINC],
        feedback="LOINC returned no results for 'glucose test'.",
    )
    assert eo.decision == "refine"
    assert SystemName.LOINC in eo.weak_systems


def test_evaluator_output_invalid_decision_raises() -> None:
    from clinical_codes.graph.state import EvaluatorOutput
    with pytest.raises(ValidationError):
        EvaluatorOutput(decision="maybe", weak_systems=[], feedback="Uncertain.")


# ── Attempt ───────────────────────────────────────────────────────────────────

def _make_planner_output():
    from clinical_codes.graph.state import PlannerOutput
    return PlannerOutput(
        selected_systems=[SystemName.ICD10CM],
        search_terms={SystemName.ICD10CM: "diabetes"},
        rationale="ICD-10-CM for the condition.",
    )


def _make_evaluator_output():
    from clinical_codes.graph.state import EvaluatorOutput
    return EvaluatorOutput(
        decision="sufficient",
        weak_systems=[],
        feedback="Results are strong.",
    )


def test_attempt_valid() -> None:
    from clinical_codes.graph.state import Attempt
    attempt = Attempt(
        iteration=0,
        planner_output=_make_planner_output(),
        raw_results={SystemName.ICD10CM: []},
        evaluator_output=_make_evaluator_output(),
    )
    assert attempt.iteration == 0
    assert attempt.planner_output.selected_systems == [SystemName.ICD10CM]
    assert attempt.raw_results == {SystemName.ICD10CM: []}
    assert attempt.evaluator_output.decision == "sufficient"


def test_attempt_missing_evaluator_output_raises() -> None:
    from clinical_codes.graph.state import Attempt
    with pytest.raises(ValidationError):
        Attempt(
            iteration=0,
            planner_output=_make_planner_output(),
            raw_results={},
            # evaluator_output omitted — required field
        )


# ── GraphState ────────────────────────────────────────────────────────────────

def test_graph_state_has_required_keys() -> None:
    from clinical_codes.graph.state import GraphState
    required = {
        "query", "iteration", "planner_output", "raw_results",
        "evaluator_output", "attempt_history", "consolidated", "summary",
    }
    assert required == set(get_type_hints(GraphState).keys())


def test_attempt_history_has_operator_add_reducer() -> None:
    from clinical_codes.graph.state import GraphState
    hints = get_type_hints(GraphState, include_extras=True)
    # get_args(Annotated[list[Attempt], operator.add]) → (list[Attempt], operator.add)
    args = get_args(hints["attempt_history"])
    assert operator.add in args


def test_initial_state_shape() -> None:
    from clinical_codes.graph.state import GraphState
    state: GraphState = {
        "query": "diabetes",
        "iteration": 0,
        "planner_output": None,
        "raw_results": {},
        "evaluator_output": None,
        "attempt_history": [],
        "consolidated": {},
        "summary": "",
    }
    assert state["query"] == "diabetes"
    assert state["iteration"] == 0
    assert state["planner_output"] is None
    assert state["attempt_history"] == []
    assert state["summary"] == ""
