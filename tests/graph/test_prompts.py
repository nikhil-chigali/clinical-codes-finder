from clinical_codes.graph.state import Attempt, EvaluatorOutput, PlannerOutput
from clinical_codes.schemas import CodeResult, SystemName


def _make_result(system: SystemName, display: str, code: str = "X00") -> CodeResult:
    return CodeResult(system=system, code=code, display=display, score=1.0, raw={})


def _planner_output(
    systems: list[SystemName] | None = None,
    terms: dict[SystemName, str] | None = None,
    rationale: str = "test rationale",
) -> PlannerOutput:
    systems = systems or [SystemName.ICD10CM]
    terms = terms or {SystemName.ICD10CM: "hypertension"}
    return PlannerOutput(selected_systems=systems, search_terms=terms, rationale=rationale)


def _evaluator_output(
    decision: str = "sufficient",
    weak: list[SystemName] | None = None,
    feedback: str = "",
) -> EvaluatorOutput:
    return EvaluatorOutput(decision=decision, weak_systems=weak or [], feedback=feedback)


def _attempt(iteration: int = 1) -> Attempt:
    return Attempt(
        iteration=iteration,
        planner_output=_planner_output(
            systems=[SystemName.ICD10CM, SystemName.LOINC],
            terms={SystemName.ICD10CM: "hypertension", SystemName.LOINC: "hypertension"},
        ),
        raw_results={SystemName.ICD10CM: [_make_result(SystemName.ICD10CM, "Essential hypertension")]},
        evaluator_output=_evaluator_output(
            decision="refine",
            weak=[SystemName.LOINC],
            feedback="LOINC returned no results for this drug query",
        ),
    )


def test_system_catalog_complete() -> None:
    from clinical_codes.graph.prompts import SYSTEM_CATALOG

    assert len(SYSTEM_CATALOG) == len(SystemName), "catalog size must equal number of SystemName members"
    for system in SystemName:
        assert system in SYSTEM_CATALOG, f"{system} missing from SYSTEM_CATALOG"
        assert SYSTEM_CATALOG[system], f"{system} has empty description"


def test_build_planner_first_pass() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from clinical_codes.graph.prompts import build_planner_messages

    messages = build_planner_messages("hypertension", [])

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    human = messages[1].content
    assert "hypertension" in human
    assert "Prior attempt" not in human


def test_build_planner_refinement() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from clinical_codes.graph.prompts import build_planner_messages

    messages = build_planner_messages("hypertension", [_attempt()])

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "clinical coding specialist" in messages[0].content
    human = messages[1].content
    assert "Prior attempt" in human
    assert "hypertension" in human
    assert "LOINC returned no results for this drug query" in human
    assert "LOINC" in human
