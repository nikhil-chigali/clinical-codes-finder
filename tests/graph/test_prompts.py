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


def test_build_evaluator_messages() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from clinical_codes.graph.prompts import build_evaluator_messages

    po = _planner_output()
    raw = {SystemName.ICD10CM: [_make_result(SystemName.ICD10CM, "Essential (primary) hypertension", "I10")]}
    messages = build_evaluator_messages("hypertension", po, raw)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "clinical code quality evaluator" in messages[0].content
    human = messages[1].content
    assert "hypertension" in human
    assert "[I10]" in human
    assert "Essential (primary) hypertension" in human


def test_evaluator_empty_results() -> None:
    from clinical_codes.graph.prompts import build_evaluator_messages

    messages = build_evaluator_messages("hypertension", _planner_output(), {})
    assert "(no results)" in messages[1].content
    assert "ICD10CM" in messages[1].content


def test_evaluator_truncates_to_five() -> None:
    from clinical_codes.graph.prompts import build_evaluator_messages

    po = _planner_output()
    raw = {
        SystemName.ICD10CM: [
            _make_result(SystemName.ICD10CM, f"Result {i}", f"X{i:02d}") for i in range(10)
        ]
    }
    human = build_evaluator_messages("hypertension", po, raw)[1].content
    assert "Result 4" in human      # 5th result is shown
    assert "Result 5" not in human  # 6th result is not shown


def test_build_summarizer_messages() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from clinical_codes.graph.prompts import build_summarizer_messages

    consolidated = [_make_result(SystemName.ICD10CM, "Essential (primary) hypertension", "I10")]
    attempt = _attempt()
    messages = build_summarizer_messages(
        "hypertension", consolidated, "ICD-10-CM covers diagnoses", [attempt]
    )

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "clinical information specialist" in messages[0].content
    human = messages[1].content
    assert "hypertension" in human
    assert "Essential (primary) hypertension" in human
    assert "ICD10CM I10" in human        # new flat format: "1. [ICD10CM I10] ..."
    assert "Reasoning trace" in human
    assert "Iteration" in human


def test_summarizer_truncates_to_five() -> None:
    from clinical_codes.graph.prompts import build_summarizer_messages

    results = [
        _make_result(SystemName.ICD10CM, f"Result {i}", f"X{i:02d}") for i in range(10)
    ]
    human = build_summarizer_messages(
        "test", results, "rationale", [_attempt()]
    )[1].content
    assert "Result 4" in human      # 5th result shown (index 4)
    assert "Result 5" not in human  # 6th result excluded


def test_build_summarizer_cap_hit_note_present() -> None:
    from clinical_codes.graph.prompts import build_summarizer_messages

    # _attempt() has decision="refine" — simulates iteration cap firing
    attempt = _attempt()
    human = build_summarizer_messages("hypertension", [], "rationale", [attempt])[1].content
    assert "Cap-hit" in human
    assert "LOINC returned no results for this drug query" in human


def test_build_summarizer_no_cap_hit_when_sufficient() -> None:
    from clinical_codes.graph.prompts import build_summarizer_messages

    attempt = Attempt(
        iteration=1,
        planner_output=_planner_output(),
        raw_results={},
        evaluator_output=_evaluator_output(decision="sufficient"),
    )
    human = build_summarizer_messages("hypertension", [], "rationale", [attempt])[1].content
    assert "Cap-hit" not in human


def test_build_re_ranker_messages() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from clinical_codes.graph.prompts import build_re_ranker_messages

    pool = [
        CodeResult(system=SystemName.ICD10CM, code="I10", display="Essential hypertension", score=1.0, raw={}),
        CodeResult(system=SystemName.RXNORM, code="854901", display="lisinopril 20 MG Oral Tablet", score=0.9, raw={}),
    ]
    messages = build_re_ranker_messages("hypertension", pool, flat_results=5)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    human = messages[1].content
    assert "hypertension" in human
    assert "[ICD10CM:I10]" in human
    assert "Essential hypertension" in human
    assert "[RXNORM:854901]" in human
    assert "lisinopril 20 MG Oral Tablet" in human
    assert "relevance ranker" in messages[0].content


def test_build_re_ranker_messages_includes_flat_results_count() -> None:
    from clinical_codes.graph.prompts import build_re_ranker_messages

    pool = [
        CodeResult(system=SystemName.ICD10CM, code="I10", display="Essential hypertension", score=1.0, raw={}),
    ]
    messages = build_re_ranker_messages("test", pool, flat_results=3)
    human = messages[1].content
    assert "top 3" in human  # flat_results count must appear in the "Return the top N" instruction
