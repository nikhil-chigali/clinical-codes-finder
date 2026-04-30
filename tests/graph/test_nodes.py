from unittest.mock import AsyncMock, MagicMock, patch

from clinical_codes.config import settings
from clinical_codes.graph.state import Attempt, EvaluatorOutput, PlannerOutput
from clinical_codes.schemas import CodeResult, SystemName


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_result(
    system: SystemName,
    code: str = "X00",
    display: str = "Test Result",
    score: float = 1.0,
) -> CodeResult:
    return CodeResult(system=system, code=code, display=display, score=score, raw={})


def _make_planner_output(
    systems: list[SystemName] | None = None,
    terms: dict[SystemName, str] | None = None,
    rationale: str = "ICD-10-CM for the condition.",
) -> PlannerOutput:
    systems = systems or [SystemName.ICD10CM]
    terms = terms or {SystemName.ICD10CM: "hypertension"}
    return PlannerOutput(selected_systems=systems, search_terms=terms, rationale=rationale)


def _make_evaluator_output(
    decision: str = "sufficient",
    weak: list[SystemName] | None = None,
    feedback: str = "Good.",
) -> EvaluatorOutput:
    return EvaluatorOutput(decision=decision, weak_systems=weak or [], feedback=feedback)


def _make_state(**overrides) -> dict:
    """Minimal GraphState-compatible dict. Override any key with kwargs."""
    base: dict = {
        "query": "hypertension",
        "iteration": 0,
        "planner_output": _make_planner_output(),
        "raw_results": {},
        "evaluator_output": None,
        "attempt_history": [],
        "consolidated": {},
        "summary": "",
    }
    base.update(overrides)
    return base


def _make_mock_client(results: list[CodeResult]) -> MagicMock:
    """Async context manager mock whose search() returns fixed results."""
    client = MagicMock()
    client.search = AsyncMock(return_value=results)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


# ── Consolidator ──────────────────────────────────────────────────────────────

def test_consolidator_filters_to_selected_systems() -> None:
    from clinical_codes.graph.nodes import consolidator

    # LOINC is in raw_results but not in selected_systems (dropped on refinement)
    state = _make_state(
        planner_output=_make_planner_output(
            systems=[SystemName.ICD10CM],
            terms={SystemName.ICD10CM: "hypertension"},
        ),
        raw_results={
            SystemName.ICD10CM: [_make_result(SystemName.ICD10CM, "I10", "Essential hypertension")],
            SystemName.LOINC: [_make_result(SystemName.LOINC, "L001", "Blood pressure panel")],
        },
    )
    result = consolidator(state)
    assert SystemName.ICD10CM in result["consolidated"]
    assert SystemName.LOINC not in result["consolidated"]


def test_consolidator_deduplicates_by_code() -> None:
    from clinical_codes.graph.nodes import consolidator

    state = _make_state(
        raw_results={
            SystemName.ICD10CM: [
                _make_result(SystemName.ICD10CM, "I10", "Essential hypertension", score=1.0),
                _make_result(SystemName.ICD10CM, "I10", "Duplicate entry", score=0.9),
                _make_result(SystemName.ICD10CM, "I11", "Hypertensive heart disease", score=0.8),
            ]
        },
    )
    result = consolidator(state)
    codes = [r.code for r in result["consolidated"][SystemName.ICD10CM]]
    assert codes.count("I10") == 1
    assert "I11" in codes


def test_consolidator_sorts_by_score_descending() -> None:
    from clinical_codes.graph.nodes import consolidator

    state = _make_state(
        raw_results={
            SystemName.ICD10CM: [
                _make_result(SystemName.ICD10CM, "I11", "Heart disease", score=0.5),
                _make_result(SystemName.ICD10CM, "I10", "Hypertension", score=1.0),
            ]
        },
    )
    result = consolidator(state)
    scores = [r.score for r in result["consolidated"][SystemName.ICD10CM]]
    assert scores == sorted(scores, reverse=True)


def test_consolidator_trims_to_display_results() -> None:
    from clinical_codes.graph.nodes import consolidator

    results = [
        _make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}", score=1.0 - i * 0.05)
        for i in range(7)
    ]
    state = _make_state(raw_results={SystemName.ICD10CM: results})
    result = consolidator(state)
    assert len(result["consolidated"][SystemName.ICD10CM]) == settings.display_results  # 5


def test_consolidator_empty_results_system() -> None:
    from clinical_codes.graph.nodes import consolidator

    # ICD10CM selected but not in raw_results (API failed for this system)
    state = _make_state(raw_results={})
    result = consolidator(state)
    assert SystemName.ICD10CM in result["consolidated"]
    assert result["consolidated"][SystemName.ICD10CM] == []


# ── Planner ───────────────────────────────────────────────────────────────────

async def test_planner_increments_iteration() -> None:
    from clinical_codes.graph.nodes import planner

    with patch("clinical_codes.graph.nodes._planner_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=_make_planner_output())
        result = await planner(_make_state(iteration=0))
    assert result["iteration"] == 1


async def test_planner_writes_planner_output() -> None:
    from clinical_codes.graph.nodes import planner

    expected = _make_planner_output()
    with patch("clinical_codes.graph.nodes._planner_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=expected)
        result = await planner(_make_state(iteration=0))
    assert result["planner_output"] == expected


async def test_planner_first_pass_passes_empty_history() -> None:
    from clinical_codes.graph.nodes import planner

    with patch("clinical_codes.graph.nodes._planner_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=_make_planner_output())
        with patch("clinical_codes.graph.nodes.build_planner_messages") as mock_build:
            mock_build.return_value = []
            await planner(_make_state(attempt_history=[]))
    mock_build.assert_called_once_with("hypertension", [])


async def test_planner_refinement_passes_attempt_history() -> None:
    from clinical_codes.graph.nodes import planner

    attempt = Attempt(
        iteration=1,
        planner_output=_make_planner_output(),
        raw_results={},
        evaluator_output=_make_evaluator_output(
            decision="refine",
            weak=[SystemName.ICD10CM],
            feedback="ICD-10-CM returned irrelevant results.",
        ),
    )
    with patch("clinical_codes.graph.nodes._planner_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=_make_planner_output())
        with patch("clinical_codes.graph.nodes.build_planner_messages") as mock_build:
            mock_build.return_value = []
            await planner(_make_state(attempt_history=[attempt]))
    mock_build.assert_called_once_with("hypertension", [attempt])
