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


# ── Executor ──────────────────────────────────────────────────────────────────

async def test_executor_queries_search_terms() -> None:
    from clinical_codes.graph.nodes import executor

    results = [_make_result(SystemName.ICD10CM, "I10", "Essential hypertension")]
    mock_client = _make_mock_client(results)
    mock_clients = {SystemName.ICD10CM: MagicMock(return_value=mock_client)}

    with patch("clinical_codes.graph.nodes.CLIENTS", mock_clients):
        state = _make_state(
            planner_output=_make_planner_output(
                systems=[SystemName.ICD10CM],
                terms={SystemName.ICD10CM: "hypertension"},
            ),
        )
        result = await executor(state)

    assert SystemName.ICD10CM in result["raw_results"]
    assert result["raw_results"][SystemName.ICD10CM] == results


async def test_executor_merges_existing_raw_results() -> None:
    from clinical_codes.graph.nodes import executor

    existing_loinc = [_make_result(SystemName.LOINC, "L001", "Glucose panel")]
    new_icd10 = [_make_result(SystemName.ICD10CM, "I10", "Essential hypertension")]
    mock_client = _make_mock_client(new_icd10)
    mock_clients = {SystemName.ICD10CM: MagicMock(return_value=mock_client)}

    with patch("clinical_codes.graph.nodes.CLIENTS", mock_clients):
        # LOINC was already queried in a previous iteration and had good results;
        # only ICD10CM is in search_terms this pass (LOINC not re-queried)
        state = _make_state(
            raw_results={SystemName.LOINC: existing_loinc},
            planner_output=_make_planner_output(
                systems=[SystemName.ICD10CM, SystemName.LOINC],
                terms={SystemName.ICD10CM: "hypertension"},
            ),
        )
        result = await executor(state)

    assert result["raw_results"][SystemName.LOINC] == existing_loinc  # preserved
    assert result["raw_results"][SystemName.ICD10CM] == new_icd10      # added


async def test_executor_overwrites_previous_results() -> None:
    from clinical_codes.graph.nodes import executor

    old = [_make_result(SystemName.ICD10CM, "I10", "Old result")]
    new = [_make_result(SystemName.ICD10CM, "I11", "Better result")]
    mock_client = _make_mock_client(new)
    mock_clients = {SystemName.ICD10CM: MagicMock(return_value=mock_client)}

    with patch("clinical_codes.graph.nodes.CLIENTS", mock_clients):
        # ICD10CM was weak in iteration 1 and is re-queried with a better term
        state = _make_state(
            raw_results={SystemName.ICD10CM: old},
            planner_output=_make_planner_output(
                systems=[SystemName.ICD10CM],
                terms={SystemName.ICD10CM: "essential hypertension"},
            ),
        )
        result = await executor(state)

    assert result["raw_results"][SystemName.ICD10CM] == new  # replaced, not appended


# ── Evaluator ─────────────────────────────────────────────────────────────────

async def test_evaluator_writes_evaluator_output() -> None:
    from clinical_codes.graph.nodes import evaluator

    expected = _make_evaluator_output()
    with patch("clinical_codes.graph.nodes._evaluator_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=expected)
        state = _make_state(
            iteration=1,
            raw_results={SystemName.ICD10CM: [_make_result(SystemName.ICD10CM, "I10", "Hypertension")]},
        )
        result = await evaluator(state)
    assert result["evaluator_output"] == expected


async def test_evaluator_appends_attempt() -> None:
    from clinical_codes.graph.nodes import evaluator

    eo = _make_evaluator_output()
    po = _make_planner_output()
    raw = {SystemName.ICD10CM: [_make_result(SystemName.ICD10CM, "I10", "Hypertension")]}

    with patch("clinical_codes.graph.nodes._evaluator_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=eo)
        state = _make_state(iteration=1, planner_output=po, raw_results=raw)
        result = await evaluator(state)

    assert len(result["attempt_history"]) == 1
    attempt = result["attempt_history"][0]
    assert attempt.iteration == 1
    assert attempt.planner_output == po
    assert attempt.raw_results == raw
    assert attempt.evaluator_output == eo


# --- Summarizer ---


async def test_summarizer_writes_summary() -> None:
    from clinical_codes.graph.nodes import summarizer
    from unittest.mock import MagicMock

    po = _make_planner_output()
    consolidated = {SystemName.ICD10CM: [_make_result(SystemName.ICD10CM, "I10", "Hypertension")]}

    fake_response = MagicMock()
    fake_response.content = "Hypertension is a condition..."

    with patch("clinical_codes.graph.nodes._summarizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)
        result = await summarizer(
            _make_state(planner_output=po, consolidated=consolidated)
        )

    assert result["summary"] == "Hypertension is a condition..."
