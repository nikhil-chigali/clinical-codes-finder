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
    relevant_codes: dict | None = None,
) -> EvaluatorOutput:
    return EvaluatorOutput(
        decision=decision,
        weak_systems=weak or [],
        feedback=feedback,
        relevant_codes=relevant_codes or {},
    )


def _make_state(**overrides) -> dict:
    """Minimal GraphState-compatible dict. Override any key with kwargs."""
    base: dict = {
        "query": "hypertension",
        "iteration": 0,
        "planner_output": _make_planner_output(),
        "raw_results": {},
        "evaluator_output": None,
        "attempt_history": [],
        "consolidated": [],
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


# ── Re_ranker ─────────────────────────────────────────────────────────────────

async def test_re_ranker_empty_pool_returns_empty() -> None:
    from clinical_codes.graph.nodes import re_ranker

    # No raw results for the selected system → empty pool, no LLM call
    state = _make_state(raw_results={})
    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        result = await re_ranker(state)
        mock_chain.ainvoke.assert_not_called()
    assert result["consolidated"] == []


async def test_re_ranker_small_pool_skips_llm() -> None:
    from clinical_codes.graph.nodes import re_ranker

    # Pool has (flat_results - 2) codes ≤ flat_results — returns as-is, no LLM call
    results = [_make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}") for i in range(settings.flat_results - 2)]
    state = _make_state(raw_results={SystemName.ICD10CM: results})

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        result = await re_ranker(state)
        mock_chain.ainvoke.assert_not_called()

    assert result["consolidated"] == results


async def test_re_ranker_calls_llm_for_large_pool() -> None:
    from clinical_codes.graph.nodes import re_ranker
    from clinical_codes.graph.state import RankedCode, ReRankerOutput

    # Pool has (flat_results + 2) codes > flat_results — LLM called, top flat_results returned in order
    results = [_make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}") for i in range(settings.flat_results + 2)]
    state = _make_state(raw_results={SystemName.ICD10CM: results})

    top_codes = [RankedCode(system=SystemName.ICD10CM, code=f"I{i:02d}") for i in range(settings.flat_results)]
    mock_output = ReRankerOutput(ranked_codes=top_codes)

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=mock_output)
        result = await re_ranker(state)

    assert len(result["consolidated"]) == settings.flat_results
    assert [r.code for r in result["consolidated"]] == [f"I{i:02d}" for i in range(settings.flat_results)]


async def test_re_ranker_applies_domain_filter() -> None:
    from clinical_codes.graph.nodes import re_ranker

    # evaluator keeps only I10 — I51 is filtered before pool, so pool size is 1 (≤ 5)
    state = _make_state(
        evaluator_output=_make_evaluator_output(
            relevant_codes={SystemName.ICD10CM: ["I10"]}
        ),
        raw_results={
            SystemName.ICD10CM: [
                _make_result(SystemName.ICD10CM, "I10", "Essential hypertension"),
                _make_result(SystemName.ICD10CM, "I51", "Unspecified heart disease"),
            ]
        },
    )
    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        result = await re_ranker(state)
        mock_chain.ainvoke.assert_not_called()  # pool ≤ flat_results after filter

    codes = [r.code for r in result["consolidated"]]
    assert codes == ["I10"]
    assert "I51" not in codes


async def test_re_ranker_applies_domain_filter_and_calls_llm_for_large_remaining_pool() -> None:
    from clinical_codes.graph.nodes import re_ranker
    from clinical_codes.graph.state import RankedCode, ReRankerOutput

    # Evaluator keeps all but one code. Remaining pool is (flat_results + 1) → LLM still called.
    n = settings.flat_results + 2
    all_codes = [f"I{i:02d}" for i in range(n)]
    results = [_make_result(SystemName.ICD10CM, code, f"Result {i}") for i, code in enumerate(all_codes)]
    # Drop the last code via domain filter — remaining pool = flat_results + 1 > flat_results
    keep_codes = all_codes[:-1]
    state = _make_state(
        evaluator_output=_make_evaluator_output(
            relevant_codes={SystemName.ICD10CM: keep_codes}
        ),
        raw_results={SystemName.ICD10CM: results},
    )

    top_codes = [RankedCode(system=SystemName.ICD10CM, code=c) for c in keep_codes[:settings.flat_results]]
    mock_output = ReRankerOutput(ranked_codes=top_codes)

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=mock_output)
        result = await re_ranker(state)
        mock_chain.ainvoke.assert_called_once()  # LLM was called despite the filter

    codes = [r.code for r in result["consolidated"]]
    assert len(codes) == settings.flat_results
    assert all_codes[-1] not in codes  # filtered code is absent


async def test_re_ranker_drops_invalid_llm_codes() -> None:
    from clinical_codes.graph.nodes import re_ranker
    from clinical_codes.graph.state import RankedCode, ReRankerOutput

    # Pool has (flat_results + 2) codes — LLM path triggered. LLM returns a code not in pool → dropped.
    results = [_make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}") for i in range(settings.flat_results + 2)]
    state = _make_state(raw_results={SystemName.ICD10CM: results})

    ranked = [
        RankedCode(system=SystemName.ICD10CM, code="I00"),    # in pool
        RankedCode(system=SystemName.ICD10CM, code="FAKE99"), # not in pool — must be dropped
        RankedCode(system=SystemName.ICD10CM, code="I01"),    # in pool
    ]

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=ReRankerOutput(ranked_codes=ranked))
        result = await re_ranker(state)

    codes = [r.code for r in result["consolidated"]]
    assert "FAKE99" not in codes
    assert "I00" in codes
    assert "I01" in codes


async def test_re_ranker_deduplicates_llm_output() -> None:
    from clinical_codes.graph.nodes import re_ranker
    from clinical_codes.graph.state import RankedCode, ReRankerOutput

    # Pool has (flat_results + 2) codes — LLM path triggered. LLM returns I00 twice — second occurrence dropped.
    results = [_make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}") for i in range(settings.flat_results + 2)]
    state = _make_state(raw_results={SystemName.ICD10CM: results})

    ranked = [
        RankedCode(system=SystemName.ICD10CM, code="I00"),
        RankedCode(system=SystemName.ICD10CM, code="I00"),  # duplicate
        RankedCode(system=SystemName.ICD10CM, code="I01"),
    ]

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=ReRankerOutput(ranked_codes=ranked))
        result = await re_ranker(state)

    codes = [r.code for r in result["consolidated"]]
    assert codes.count("I00") == 1
    assert "I01" in codes


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


# ── Summarizer ────────────────────────────────────────────────────────────────


async def test_summarizer_writes_summary() -> None:
    from clinical_codes.graph.nodes import summarizer

    po = _make_planner_output()
    consolidated = [_make_result(SystemName.ICD10CM, "I10", "Hypertension")]

    fake_response = MagicMock()
    fake_response.content = "Hypertension is a condition..."

    with patch("clinical_codes.graph.nodes._summarizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)
        with patch("clinical_codes.graph.nodes.build_summarizer_messages") as mock_build:
            mock_build.return_value = []
            result = await summarizer(
                _make_state(planner_output=po, consolidated=consolidated)
            )
            mock_build.assert_called_once_with("hypertension", consolidated, po.rationale, [])

    assert result["summary"] == "Hypertension is a condition..."
