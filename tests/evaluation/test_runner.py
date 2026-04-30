import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from clinical_codes.schemas import SystemName


# ── schema ────────────────────────────────────────────────────────────────────

def test_gold_query_parses_from_dict() -> None:
    from clinical_codes.evaluation.schema import GoldQuery

    data = {
        "id": "q001",
        "query": "diabetes",
        "query_type": "simple",
        "expected_systems": ["ICD10CM"],
        "expected_codes": {"ICD10CM": ["E11.9"]},
        "must_include": [],
        "must_not_include": [],
    }
    gq = GoldQuery.model_validate(data)

    assert gq.id == "q001"
    assert gq.expected_systems == [SystemName.ICD10CM]
    assert gq.expected_codes == {SystemName.ICD10CM: ["E11.9"]}


def test_run_result_error_is_nullable() -> None:
    from clinical_codes.evaluation.schema import RunResult

    result = RunResult(
        query_id="q001",
        query="diabetes",
        query_type="simple",
        predicted_systems=[],
        predicted_codes={},
        iterations=0,
        api_calls=0,
        latency_s=0.1,
        error=None,
        summary="",
    )

    assert result.error is None
    assert result.predicted_systems == []


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_final_state() -> dict:
    from clinical_codes.graph.state import Attempt, EvaluatorOutput, PlannerOutput
    from clinical_codes.schemas import CodeResult

    planner_out = PlannerOutput(
        selected_systems=[SystemName.ICD10CM],
        search_terms={SystemName.ICD10CM: "diabetes"},
        rationale="ICD-10-CM for conditions.",
    )
    evaluator_out = EvaluatorOutput(
        decision="sufficient", weak_systems=[], feedback="Good."
    )
    return {
        "query": "diabetes",
        "iteration": 1,
        "planner_output": planner_out,
        "raw_results": {},
        "evaluator_output": evaluator_out,
        "attempt_history": [
            Attempt(
                iteration=1,
                planner_output=planner_out,
                raw_results={},
                evaluator_output=evaluator_out,
            )
        ],
        "consolidated": {
            SystemName.ICD10CM: [
                CodeResult(
                    system=SystemName.ICD10CM,
                    code="E11.9",
                    display="Type 2 diabetes",
                    score=1.0,
                    raw={},
                ),
                CodeResult(
                    system=SystemName.ICD10CM,
                    code="E10.9",
                    display="Type 1 diabetes",
                    score=0.8,
                    raw={},
                ),
            ]
        },
        "summary": "Found ICD-10-CM codes for diabetes.",
    }


# ── run_query ─────────────────────────────────────────────────────────────────

def test_run_query_happy_path() -> None:
    from clinical_codes.evaluation.runner import run_query
    from clinical_codes.evaluation.schema import GoldQuery

    gold_query = GoldQuery(
        id="q001",
        query="diabetes",
        query_type="simple",
        expected_systems=[SystemName.ICD10CM],
        expected_codes={SystemName.ICD10CM: ["E11.9"]},
        must_include=[],
        must_not_include=[],
    )

    with patch("clinical_codes.evaluation.runner._get_graph") as mock_get_graph:
        mock_get_graph.return_value.ainvoke = AsyncMock(return_value=_fake_final_state())
        result = run_query(gold_query)

    assert result.query_id == "q001"
    assert result.predicted_systems == [SystemName.ICD10CM]
    assert result.predicted_codes == {SystemName.ICD10CM: ["E11.9", "E10.9"]}
    assert result.iterations == 1
    assert result.api_calls == 1  # 1 attempt × 1 system selected
    assert result.error is None
    assert result.summary == "Found ICD-10-CM codes for diabetes."
    assert result.latency_s >= 0


def test_run_query_graph_error() -> None:
    from clinical_codes.evaluation.runner import run_query
    from clinical_codes.evaluation.schema import GoldQuery

    gold_query = GoldQuery(
        id="q001",
        query="diabetes",
        query_type="simple",
        expected_systems=[SystemName.ICD10CM],
        expected_codes={},
        must_include=[],
        must_not_include=[],
    )

    with patch("clinical_codes.evaluation.runner._get_graph") as mock_get_graph:
        mock_get_graph.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("api error"))
        result = run_query(gold_query)

    assert result.error == "api error"
    assert result.predicted_systems == []
    assert result.predicted_codes == {}
    assert result.iterations == 0
    assert result.summary == ""
    assert result.latency_s >= 0


# ── run_gold_set ──────────────────────────────────────────────────────────────

def test_run_gold_set_loads_and_loops(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from clinical_codes.evaluation.runner import run_gold_set
    from clinical_codes.evaluation.schema import RunResult

    gold_json = {
        "version": "0.1.0",
        "queries": [
            {
                "id": "q001",
                "query": "diabetes",
                "query_type": "simple",
                "expected_systems": ["ICD10CM"],
                "expected_codes": {"ICD10CM": ["E11.9"]},
                "must_include": [],
                "must_not_include": [],
            },
            {
                "id": "q002",
                "query": "hypertension",
                "query_type": "simple",
                "expected_systems": ["ICD10CM"],
                "expected_codes": {"ICD10CM": ["I10"]},
                "must_include": [],
                "must_not_include": [],
            },
        ],
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold_json))

    fixed_result = RunResult(
        query_id="placeholder",
        query="placeholder",
        query_type="simple",
        predicted_systems=[],
        predicted_codes={},
        iterations=1,
        api_calls=0,
        latency_s=0.1,
        error=None,
        summary="",
    )

    with patch("clinical_codes.evaluation.runner.run_query", return_value=fixed_result) as mock_rq:
        results = run_gold_set(path)

    assert len(results) == 2
    assert mock_rq.call_count == 2
    assert mock_rq.call_args_list[0].args[0].id == "q001"
    assert mock_rq.call_args_list[1].args[0].id == "q002"

    captured = capsys.readouterr()
    assert "q001" in captured.out
    assert "q002" in captured.out


# ── integration ───────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_run_query_real_graph_hypertension() -> None:
    from clinical_codes.evaluation.runner import run_query
    from clinical_codes.evaluation.schema import GoldQuery

    gold_query = GoldQuery(
        id="q002",
        query="hypertension",
        query_type="simple",
        expected_systems=[SystemName.ICD10CM],
        expected_codes={SystemName.ICD10CM: ["I10"]},
        must_include=["I10"],
        must_not_include=[],
    )

    result = run_query(gold_query)

    assert result.error is None
    assert SystemName.ICD10CM in result.predicted_systems
    assert len(result.predicted_codes.get(SystemName.ICD10CM, [])) > 0
    assert result.summary != ""
    assert result.latency_s > 0
