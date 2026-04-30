import json
from pathlib import Path
from unittest.mock import patch

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
