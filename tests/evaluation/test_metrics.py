from clinical_codes.evaluation.metrics import MetricsSummary, QueryMetrics, QueryTypeMetrics
from clinical_codes.schemas import SystemName


def test_query_metrics_instantiates() -> None:
    qm = QueryMetrics(
        query_id="q001",
        query="diabetes",
        query_type="simple",
        system_precision=1.0,
        system_recall=1.0,
        system_f1=1.0,
        recall_at_3=1.0,
        must_include_hit_rate=None,
        iterations=1,
        api_calls=1,
        latency_s=0.5,
        error=None,
    )
    assert qm.query_id == "q001"
    assert qm.must_include_hit_rate is None


def test_metrics_summary_instantiates() -> None:
    summary = MetricsSummary(
        n_total=1,
        n_errors=0,
        system_selection_f1=1.0,
        top3_recall=1.0,
        must_include_hit_rate=1.0,
        mean_iterations=1.0,
        mean_api_calls=1.0,
        by_type={},
        per_query=[],
    )
    assert summary.n_total == 1


# ── _system_f1 ────────────────────────────────────────────────────────────────

from clinical_codes.evaluation.metrics import _system_f1


def test_system_f1_perfect_match() -> None:
    p, r, f1 = _system_f1([SystemName.ICD10CM], [SystemName.ICD10CM])
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_system_f1_miss_query_both_empty() -> None:
    p, r, f1 = _system_f1([], [])
    assert f1 == 1.0


def test_system_f1_hallucinated_systems() -> None:
    # predicted non-empty, expected empty → F1 = 0
    p, r, f1 = _system_f1([SystemName.ICD10CM], [])
    assert f1 == 0.0


def test_system_f1_partial_overlap() -> None:
    p, r, f1 = _system_f1(
        [SystemName.ICD10CM, SystemName.LOINC],
        [SystemName.ICD10CM, SystemName.RXNORM],
    )
    assert p == 0.5
    assert r == 0.5
    assert abs(f1 - 0.5) < 1e-9


# ── _recall_at_k ──────────────────────────────────────────────────────────────

from clinical_codes.evaluation.metrics import _recall_at_k


def test_recall_at_k_hit_in_top3() -> None:
    result = _recall_at_k(
        {SystemName.ICD10CM: ["E11.9", "E10.9", "E11.65"]},
        {SystemName.ICD10CM: ["E11.9"]},
    )
    assert result == 1.0


def test_recall_at_k_miss_beyond_k() -> None:
    # E11.9 at position index 3 (4th result), outside top-3
    result = _recall_at_k(
        {SystemName.ICD10CM: ["E10.9", "E11.65", "J45.9", "E11.9"]},
        {SystemName.ICD10CM: ["E11.9"]},
        k=3,
    )
    assert result == 0.0


def test_recall_at_k_miss_query_returns_none() -> None:
    result = _recall_at_k({}, {})
    assert result is None


def test_recall_at_k_multi_system_partial() -> None:
    # expected: E11.9 (ICD10CM) + 860975, 6809 (RXNORM)
    # predicted: E11.9 hit; 860975 miss (not in list); 6809 hit
    result = _recall_at_k(
        {SystemName.ICD10CM: ["E11.9"], SystemName.RXNORM: ["6809", "862001"]},
        {SystemName.ICD10CM: ["E11.9"], SystemName.RXNORM: ["860975", "6809"]},
        k=3,
    )
    assert result is not None
    assert abs(result - 2 / 3) < 1e-9


# ── _must_include_hit_rate ────────────────────────────────────────────────────

from clinical_codes.evaluation.metrics import _must_include_hit_rate


def test_must_include_hit() -> None:
    result = _must_include_hit_rate(
        {SystemName.ICD10CM: ["E11.9", "E10.9"]},
        ["E11.9"],
    )
    assert result == 1.0


def test_must_include_miss() -> None:
    result = _must_include_hit_rate(
        {SystemName.ICD10CM: ["E10.9"]},
        ["E11.9"],
    )
    assert result == 0.0


def test_must_include_empty_returns_none() -> None:
    result = _must_include_hit_rate(
        {SystemName.ICD10CM: ["E11.9"]},
        [],
    )
    assert result is None


def test_must_include_partial() -> None:
    # E11.9 present; I10 not present → 1/2
    result = _must_include_hit_rate(
        {SystemName.ICD10CM: ["E11.9"]},
        ["E11.9", "I10"],
    )
    assert result == 0.5


# ── compute_metrics ───────────────────────────────────────────────────────────

from clinical_codes.evaluation.metrics import compute_metrics
from clinical_codes.evaluation.schema import GoldQuery, RunResult


def _gold(
    id: str,
    query_type: str,
    expected_systems: list[SystemName],
    expected_codes: dict[SystemName, list[str]],
    must_include: list[str],
) -> GoldQuery:
    return GoldQuery(
        id=id,
        query=f"query_{id}",
        query_type=query_type,
        expected_systems=expected_systems,
        expected_codes=expected_codes,
        must_include=must_include,
        must_not_include=[],
    )


def _result(
    query_id: str,
    query_type: str,
    predicted_systems: list[SystemName],
    predicted_codes: dict[SystemName, list[str]],
    error: str | None = None,
) -> RunResult:
    return RunResult(
        query_id=query_id,
        query=f"query_{query_id}",
        query_type=query_type,
        predicted_systems=predicted_systems,
        predicted_codes=predicted_codes,
        iterations=1,
        api_calls=len(predicted_systems) or 1,
        latency_s=0.5,
        error=error,
        summary="",
    )


def test_compute_metrics_happy_path() -> None:
    gold = [
        _gold("q1", "simple", [SystemName.ICD10CM], {SystemName.ICD10CM: ["E11.9"]}, ["E11.9"]),
        _gold("q2", "simple", [SystemName.ICD10CM], {SystemName.ICD10CM: ["I10"]}, []),
    ]
    results = [
        _result("q1", "simple", [SystemName.ICD10CM], {SystemName.ICD10CM: ["E11.9", "E10.9"]}),
        _result("q2", "simple", [], {}, error="timeout"),
    ]
    summary = compute_metrics(results, gold)

    assert summary.n_total == 2
    assert summary.n_errors == 1
    assert summary.system_selection_f1 < 1.0  # error on q2 drags F1 below 1.0
    assert "simple" in summary.by_type
    assert summary.by_type["simple"].n == 2
    assert len(summary.per_query) == 2


def test_compute_metrics_miss_query_excluded_from_recall() -> None:
    gold = [_gold("q1", "miss", [], {}, [])]
    results = [_result("q1", "miss", [], {})]
    summary = compute_metrics(results, gold)

    assert summary.system_selection_f1 == 1.0
    assert summary.per_query[0].recall_at_3 is None
    assert summary.by_type["miss"].top3_recall is None


def test_compute_metrics_by_type_keys() -> None:
    gold = [
        _gold("q1", "simple", [SystemName.ICD10CM], {SystemName.ICD10CM: ["E11.9"]}, []),
        _gold("q2", "multi_system", [SystemName.ICD10CM, SystemName.LOINC], {}, []),
    ]
    results = [
        _result("q1", "simple", [SystemName.ICD10CM], {SystemName.ICD10CM: ["E11.9"]}),
        _result("q2", "multi_system", [SystemName.ICD10CM, SystemName.LOINC], {}),
    ]
    summary = compute_metrics(results, gold)

    assert set(summary.by_type.keys()) == {"simple", "multi_system"}
