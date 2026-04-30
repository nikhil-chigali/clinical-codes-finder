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
