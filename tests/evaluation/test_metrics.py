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
