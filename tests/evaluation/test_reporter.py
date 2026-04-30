from clinical_codes.evaluation.metrics import (
    MetricsSummary,
    QueryMetrics,
    QueryTypeMetrics,
)
from clinical_codes.evaluation.reporter import format_markdown


def _make_qm(
    query_id: str,
    query_type: str,
    system_f1: float,
    recall_at_3: float | None,
    must_include_hit_rate: float | None,
    error: str | None = None,
) -> QueryMetrics:
    return QueryMetrics(
        query_id=query_id,
        query=f"query for {query_id}",
        query_type=query_type,
        system_precision=system_f1,
        system_recall=system_f1,
        system_f1=system_f1,
        recall_at_3=recall_at_3,
        must_include_hit_rate=must_include_hit_rate,
        iterations=1,
        api_calls=1,
        latency_s=0.5,
        error=error,
    )


def make_summary() -> MetricsSummary:
    # q001: simple, system_f1=0.5, no error
    # q002: miss,   system_f1=1.0, no error, recall=None, must_include=None
    # q003: simple, system_f1=0.0, error="timeout"
    q001 = _make_qm("q001", "simple", system_f1=0.5, recall_at_3=0.8, must_include_hit_rate=1.0)
    q002 = _make_qm("q002", "miss", system_f1=1.0, recall_at_3=None, must_include_hit_rate=None)
    q003 = _make_qm("q003", "simple", system_f1=0.0, recall_at_3=0.0, must_include_hit_rate=0.0, error="timeout")
    return MetricsSummary(
        n_total=3,
        n_errors=1,
        system_selection_f1=0.5,    # mean(0.5, 1.0, 0.0)
        top3_recall=0.4,             # mean(0.8, 0.0) — q002 excluded (None)
        must_include_hit_rate=0.5,   # mean(1.0, 0.0) — q002 excluded (None)
        mean_iterations=1.0,
        mean_api_calls=1.0,
        by_type={
            "simple": QueryTypeMetrics(
                query_type="simple",
                n=2,
                system_selection_f1=0.25,
                top3_recall=0.4,
                must_include_hit_rate=0.5,
                mean_iterations=1.0,
                mean_api_calls=1.0,
            ),
            "miss": QueryTypeMetrics(
                query_type="miss",
                n=1,
                system_selection_f1=1.0,
                top3_recall=None,
                must_include_hit_rate=None,
                mean_iterations=1.0,
                mean_api_calls=1.0,
            ),
        },
        per_query=[q001, q002, q003],
    )


def make_perfect_summary() -> MetricsSummary:
    q001 = _make_qm("q001", "simple", system_f1=1.0, recall_at_3=1.0, must_include_hit_rate=1.0)
    return MetricsSummary(
        n_total=1,
        n_errors=0,
        system_selection_f1=1.0,
        top3_recall=1.0,
        must_include_hit_rate=1.0,
        mean_iterations=1.0,
        mean_api_calls=1.0,
        by_type={
            "simple": QueryTypeMetrics(
                query_type="simple",
                n=1,
                system_selection_f1=1.0,
                top3_recall=1.0,
                must_include_hit_rate=1.0,
                mean_iterations=1.0,
                mean_api_calls=1.0,
            ),
        },
        per_query=[q001],
    )


def make_all_miss_summary() -> MetricsSummary:
    # All queries are miss-type — top3_recall is None for every QueryTypeMetrics.
    # compute_metrics substitutes 0.0 at the MetricsSummary level.
    q001 = _make_qm("q001", "miss", system_f1=1.0, recall_at_3=None, must_include_hit_rate=None)
    return MetricsSummary(
        n_total=1,
        n_errors=0,
        system_selection_f1=1.0,
        top3_recall=0.0,   # substituted by compute_metrics (all None → 0.0)
        must_include_hit_rate=0.0,
        mean_iterations=1.0,
        mean_api_calls=1.0,
        by_type={
            "miss": QueryTypeMetrics(
                query_type="miss",
                n=1,
                system_selection_f1=1.0,
                top3_recall=None,
                must_include_hit_rate=None,
                mean_iterations=1.0,
                mean_api_calls=1.0,
            ),
        },
        per_query=[q001],
    )


# ── format_markdown — overall ────────────────────────────────────────────────


def test_overall_table_values() -> None:
    summary = make_summary()
    md = format_markdown(summary)
    assert "## Overall" in md
    assert str(summary.n_total) in md
