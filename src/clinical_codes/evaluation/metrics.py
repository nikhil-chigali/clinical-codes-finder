from pydantic import BaseModel

from clinical_codes.evaluation.schema import GoldQuery, RunResult
from clinical_codes.schemas import SystemName


class QueryMetrics(BaseModel):
    query_id: str
    query: str
    query_type: str
    system_precision: float
    system_recall: float
    system_f1: float
    recall_at_3: float | None
    must_include_hit_rate: float | None
    iterations: int
    api_calls: int
    latency_s: float
    error: str | None


class QueryTypeMetrics(BaseModel):
    query_type: str
    n: int
    system_selection_f1: float
    top3_recall: float | None
    must_include_hit_rate: float | None
    mean_iterations: float
    mean_api_calls: float


class MetricsSummary(BaseModel):
    n_total: int
    n_errors: int
    system_selection_f1: float
    top3_recall: float
    must_include_hit_rate: float
    mean_iterations: float
    mean_api_calls: float
    by_type: dict[str, QueryTypeMetrics]
    per_query: list[QueryMetrics]


def _system_f1(
    predicted: list[SystemName],
    expected: list[SystemName],
) -> tuple[float, float, float]:
    tp = len(set(predicted) & set(expected))
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _recall_at_k(
    predicted_codes: dict[SystemName, list[str]],
    expected_codes: dict[SystemName, list[str]],
    k: int = 3,
) -> float | None:
    if not expected_codes:
        return None
    total = 0
    hits = 0
    for system, codes in expected_codes.items():
        top_k = predicted_codes.get(system, [])[:k]
        for code in codes:
            total += 1
            if code in top_k:
                hits += 1
    return hits / total if total else None


def _must_include_hit_rate(
    predicted_codes: dict[SystemName, list[str]],
    must_include: list[str],
) -> float | None:
    if not must_include:
        return None
    all_predicted = {code for codes in predicted_codes.values() for code in codes}
    hits = sum(1 for code in must_include if code in all_predicted)
    return hits / len(must_include)


def _aggregate_query_metrics(qms: list[QueryMetrics]) -> dict:
    n = len(qms)
    recall_vals = [qm.recall_at_3 for qm in qms if qm.recall_at_3 is not None]
    mi_vals = [qm.must_include_hit_rate for qm in qms if qm.must_include_hit_rate is not None]
    return {
        "n": n,
        "system_selection_f1": sum(qm.system_f1 for qm in qms) / n if n else 0.0,
        "top3_recall": sum(recall_vals) / len(recall_vals) if recall_vals else None,
        "must_include_hit_rate": sum(mi_vals) / len(mi_vals) if mi_vals else None,
        "mean_iterations": sum(qm.iterations for qm in qms) / n if n else 0.0,
        "mean_api_calls": sum(qm.api_calls for qm in qms) / n if n else 0.0,
    }


def compute_metrics(
    results: list[RunResult],
    gold: list[GoldQuery],
) -> MetricsSummary:
    gold_by_id = {gq.id: gq for gq in gold}
    missing = [r.query_id for r in results if r.query_id not in gold_by_id]
    if missing:
        raise ValueError(f"RunResult query_ids not present in gold set: {missing}")
    per_query: list[QueryMetrics] = []
    for result in results:
        gq = gold_by_id[result.query_id]
        p, r, f1 = _system_f1(result.predicted_systems, gq.expected_systems)
        per_query.append(QueryMetrics(
            query_id=result.query_id,
            query=result.query,
            query_type=result.query_type,
            system_precision=p,
            system_recall=r,
            system_f1=f1,
            recall_at_3=_recall_at_k(result.predicted_codes, gq.expected_codes),
            must_include_hit_rate=_must_include_hit_rate(result.predicted_codes, gq.must_include),
            iterations=result.iterations,
            api_calls=result.api_calls,
            latency_s=result.latency_s,
            error=result.error,
        ))

    by_type_groups: dict[str, list[QueryMetrics]] = {}
    for qm in per_query:
        by_type_groups.setdefault(qm.query_type, []).append(qm)

    by_type = {
        qt: QueryTypeMetrics(query_type=qt, **_aggregate_query_metrics(qms))
        for qt, qms in by_type_groups.items()
    }

    overall = _aggregate_query_metrics(per_query)
    return MetricsSummary(
        n_total=len(results),
        n_errors=sum(1 for r in results if r.error is not None),
        system_selection_f1=overall["system_selection_f1"],
        top3_recall=overall["top3_recall"] if overall["top3_recall"] is not None else 0.0,
        must_include_hit_rate=overall["must_include_hit_rate"] if overall["must_include_hit_rate"] is not None else 0.0,
        mean_iterations=overall["mean_iterations"],
        mean_api_calls=overall["mean_api_calls"],
        by_type=by_type,
        per_query=per_query,
    )
