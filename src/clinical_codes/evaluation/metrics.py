from pydantic import BaseModel

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
