from pydantic import BaseModel


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
