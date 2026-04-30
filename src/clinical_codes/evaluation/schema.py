from pydantic import BaseModel

from clinical_codes.schemas import SystemName


class GoldQuery(BaseModel):
    id: str
    query: str
    query_type: str
    expected_systems: list[SystemName]
    expected_codes: dict[SystemName, list[str]]
    must_include: list[str]
    must_not_include: list[str]
    notes: str = ""


class GoldSet(BaseModel):
    version: str
    queries: list[GoldQuery]


class RunResult(BaseModel):
    query_id: str
    query: str
    query_type: str
    predicted_systems: list[SystemName]
    predicted_codes: dict[SystemName, list[str]]
    iterations: int
    api_calls: int
    latency_s: float
    error: str | None
    summary: str
