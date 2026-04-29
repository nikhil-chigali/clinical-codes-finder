import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from clinical_codes.config import MAX_ITERATIONS
from clinical_codes.schemas import CodeResult, SystemName


class PlannerOutput(BaseModel):
    selected_systems: list[SystemName]
    search_terms: dict[SystemName, str]
    rationale: str


class EvaluatorOutput(BaseModel):
    decision: Literal["sufficient", "refine"]
    weak_systems: list[SystemName]
    feedback: str


class Attempt(BaseModel):
    iteration: int
    planner_output: PlannerOutput
    raw_results: dict[SystemName, list[CodeResult]]
    evaluator_output: EvaluatorOutput


class GraphState(TypedDict):
    query: str
    iteration: int
    planner_output: PlannerOutput | None
    raw_results: dict[SystemName, list[CodeResult]]
    evaluator_output: EvaluatorOutput | None
    attempt_history: Annotated[list[Attempt], operator.add]
    consolidated: dict[SystemName, list[CodeResult]]
    summary: str


def route_after_evaluator(state: GraphState) -> str:
    if state["iteration"] >= MAX_ITERATIONS:
        return "consolidator"
    if state["evaluator_output"].decision == "refine":
        return "planner"
    return "consolidator"
