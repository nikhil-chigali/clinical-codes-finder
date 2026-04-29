"""
LangGraph state shape for the clinical-codes-finder pipeline.

State lifecycle:
  - query: set at entry, read-only thereafter
  - iteration: incremented by the planner node at the start of each pass (1-based after first pass)
  - planner_output: None at start; populated by planner, overwritten on each iteration
  - raw_results: empty dict at start; merged by executor (keys from search_terms only)
  - evaluator_output: None at start; overwritten by evaluator on each iteration
  - attempt_history: append-only (operator.add reducer); evaluator appends one Attempt per pass
  - consolidated: empty dict at start; populated by consolidator (single pass at end)
  - summary: empty string at start; populated by summarizer (single pass at end)
"""
import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from clinical_codes.config import MAX_ITERATIONS
from clinical_codes.schemas import CodeResult, SystemName


class PlannerOutput(BaseModel):
    # per-iteration selection — may change on refinement (planner can add or drop systems)
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
    consolidated: dict[SystemName, list[CodeResult]]  # empty dict until consolidator runs
    summary: str                                       # empty string until summarizer runs


def route_after_evaluator(state: GraphState) -> str:
    # `iteration` is incremented by the planner node at the start of each pass.
    # After 2 complete planner→executor→evaluator cycles, iteration==2==MAX_ITERATIONS
    # and the cap forces forward to consolidator regardless of the evaluator's decision.
    # `evaluator_output` is guaranteed non-None here by graph wiring (evaluator always
    # runs before this edge fires).
    # TODO: replace "consolidator"/"planner" with shared constants from builder.py
    if state["iteration"] >= MAX_ITERATIONS:
        return "consolidator"
    if state["evaluator_output"].decision == "refine":
        return "planner"
    return "consolidator"
