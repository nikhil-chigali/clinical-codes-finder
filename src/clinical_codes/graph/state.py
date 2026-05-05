"""
LangGraph state shape for the clinical-codes-finder pipeline.

State lifecycle:
  - query: set at entry, read-only thereafter
  - iteration: incremented by the planner node at the start of each pass (1-based after first pass)
  - planner_output: None at start; populated by planner, overwritten on each iteration
  - raw_results: empty dict at start; merged by executor (keys from search_terms only)
  - evaluator_output: None at start; overwritten by evaluator on each iteration
  - attempt_history: append-only (operator.add reducer); evaluator appends one Attempt per pass
  - consolidated: empty list at start; populated by re_ranker (flat, ordered by query relevance)
  - summary: empty string at start; populated by summarizer (single pass at end)
"""
import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

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
    relevant_codes: dict[SystemName, list[str]] = {}
    # codes to keep per system; populated on every pass, empty on refine when no filtering needed


class RankedCode(BaseModel):
    system: SystemName
    code: str


class ReRankerOutput(BaseModel):
    ranked_codes: list[RankedCode]  # ordered most → least relevant, max flat_results entries


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
    consolidated: list[CodeResult]  # flat, ordered by query relevance; empty list until re_ranker runs
    summary: str                    # empty string until summarizer runs
