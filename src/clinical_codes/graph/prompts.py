from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from clinical_codes.graph.state import Attempt, PlannerOutput
from clinical_codes.schemas import CodeResult, SystemName

SYSTEM_CATALOG: dict[SystemName, str] = {}


def build_planner_messages(query: str, attempt_history: list[Attempt]) -> list[BaseMessage]:
    raise NotImplementedError


def build_evaluator_messages(
    query: str,
    planner_output: PlannerOutput,
    raw_results: dict[SystemName, list[CodeResult]],
) -> list[BaseMessage]:
    raise NotImplementedError


def build_summarizer_messages(
    query: str,
    consolidated: dict[SystemName, list[CodeResult]],
    rationale: str,
) -> list[BaseMessage]:
    raise NotImplementedError
