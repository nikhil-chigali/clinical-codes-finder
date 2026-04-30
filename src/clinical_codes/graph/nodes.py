# Imports marked `noqa: F401` are forward-use scaffolding for Tasks 3–5.
# They are not referenced by the currently implemented nodes (planner, consolidator)
# but will be used by executor (Task 3), evaluator (Task 4), and summarizer (Task 5).
import asyncio  # noqa: F401 — used by executor (Task 3)

from langchain_anthropic import ChatAnthropic

from clinical_codes.config import settings
from clinical_codes.graph.prompts import (
    build_evaluator_messages,  # noqa: F401 — used by evaluator (Task 4)
    build_planner_messages,
    build_summarizer_messages,  # noqa: F401 — used by summarizer (Task 5)
)
from clinical_codes.graph.state import Attempt  # noqa: F401 — used by evaluator (Task 4)
from clinical_codes.graph.state import EvaluatorOutput, GraphState, PlannerOutput
from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools import CLIENTS  # noqa: F401 — used by executor (Task 3)

# api_key passed explicitly so construction succeeds when ANTHROPIC_API_KEY is
# absent at test time — all LLM calls are monkeypatched in tests.
_planner_chain = (
    ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.planner_temperature,
        api_key=settings.anthropic_api_key or "placeholder-for-tests",
    )
    .with_structured_output(PlannerOutput)
)

_evaluator_chain = (
    ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.evaluator_temperature,
        api_key=settings.anthropic_api_key or "placeholder-for-tests",
    )
    .with_structured_output(EvaluatorOutput)
)

_summarizer_llm = ChatAnthropic(
    model=settings.llm_model,
    temperature=settings.summarizer_temperature,
    api_key=settings.anthropic_api_key or "placeholder-for-tests",
)


async def planner(state: GraphState) -> dict:
    messages = build_planner_messages(state["query"], state["attempt_history"])
    output: PlannerOutput = await _planner_chain.ainvoke(messages)
    return {"planner_output": output, "iteration": state["iteration"] + 1}


def consolidator(state: GraphState) -> dict:
    selected = state["planner_output"].selected_systems
    raw = state["raw_results"]

    consolidated: dict[SystemName, list[CodeResult]] = {}
    for system in selected:
        results = raw.get(system, [])
        seen: set[str] = set()
        deduped: list[CodeResult] = []
        for r in results:
            if r.code not in seen:
                seen.add(r.code)
                deduped.append(r)
        deduped.sort(key=lambda r: r.score, reverse=True)
        consolidated[system] = deduped[:settings.display_results]

    return {"consolidated": consolidated}
