import asyncio

from langchain_anthropic import ChatAnthropic

from clinical_codes.config import settings
from clinical_codes.graph.prompts import (
    build_evaluator_messages,
    build_planner_messages,
    build_summarizer_messages,
)
from clinical_codes.graph.state import Attempt, EvaluatorOutput, GraphState, PlannerOutput
from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools import CLIENTS

# api_key passed explicitly to all three LLM clients so construction succeeds
# when ANTHROPIC_API_KEY is absent at test time (all LLM calls are mocked).
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


async def executor(state: GraphState) -> dict:
    search_terms = state["planner_output"].search_terms

    async def _search_one(
        system: SystemName, term: str
    ) -> tuple[SystemName, list[CodeResult]]:
        async with CLIENTS[system]() as client:
            results = await client.search(term)
        return system, results

    pairs: list[tuple[SystemName, list[CodeResult]]] = await asyncio.gather(
        *[_search_one(system, term) for system, term in search_terms.items()]
    )
    merged = dict(state["raw_results"])
    for system, results in pairs:
        merged[system] = results
    return {"raw_results": merged}


async def evaluator(state: GraphState) -> dict:
    messages = build_evaluator_messages(
        state["query"],
        state["planner_output"],
        state["raw_results"],
    )
    output: EvaluatorOutput = await _evaluator_chain.ainvoke(messages)
    attempt = Attempt(
        iteration=state["iteration"],
        planner_output=state["planner_output"],
        raw_results=state["raw_results"],
        evaluator_output=output,
    )
    return {"evaluator_output": output, "attempt_history": [attempt]}


def consolidator(state: GraphState) -> dict:
    selected = state["planner_output"].selected_systems
    raw = state["raw_results"]

    consolidated: dict[SystemName, list[CodeResult]] = {}
    for system in selected:
        # Sort first so the highest-score entry for each code is seen first,
        # then dedup by keeping first occurrence. List stays sorted after dedup.
        results = sorted(raw.get(system, []), key=lambda r: r.score, reverse=True)
        seen: set[str] = set()
        deduped: list[CodeResult] = []
        for r in results:
            if r.code not in seen:
                seen.add(r.code)
                deduped.append(r)
        consolidated[system] = deduped[:settings.display_results]

    return {"consolidated": consolidated}


async def summarizer(state: GraphState) -> dict:
    messages = build_summarizer_messages(
        state["query"],
        state["consolidated"],
        state["planner_output"].rationale,
    )
    response = await _summarizer_llm.ainvoke(messages)
    return {"summary": response.content}
