import asyncio

from langchain_anthropic import ChatAnthropic

from clinical_codes.config import settings
from clinical_codes.graph.prompts import (
    build_evaluator_messages,
    build_planner_messages,
    build_re_ranker_messages,
    build_summarizer_messages,
)
from clinical_codes.graph.state import (
    Attempt,
    EvaluatorOutput,
    GraphState,
    PlannerOutput,
    ReRankerOutput,
)
from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools import CLIENTS

# api_key passed explicitly to all LLM clients so construction succeeds
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

_re_ranker_chain = (
    ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.re_ranker_temperature,
        api_key=settings.anthropic_api_key or "placeholder-for-tests",
    )
    .with_structured_output(ReRankerOutput)
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


async def re_ranker(state: GraphState) -> dict:
    ev = state["evaluator_output"]
    relevant = ev.relevant_codes if ev else {}
    raw = state["raw_results"]
    selected = state["planner_output"].selected_systems if state["planner_output"] else []

    # Build pool: apply domain filter per system, then flatten
    pool: list[CodeResult] = []
    for system in selected:
        results = raw.get(system, [])
        # keep=None → no filter; keep=[] → remove all; non-empty → keep only those codes
        keep = relevant.get(system, None)
        if keep is not None:
            keep_set = set(keep)
            results = [r for r in results if r.code in keep_set]
        pool.extend(results)

    if not pool:
        return {"consolidated": []}
    if len(pool) <= settings.flat_results:  # flat_results is both the LLM-call threshold and the output cap
        return {"consolidated": pool}

    messages = build_re_ranker_messages(state["query"], pool, settings.flat_results)
    output: ReRankerOutput = await _re_ranker_chain.ainvoke(messages)

    pool_index: dict[tuple[SystemName, str], CodeResult] = {
        (r.system, r.code): r for r in pool
    }
    ranked: list[CodeResult] = []
    seen: set[tuple[SystemName, str]] = set()
    for rc in output.ranked_codes:
        key = (rc.system, rc.code)
        if key in pool_index and key not in seen:
            seen.add(key)
            ranked.append(pool_index[key])

    # Fall back to pool order if LLM output is entirely invalid (hallucinated codes)
    return {"consolidated": ranked[:settings.flat_results] or pool[:settings.flat_results]}


async def summarizer(state: GraphState) -> dict:
    messages = build_summarizer_messages(
        state["query"],
        state["consolidated"],
        state["planner_output"].rationale,
        state["attempt_history"],
    )
    response = await _summarizer_llm.ainvoke(messages)
    return {"summary": response.content}
