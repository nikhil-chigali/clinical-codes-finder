from clinical_codes.config import settings
from clinical_codes.graph.state import GraphState
from clinical_codes.schemas import CodeResult, SystemName


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
