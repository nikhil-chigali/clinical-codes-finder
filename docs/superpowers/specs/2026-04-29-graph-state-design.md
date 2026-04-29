# GraphState Design

**Date:** 2026-04-29
**Scope:** `src/clinical_codes/graph/state.py` — LangGraph state shape, sub-types, data flow, conditional edge

---

## Context

The LangGraph pipeline has 5 nodes: `planner → executor → evaluator → consolidator → summarizer`, with a bounded refinement loop (max 2 iterations) between `evaluator` and `planner`. This spec defines the shared state that flows through the graph.

---

## Sub-types

Three Pydantic `BaseModel`s, one per LLM-produced output. These live in `state.py` alongside `GraphState`.

### `PlannerOutput`

```python
class PlannerOutput(BaseModel):
    selected_systems: list[SystemName]   # all systems for this query (stable across iterations)
    search_terms: dict[SystemName, str]  # systems to query this iteration only (subset on refinement)
    rationale: str                       # why these systems were selected; read by summarizer
```

`selected_systems` is the planner's overall system selection for the query and does not change between iterations. `search_terms` is the per-iteration subset — on refinement it contains only the weak systems that need re-querying. This distinction drives the executor's merge logic: it iterates over `search_terms`, not `selected_systems`.

### `EvaluatorOutput`

```python
class EvaluatorOutput(BaseModel):
    decision: Literal["sufficient", "refine"]
    weak_systems: list[SystemName]  # empty when decision == "sufficient"
    feedback: str                   # prose diagnosis for the planner on refinement
```

`weak_systems` directly informs the next iteration's `search_terms` — the planner uses it to determine which systems to re-query. `feedback` is the prose explanation of what went wrong and what to try differently.

### `Attempt`

```python
class Attempt(BaseModel):
    iteration: int
    planner_output: PlannerOutput
    raw_results: dict[SystemName, list[CodeResult]]
    evaluator_output: EvaluatorOutput
```

A full snapshot of one pass through `planner → executor → evaluator`. On refinement, the planner reads `attempt_history[-1]` to access prior search terms, results, and the evaluator's diagnosis. No separate `refinement_context` field is needed — the history is the context.

---

## GraphState

```python
import operator
from typing import Annotated

class GraphState(TypedDict):
    query: str                                            # original user input; read-only after entry
    iteration: int                                        # starts at 0; incremented by evaluator
    planner_output: PlannerOutput | None                  # None at graph start
    raw_results: dict[SystemName, list[CodeResult]]       # empty dict at graph start; merged on refinement
    evaluator_output: EvaluatorOutput | None              # None at graph start
    attempt_history: Annotated[list[Attempt], operator.add]  # append-only reducer
    consolidated: dict[SystemName, list[CodeResult]]      # empty dict at graph start
    summary: str                                          # empty string at graph start
```

### Field notes

- **`attempt_history`** is the only field with a LangGraph reducer. `operator.add` appends rather than replaces — nodes return `{"attempt_history": [new_attempt]}` and LangGraph handles the append.
- **`raw_results`** uses default replacement but the executor manually merges: it reads `state["raw_results"]`, updates only the keys it queried this iteration, and returns the full merged dict. Merge logic lives in the executor node, not in a reducer.
- **`planner_output` / `evaluator_output`** are `None` only in the initial state. Downstream nodes are only ever reached after these fields are populated — graph wiring enforces ordering, so no runtime `None` checks are needed inside node bodies.

### Initial state

```python
{
    "query": user_input,
    "iteration": 0,
    "planner_output": None,
    "raw_results": {},
    "evaluator_output": None,
    "attempt_history": [],
    "consolidated": {},
    "summary": "",
}
```

---

## Data flow

| Node | Reads | Writes |
|---|---|---|
| `planner` | `query`, `attempt_history` | `planner_output` |
| `executor` | `planner_output.search_terms`, `raw_results` | `raw_results` (merged) |
| `evaluator` | `planner_output`, `raw_results` | `evaluator_output`, `attempt_history` (append), `iteration` |
| `consolidator` | `raw_results` | `consolidated` |
| `summarizer` | `consolidated`, `planner_output.rationale`, `query` | `summary` |

### Evaluator return (three fields, one atomic update)

```python
return {
    "evaluator_output": evaluator_output,
    "attempt_history": [Attempt(
        iteration=state["iteration"],
        planner_output=state["planner_output"],
        raw_results=state["raw_results"],
        evaluator_output=evaluator_output,
    )],
    "iteration": state["iteration"] + 1,
}
```

`iteration` and `attempt_history` are always updated together in the same evaluator return, so they cannot drift out of sync.

---

## Conditional edge

```python
MAX_ITERATIONS = 2  # cap from scope doc; enforced here, not in the evaluator prompt

def route_after_evaluator(state: GraphState) -> str:
    if state["iteration"] >= MAX_ITERATIONS:
        return "consolidator"
    if state["evaluator_output"].decision == "refine":
        return "planner"
    return "consolidator"
```

`iteration` is read post-increment (evaluator already bumped it), so after 2 full passes (`iteration == 2`) the graph always routes forward regardless of the evaluator's decision. `MAX_ITERATIONS` lives in `config.py` alongside `confidence_threshold`.

---

## Error handling

API failures are isolated in the tools layer — `ClinicalTablesClient.search()` returns `[]` after exhausting retries. Empty lists land in `raw_results` normally. The evaluator treats an empty result list as a weak system and includes it in `weak_systems`. The planner decides on refinement whether to retry with a different term or drop the system.

No `failed_systems` field and no `error` field in `GraphState`. The empty list in `raw_results` is the signal; graph-level exceptions propagate through LangGraph's own error handling.

---

## File layout

Everything in `src/clinical_codes/graph/state.py`:

- `PlannerOutput` (Pydantic `BaseModel`)
- `EvaluatorOutput` (Pydantic `BaseModel`)
- `Attempt` (Pydantic `BaseModel`)
- `GraphState` (TypedDict)
- `MAX_ITERATIONS` imported from `config.py` (lives alongside `confidence_threshold`)

No other files changed by this spec.
