# Builder Design Spec

> **For agentic workers:** implement via `superpowers:subagent-driven-development` or `superpowers:executing-plans`.

**Goal:** Implement `src/clinical_codes/graph/builder.py` — the graph assembly file that wires the five LangGraph nodes into a compilable `StateGraph`, moves routing logic out of `state.py`, and exposes `build_graph()` and `make_initial_state()` for callers.

**Architecture:** Approach A — full cleanup. `route_after_evaluator` moves from `state.py` to `builder.py`. Node-name constants (`NODE_PLANNER`, `NODE_CONSOLIDATOR`) live in `config.py` alongside `MAX_ITERATIONS`. `state.py` becomes a pure schema file.

**Tech stack:** `langgraph` (`StateGraph`, `END`), existing `clinical_codes.graph.nodes`, `clinical_codes.graph.state`, `clinical_codes.config`.

---

## Files changed

| File | Action |
|---|---|
| `src/clinical_codes/config.py` | Add `NODE_PLANNER` and `NODE_CONSOLIDATOR` module-level constants |
| `src/clinical_codes/graph/state.py` | Remove `route_after_evaluator` and the `MAX_ITERATIONS` import |
| `src/clinical_codes/graph/builder.py` | **Create.** `route_after_evaluator`, `make_initial_state`, `build_graph` |
| `tests/graph/test_builder.py` | **Create.** 7 tests: compile check, routing (4 moved), initial state (2) |
| `tests/graph/test_state.py` | Remove 4 routing tests and `_base_state` helper |

---

## `config.py` additions

Add two module-level constants after `MAX_ITERATIONS`:

```python
MAX_ITERATIONS = 2  # cap enforced in route_after_evaluator, not in LLM prompts
NODE_PLANNER = "planner"
NODE_CONSOLIDATOR = "consolidator"
```

These are not on `Settings` (not env-configurable) — same pattern as `MAX_ITERATIONS`.

---

## `state.py` changes

Remove:
- `from clinical_codes.config import MAX_ITERATIONS` (no longer needed)
- The entire `route_after_evaluator` function

After removal `state.py` contains only: module docstring, imports, `PlannerOutput`, `EvaluatorOutput`, `Attempt`, `GraphState`. No logic.

---

## `builder.py`

```python
from langgraph.graph import END, StateGraph

from clinical_codes.config import MAX_ITERATIONS, NODE_CONSOLIDATOR, NODE_PLANNER
from clinical_codes.graph.nodes import (
    consolidator,
    evaluator,
    executor,
    planner,
    summarizer,
)
from clinical_codes.graph.state import GraphState


def route_after_evaluator(state: GraphState) -> str:
    # `iteration` is post-increment (planner writes iteration + 1 at the start of each pass).
    # At MAX_ITERATIONS the cap fires regardless of the evaluator's decision.
    if state["iteration"] >= MAX_ITERATIONS:
        return NODE_CONSOLIDATOR
    if state["evaluator_output"].decision == "refine":
        return NODE_PLANNER
    return NODE_CONSOLIDATOR


def make_initial_state(query: str) -> GraphState:
    return {
        "query": query,
        "iteration": 0,
        "planner_output": None,
        "raw_results": {},
        "evaluator_output": None,
        "attempt_history": [],
        "consolidated": {},
        "summary": "",
    }


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node(NODE_PLANNER, planner)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator)
    graph.add_node(NODE_CONSOLIDATOR, consolidator)
    graph.add_node("summarizer", summarizer)

    graph.set_entry_point(NODE_PLANNER)

    graph.add_edge(NODE_PLANNER, "executor")
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {NODE_PLANNER: NODE_PLANNER, NODE_CONSOLIDATOR: NODE_CONSOLIDATOR},
    )
    graph.add_edge(NODE_CONSOLIDATOR, "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
```

Only `NODE_PLANNER` and `NODE_CONSOLIDATOR` use the named constants — they are the routing targets that `route_after_evaluator` returns. `"executor"`, `"evaluator"`, and `"summarizer"` use inline string literals because they are not routing destinations.

---

## `tests/graph/test_builder.py`

### `make_initial_state` tests (pure — no mocks)

| Test | What it asserts |
|---|---|
| `test_make_initial_state_sets_query` | `state["query"]` equals the argument passed |
| `test_make_initial_state_defaults` | All other fields have their zero values: `iteration=0`, `planner_output=None`, `raw_results={}`, `evaluator_output=None`, `attempt_history=[]`, `consolidated={}`, `summary=""` |

### Routing tests (moved from `test_state.py` — logic unchanged)

These tests import `route_after_evaluator` from `clinical_codes.graph.builder` (not `state`).

| Test | What it asserts |
|---|---|
| `test_route_sufficient_under_cap` | `"sufficient"` decision + iteration < `MAX_ITERATIONS` → `NODE_CONSOLIDATOR` |
| `test_route_refine_under_cap` | `"refine"` decision + iteration < `MAX_ITERATIONS` → `NODE_PLANNER` |
| `test_route_cap_forces_consolidate` | iteration == `MAX_ITERATIONS`, even `"refine"` → `NODE_CONSOLIDATOR` |
| `test_route_one_below_cap_still_refines` | iteration == `MAX_ITERATIONS - 1` + `"refine"` → `NODE_PLANNER` |

### Compile check

| Test | What it asserts |
|---|---|
| `test_graph_compiles` | `build_graph()` returns a non-None object without raising |

### `_base_state` helper (needed by routing tests)

```python
def _base_state(**overrides) -> dict:
    from clinical_codes.graph.state import EvaluatorOutput, PlannerOutput
    state = {
        "query": "diabetes",
        "iteration": 1,
        "planner_output": PlannerOutput(
            selected_systems=[SystemName.ICD10CM],
            search_terms={SystemName.ICD10CM: "diabetes"},
            rationale="ICD-10-CM for condition.",
        ),
        "raw_results": {},
        "evaluator_output": EvaluatorOutput(
            decision="sufficient",
            weak_systems=[],
            feedback="Good.",
        ),
        "attempt_history": [],
        "consolidated": {},
        "summary": "",
    }
    state.update(overrides)
    return state
```

---

## `tests/graph/test_state.py` changes

Remove:
- `_base_state` helper function
- `test_route_sufficient_under_cap`
- `test_route_refine_under_cap`
- `test_route_cap_forces_consolidate`
- `test_route_one_below_cap_still_refines`

The `from clinical_codes.config import MAX_ITERATIONS` import stays — it is still used by `test_max_iterations_value` and `test_max_iterations_type`.

---

## Expected test counts

| File | Before | After |
|---|---|---|
| `test_state.py` | 17 tests | 13 tests (−4 moved) |
| `test_builder.py` | 0 | 7 tests (+7 new) |
| **Total** | **60** | **63** |

All other test files are unchanged.
