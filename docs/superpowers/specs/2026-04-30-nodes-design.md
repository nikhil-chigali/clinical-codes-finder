# Nodes Design Spec

> **For agentic workers:** implement via `superpowers:subagent-driven-development` or `superpowers:executing-plans`.

**Goal:** Implement `src/clinical_codes/graph/nodes.py` — the five LangGraph node functions that form the planner → executor → evaluator → consolidator → summarizer pipeline.

**Architecture:** Approach A — all five nodes in a single file, module-level LLM chain singletons constructed at import time, `asyncio.gather` for executor fan-out.

**Tech stack:** `langchain-anthropic` (`ChatAnthropic`, `.with_structured_output()`), `asyncio`, existing `clinical_codes.tools.CLIENTS`, `clinical_codes.graph.prompts`, `clinical_codes.graph.state`, `clinical_codes.config.settings`.

---

## Prerequisites

`config.py` already has (added before this spec was written):
- `llm_model: str = "claude-sonnet-4-6"`
- `planner_temperature: float = 0.0`
- `evaluator_temperature: float = 0.0`
- `summarizer_temperature: float = 0.3`

No other file changes are needed before implementing `nodes.py`.

---

## File to create

`src/clinical_codes/graph/nodes.py` — all five node functions plus module-level LLM chains.

`tests/graph/test_nodes.py` — offline tests for all five nodes.

---

## Module-level LLM chains

Three chains constructed once at import time:

```python
from langchain_anthropic import ChatAnthropic
from clinical_codes.config import settings
from clinical_codes.graph.state import EvaluatorOutput, PlannerOutput

_planner_chain = (
    ChatAnthropic(model=settings.llm_model, temperature=settings.planner_temperature)
    .with_structured_output(PlannerOutput)
)

_evaluator_chain = (
    ChatAnthropic(model=settings.llm_model, temperature=settings.evaluator_temperature)
    .with_structured_output(EvaluatorOutput)
)

_summarizer_llm = ChatAnthropic(
    model=settings.llm_model,
    temperature=settings.summarizer_temperature,
)
```

`_planner_chain` and `_evaluator_chain` return typed Pydantic models via `with_structured_output`. `_summarizer_llm` returns an `AIMessage` — its `.content` is the summary string.

---

## Node specifications

### `planner(state: GraphState) -> dict` — async

**Reads:** `state["query"]`, `state["attempt_history"]`

**LLM call:** `build_planner_messages(query, attempt_history)` → `_planner_chain.ainvoke(messages)` → `PlannerOutput`

**Iteration contract (CLAUDE.md §Open obligations):** Must return `{"iteration": state["iteration"] + 1}` on every call. `route_after_evaluator` reads `iteration` post-increment; the cap fires when `iteration >= MAX_ITERATIONS` (2). If the planner does not increment, the cap never fires.

**Returns:**
```python
{"planner_output": output, "iteration": state["iteration"] + 1}
```

---

### `executor(state: GraphState) -> dict` — async

**Reads:** `state["planner_output"].search_terms`, `state["raw_results"]`

**Fan-out:** `asyncio.gather` over all systems in `search_terms`. Each system uses `CLIENTS[system]()` as an async context manager. Error isolation is already handled by `ClinicalTablesClient.search()` — it returns `[]` on total API failure (all retries exhausted). No `return_exceptions=True` needed.

**Inner helper** (defined inside `executor`):
```python
async def _search_one(system: SystemName, term: str) -> tuple[SystemName, list[CodeResult]]:
    async with CLIENTS[system]() as client:
        results = await client.search(term)
    return system, results
```

**Merge logic:** Preserves existing `raw_results` keys not in this iteration's `search_terms`. Writes/overwrites only the keys that were queried this pass.
```python
merged = dict(state["raw_results"])
for system, results in pairs:
    merged[system] = results
return {"raw_results": merged}
```

**Why merge:** On refinement, `search_terms` contains only weak/new systems. Systems that returned strong results in iteration 1 are not re-queried; their results must survive in `raw_results` for the consolidator.

---

### `evaluator(state: GraphState) -> dict` — async

**Reads:** `state["query"]`, `state["planner_output"]`, `state["raw_results"]`, `state["iteration"]`

**LLM call:** `build_evaluator_messages(query, planner_output, raw_results)` → `_evaluator_chain.ainvoke(messages)` → `EvaluatorOutput`

**Attempt assembly:** Snapshot of the current iteration, appended via the `operator.add` reducer (return a single-element list):
```python
attempt = Attempt(
    iteration=state["iteration"],
    planner_output=state["planner_output"],
    raw_results=state["raw_results"],
    evaluator_output=output,
)
return {"evaluator_output": output, "attempt_history": [attempt]}
```

The `operator.add` reducer on `attempt_history` means LangGraph appends `[attempt]` to the existing list — do NOT return the full accumulated list.

---

### `consolidator(state: GraphState) -> dict` — sync (no LLM)

**Reads:** `state["planner_output"].selected_systems`, `state["raw_results"]`

**Filter:** Iterates `selected_systems` only. Systems present in `raw_results` but absent from `selected_systems` (e.g., a system dropped on refinement) are excluded from the consolidated output.

**Per-system processing:**
1. `results = raw_results.get(system, [])` — empty list if system was selected but API failed entirely
2. Deduplicate by `code` (preserve first occurrence, which has the highest score since results are API-ordered)
3. Sort by `score` descending
4. Trim to `settings.display_results` (5)

```python
seen: set[str] = set()
deduped = []
for r in results:
    if r.code not in seen:
        seen.add(r.code)
        deduped.append(r)
deduped.sort(key=lambda r: r.score, reverse=True)
consolidated[system] = deduped[:settings.display_results]
```

**Returns:** `{"consolidated": {system: list[CodeResult], ...}}`

---

### `summarizer(state: GraphState) -> dict` — async

**Reads:** `state["query"]`, `state["consolidated"]`, `state["planner_output"].rationale`

**LLM call:** `build_summarizer_messages(query, consolidated, rationale)` → `_summarizer_llm.ainvoke(messages)` → `AIMessage`

**Returns:**
```python
{"summary": response.content}
```

---

## Testing spec — `tests/graph/test_nodes.py`

All async tests use `pytest-asyncio` with `asyncio_mode = "auto"` (already configured in `pyproject.toml`).

### Consolidator tests (pure — no mocks)

| Test | What it asserts |
|---|---|
| `test_consolidator_filters_to_selected_systems` | A system in `raw_results` but not in `selected_systems` is absent from `consolidated` |
| `test_consolidator_deduplicates_by_code` | Duplicate code entries → only first occurrence kept |
| `test_consolidator_sorts_by_score_descending` | Lower-score result does not appear before higher-score result |
| `test_consolidator_trims_to_display_results` | More than 5 results → only top 5 in output |
| `test_consolidator_empty_results_system` | System in `selected_systems` with no results → empty list in consolidated (not absent) |

### Planner tests (monkeypatch `_planner_chain`)

Monkeypatch target: `clinical_codes.graph.nodes._planner_chain`

| Test | What it asserts |
|---|---|
| `test_planner_increments_iteration` | `state["iteration"] + 1` in returned dict |
| `test_planner_writes_planner_output` | Returned `planner_output` matches mock chain output |
| `test_planner_first_pass_calls_with_empty_history` | Chain invoked with messages built from empty `attempt_history` |
| `test_planner_refinement_calls_with_history` | Chain invoked with messages built from non-empty `attempt_history` |

### Executor tests (monkeypatch `CLIENTS`)

Monkeypatch target: `clinical_codes.graph.nodes.CLIENTS`

| Test | What it asserts |
|---|---|
| `test_executor_queries_search_terms` | One `client.search()` call per system in `search_terms` |
| `test_executor_merges_existing_raw_results` | Keys not in `search_terms` survive in returned `raw_results` |
| `test_executor_overwrites_previous_results` | Re-queried system's results are replaced, not appended |

### Evaluator tests (monkeypatch `_evaluator_chain`)

Monkeypatch target: `clinical_codes.graph.nodes._evaluator_chain`

| Test | What it asserts |
|---|---|
| `test_evaluator_writes_evaluator_output` | Returned `evaluator_output` matches mock chain output |
| `test_evaluator_appends_attempt` | `attempt_history` list contains one `Attempt` with correct `iteration`, `planner_output`, `raw_results`, `evaluator_output` |

### Summarizer tests (monkeypatch `_summarizer_llm`)

Monkeypatch target: `clinical_codes.graph.nodes._summarizer_llm`

| Test | What it asserts |
|---|---|
| `test_summarizer_writes_summary` | `summary` in returned dict equals mock `response.content` |

---

## Monkeypatching pattern

For async chain mocks, use `AsyncMock`:

```python
from unittest.mock import AsyncMock, patch
import pytest
from clinical_codes.graph.state import PlannerOutput
from clinical_codes.schemas import SystemName

@pytest.fixture
def mock_planner_output():
    return PlannerOutput(
        selected_systems=[SystemName.ICD10CM],
        search_terms={SystemName.ICD10CM: "hypertension"},
        rationale="ICD-10-CM for condition.",
    )

async def test_planner_increments_iteration(mock_planner_output):
    with patch("clinical_codes.graph.nodes._planner_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=mock_planner_output)
        from clinical_codes.graph.nodes import planner
        state = _base_state(iteration=0, attempt_history=[])
        result = await planner(state)
    assert result["iteration"] == 1
    assert result["planner_output"] == mock_planner_output
```

For executor CLIENTS mock, return a mock async context manager whose `search()` is an `AsyncMock`.

---

## Imports summary for `nodes.py`

```python
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
```
