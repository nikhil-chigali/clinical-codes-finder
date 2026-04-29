# GraphState Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/clinical_codes/graph/state.py` — three Pydantic sub-types (`PlannerOutput`, `EvaluatorOutput`, `Attempt`), a `GraphState` TypedDict with a LangGraph append reducer on `attempt_history`, and the `route_after_evaluator` conditional edge function.

**Architecture:** All types live in one file (`state.py`). `MAX_ITERATIONS = 2` is a module-level constant in `config.py`. Tests use local imports inside each function so the file can always be collected by pytest even while types are being added one by one. No LangGraph runtime is exercised in unit tests — routing is a pure function over plain dicts.

**Tech Stack:** Python 3.12, Pydantic v2, `typing.TypedDict`, `typing.Annotated`, `operator.add`, pytest, uv

---

### Task 1: Setup — dependency, constant, test directory, skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/clinical_codes/config.py`
- Create: `tests/graph/__init__.py`
- Create: `src/clinical_codes/graph/state.py` (imports only — no classes yet)
- Create: `tests/graph/test_state.py` (MAX_ITERATIONS tests only)

- [ ] **Step 1: Add `langgraph` to runtime deps**

Edit `pyproject.toml`, add `"langgraph>=0.2"` to the `dependencies` list:

```toml
dependencies = [
    "httpx>=0.28",
    "langgraph>=0.2",
    "pydantic>=2.13.3",
    "pydantic-settings>=2.14.0",
    "tenacity>=8.0",
]
```

- [ ] **Step 2: Sync deps**

```bash
uv sync
```

Expected: resolves and installs `langgraph` and its transitive deps without errors.

- [ ] **Step 3: Add `MAX_ITERATIONS` to config.py**

Append one line after `settings = Settings()` in `src/clinical_codes/config.py`:

```python
MAX_ITERATIONS = 2
```

Final tail of the file:

```python
settings = Settings()

MAX_ITERATIONS = 2
```

- [ ] **Step 4: Create test directory and init**

```bash
mkdir -p tests/graph && touch tests/graph/__init__.py
```

- [ ] **Step 5: Create the state.py skeleton (imports only)**

Create `src/clinical_codes/graph/state.py`:

```python
import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from clinical_codes.config import MAX_ITERATIONS
from clinical_codes.schemas import CodeResult, SystemName
```

No classes yet — just imports. This lets pytest collect the test file without `ImportError` at module level.

- [ ] **Step 6: Write the MAX_ITERATIONS tests**

Create `tests/graph/test_state.py`:

```python
import operator
from typing import get_args, get_type_hints

import pytest
from pydantic import ValidationError

from clinical_codes.config import MAX_ITERATIONS
from clinical_codes.schemas import SystemName


# ── MAX_ITERATIONS ────────────────────────────────────────────────────────────

def test_max_iterations_value() -> None:
    assert MAX_ITERATIONS == 2


def test_max_iterations_type() -> None:
    assert isinstance(MAX_ITERATIONS, int)
```

- [ ] **Step 7: Run and confirm both pass**

```bash
uv run pytest tests/graph/test_state.py -v
```

Expected:
```
tests/graph/test_state.py::test_max_iterations_value PASSED
tests/graph/test_state.py::test_max_iterations_type PASSED
2 passed
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/clinical_codes/config.py \
        src/clinical_codes/graph/state.py \
        tests/graph/__init__.py tests/graph/test_state.py
git commit -m "Setup: add langgraph dep, MAX_ITERATIONS constant, graph state skeleton"
```

---

### Task 2: Write the full test file

**Files:**
- Modify: `tests/graph/test_state.py` (append all remaining tests)

> All tests use **local imports** inside each function. This lets pytest collect the file throughout the implementation — each function fails with `ImportError` only until its type is added to `state.py`, leaving already-passing tests unaffected.

- [ ] **Step 1: Append all tests to `tests/graph/test_state.py`**

Append the following (after the `test_max_iterations_type` function):

```python
# ── PlannerOutput ─────────────────────────────────────────────────────────────

def test_planner_output_valid() -> None:
    from clinical_codes.graph.state import PlannerOutput
    po = PlannerOutput(
        selected_systems=[SystemName.ICD10CM, SystemName.LOINC],
        search_terms={SystemName.ICD10CM: "diabetes", SystemName.LOINC: "glucose"},
        rationale="ICD-10-CM for condition, LOINC for lab test.",
    )
    assert po.selected_systems == [SystemName.ICD10CM, SystemName.LOINC]
    assert po.search_terms[SystemName.ICD10CM] == "diabetes"
    assert po.rationale == "ICD-10-CM for condition, LOINC for lab test."


def test_planner_output_search_terms_can_be_subset() -> None:
    from clinical_codes.graph.state import PlannerOutput
    # On refinement, search_terms contains only weak systems — not all selected_systems
    po = PlannerOutput(
        selected_systems=[SystemName.ICD10CM, SystemName.LOINC],
        search_terms={SystemName.LOINC: "glucose panel"},
        rationale="Re-querying LOINC only.",
    )
    assert SystemName.ICD10CM not in po.search_terms
    assert SystemName.LOINC in po.search_terms


def test_planner_output_missing_rationale_raises() -> None:
    from clinical_codes.graph.state import PlannerOutput
    with pytest.raises(ValidationError):
        PlannerOutput(
            selected_systems=[SystemName.ICD10CM],
            search_terms={SystemName.ICD10CM: "diabetes"},
        )


# ── EvaluatorOutput ───────────────────────────────────────────────────────────

def test_evaluator_output_sufficient() -> None:
    from clinical_codes.graph.state import EvaluatorOutput
    eo = EvaluatorOutput(
        decision="sufficient",
        weak_systems=[],
        feedback="All systems returned strong results.",
    )
    assert eo.decision == "sufficient"
    assert eo.weak_systems == []


def test_evaluator_output_refine() -> None:
    from clinical_codes.graph.state import EvaluatorOutput
    eo = EvaluatorOutput(
        decision="refine",
        weak_systems=[SystemName.LOINC],
        feedback="LOINC returned no results for 'glucose test'.",
    )
    assert eo.decision == "refine"
    assert SystemName.LOINC in eo.weak_systems


def test_evaluator_output_invalid_decision_raises() -> None:
    from clinical_codes.graph.state import EvaluatorOutput
    with pytest.raises(ValidationError):
        EvaluatorOutput(decision="maybe", weak_systems=[], feedback="Uncertain.")


# ── Attempt ───────────────────────────────────────────────────────────────────

def _make_planner_output():
    from clinical_codes.graph.state import PlannerOutput
    return PlannerOutput(
        selected_systems=[SystemName.ICD10CM],
        search_terms={SystemName.ICD10CM: "diabetes"},
        rationale="ICD-10-CM for the condition.",
    )


def _make_evaluator_output():
    from clinical_codes.graph.state import EvaluatorOutput
    return EvaluatorOutput(
        decision="sufficient",
        weak_systems=[],
        feedback="Results are strong.",
    )


def test_attempt_valid() -> None:
    from clinical_codes.graph.state import Attempt
    attempt = Attempt(
        iteration=0,
        planner_output=_make_planner_output(),
        raw_results={SystemName.ICD10CM: []},
        evaluator_output=_make_evaluator_output(),
    )
    assert attempt.iteration == 0
    assert attempt.planner_output.selected_systems == [SystemName.ICD10CM]
    assert attempt.raw_results == {SystemName.ICD10CM: []}
    assert attempt.evaluator_output.decision == "sufficient"


def test_attempt_missing_evaluator_output_raises() -> None:
    from clinical_codes.graph.state import Attempt
    with pytest.raises(ValidationError):
        Attempt(
            iteration=0,
            planner_output=_make_planner_output(),
            raw_results={},
            # evaluator_output omitted — required field
        )


# ── GraphState ────────────────────────────────────────────────────────────────

def test_graph_state_has_required_keys() -> None:
    from clinical_codes.graph.state import GraphState
    required = {
        "query", "iteration", "planner_output", "raw_results",
        "evaluator_output", "attempt_history", "consolidated", "summary",
    }
    assert required == set(get_type_hints(GraphState).keys())


def test_attempt_history_has_operator_add_reducer() -> None:
    from clinical_codes.graph.state import GraphState
    hints = get_type_hints(GraphState, include_extras=True)
    # get_args(Annotated[list[Attempt], operator.add]) → (list[Attempt], operator.add)
    args = get_args(hints["attempt_history"])
    assert operator.add in args


def test_initial_state_shape() -> None:
    from clinical_codes.graph.state import GraphState
    state: GraphState = {
        "query": "diabetes",
        "iteration": 0,
        "planner_output": None,
        "raw_results": {},
        "evaluator_output": None,
        "attempt_history": [],
        "consolidated": {},
        "summary": "",
    }
    assert state["query"] == "diabetes"
    assert state["iteration"] == 0
    assert state["planner_output"] is None
    assert state["attempt_history"] == []
    assert state["summary"] == ""


# ── route_after_evaluator ─────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    from clinical_codes.graph.state import EvaluatorOutput, GraphState, PlannerOutput
    state: GraphState = {
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
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_route_sufficient_under_cap() -> None:
    from clinical_codes.graph.state import EvaluatorOutput, route_after_evaluator
    state = _base_state(
        iteration=1,
        evaluator_output=EvaluatorOutput(
            decision="sufficient", weak_systems=[], feedback="Good."
        ),
    )
    assert route_after_evaluator(state) == "consolidator"


def test_route_refine_under_cap() -> None:
    from clinical_codes.graph.state import EvaluatorOutput, route_after_evaluator
    state = _base_state(
        iteration=1,
        evaluator_output=EvaluatorOutput(
            decision="refine",
            weak_systems=[SystemName.ICD10CM],
            feedback="ICD-10-CM returned no results.",
        ),
    )
    assert route_after_evaluator(state) == "planner"


def test_route_cap_forces_consolidate() -> None:
    from clinical_codes.graph.state import EvaluatorOutput, route_after_evaluator
    # iteration == MAX_ITERATIONS: cap hit, must forward regardless of decision
    state = _base_state(
        iteration=MAX_ITERATIONS,
        evaluator_output=EvaluatorOutput(
            decision="refine",
            weak_systems=[SystemName.ICD10CM],
            feedback="Still weak.",
        ),
    )
    assert route_after_evaluator(state) == "consolidator"


def test_route_one_below_cap_still_refines() -> None:
    from clinical_codes.graph.state import EvaluatorOutput, route_after_evaluator
    state = _base_state(
        iteration=MAX_ITERATIONS - 1,
        evaluator_output=EvaluatorOutput(
            decision="refine",
            weak_systems=[SystemName.LOINC],
            feedback="LOINC empty.",
        ),
    )
    assert route_after_evaluator(state) == "planner"
```

- [ ] **Step 2: Confirm the two MAX_ITERATIONS tests still pass; the rest fail**

```bash
uv run pytest tests/graph/test_state.py -v
```

Expected:
```
tests/graph/test_state.py::test_max_iterations_value PASSED
tests/graph/test_state.py::test_max_iterations_type PASSED
tests/graph/test_state.py::test_planner_output_valid FAILED  (ImportError)
... (all remaining tests FAILED with ImportError)
2 passed, 15 failed
```

- [ ] **Step 3: Commit the test file**

```bash
git add tests/graph/test_state.py
git commit -m "Add full test suite for graph state (all failing — red)"
```

---

### Task 3: Implement `PlannerOutput`

**Files:**
- Modify: `src/clinical_codes/graph/state.py`

- [ ] **Step 1: Append `PlannerOutput` to `state.py`**

Add after the imports in `src/clinical_codes/graph/state.py`:

```python
class PlannerOutput(BaseModel):
    selected_systems: list[SystemName]
    search_terms: dict[SystemName, str]
    rationale: str
```

- [ ] **Step 2: Run PlannerOutput tests**

```bash
uv run pytest tests/graph/test_state.py::test_planner_output_valid \
              tests/graph/test_state.py::test_planner_output_search_terms_can_be_subset \
              tests/graph/test_state.py::test_planner_output_missing_rationale_raises -v
```

Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add src/clinical_codes/graph/state.py
git commit -m "Implement PlannerOutput model"
```

---

### Task 4: Implement `EvaluatorOutput`

**Files:**
- Modify: `src/clinical_codes/graph/state.py`

- [ ] **Step 1: Append `EvaluatorOutput` to `state.py`**

Add after `PlannerOutput`:

```python
class EvaluatorOutput(BaseModel):
    decision: Literal["sufficient", "refine"]
    weak_systems: list[SystemName]
    feedback: str
```

- [ ] **Step 2: Run EvaluatorOutput tests**

```bash
uv run pytest tests/graph/test_state.py::test_evaluator_output_sufficient \
              tests/graph/test_state.py::test_evaluator_output_refine \
              tests/graph/test_state.py::test_evaluator_output_invalid_decision_raises -v
```

Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add src/clinical_codes/graph/state.py
git commit -m "Implement EvaluatorOutput model"
```

---

### Task 5: Implement `Attempt`

**Files:**
- Modify: `src/clinical_codes/graph/state.py`

- [ ] **Step 1: Append `Attempt` to `state.py`**

Add after `EvaluatorOutput`:

```python
class Attempt(BaseModel):
    iteration: int
    planner_output: PlannerOutput
    raw_results: dict[SystemName, list[CodeResult]]
    evaluator_output: EvaluatorOutput
```

- [ ] **Step 2: Run Attempt tests**

```bash
uv run pytest tests/graph/test_state.py::test_attempt_valid \
              tests/graph/test_state.py::test_attempt_missing_evaluator_output_raises -v
```

Expected: 2 passed

- [ ] **Step 3: Commit**

```bash
git add src/clinical_codes/graph/state.py
git commit -m "Implement Attempt model"
```

---

### Task 6: Implement `GraphState`

**Files:**
- Modify: `src/clinical_codes/graph/state.py`

- [ ] **Step 1: Append `GraphState` to `state.py`**

Add after `Attempt`:

```python
class GraphState(TypedDict):
    query: str
    iteration: int
    planner_output: PlannerOutput | None
    raw_results: dict[SystemName, list[CodeResult]]
    evaluator_output: EvaluatorOutput | None
    attempt_history: Annotated[list[Attempt], operator.add]
    consolidated: dict[SystemName, list[CodeResult]]
    summary: str
```

- [ ] **Step 2: Run GraphState tests**

```bash
uv run pytest tests/graph/test_state.py::test_graph_state_has_required_keys \
              tests/graph/test_state.py::test_attempt_history_has_operator_add_reducer \
              tests/graph/test_state.py::test_initial_state_shape -v
```

Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add src/clinical_codes/graph/state.py
git commit -m "Implement GraphState TypedDict with operator.add reducer on attempt_history"
```

---

### Task 7: Implement `route_after_evaluator` and final verification

**Files:**
- Modify: `src/clinical_codes/graph/state.py`

- [ ] **Step 1: Append `route_after_evaluator` to `state.py`**

Add after `GraphState`:

```python
def route_after_evaluator(state: GraphState) -> str:
    if state["iteration"] >= MAX_ITERATIONS:
        return "consolidator"
    if state["evaluator_output"].decision == "refine":
        return "planner"
    return "consolidator"
```

- [ ] **Step 2: Run route_after_evaluator tests**

```bash
uv run pytest tests/graph/test_state.py::test_route_sufficient_under_cap \
              tests/graph/test_state.py::test_route_refine_under_cap \
              tests/graph/test_state.py::test_route_cap_forces_consolidate \
              tests/graph/test_state.py::test_route_one_below_cap_still_refines -v
```

Expected: 4 passed

- [ ] **Step 3: Run the full test suite — confirm no regressions**

```bash
uv run pytest -v
```

Expected: all 36 tests pass (19 tools tests + 17 graph/state tests)

- [ ] **Step 4: Commit**

```bash
git add src/clinical_codes/graph/state.py
git commit -m "Implement route_after_evaluator — graph state complete"
```
