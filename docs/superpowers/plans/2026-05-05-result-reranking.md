# Result Re-Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `consolidator` node with a `re_ranker` node that pools all domain-filtered codes from all systems, calls an LLM to rank them by query relevance, and returns a flat `list[CodeResult]` ordered most-to-least relevant.

**Architecture:** The `consolidator` (deterministic dedup/trim per system, grouped output) is replaced by `re_ranker` (domain filter → pool → optional LLM ranking → flat list). The `consolidated` state field type changes from `dict[SystemName, list[CodeResult]]` to `list[CodeResult]`. All downstream consumers (summarizer, Streamlit UI) are updated to work with the flat list.

**Tech Stack:** Python 3.12, LangGraph, LangChain Anthropic, Pydantic, Streamlit, pytest

---

## File Map

| File | Change |
|---|---|
| `src/clinical_codes/config.py` | Add `flat_results`, `re_ranker_temperature`; remove `display_results`; rename `NODE_CONSOLIDATOR` → `NODE_RE_RANKER` |
| `src/clinical_codes/graph/state.py` | Add `RankedCode`, `ReRankerOutput`; change `consolidated` type to `list[CodeResult]`; update docstring |
| `src/clinical_codes/graph/prompts.py` | Add `_RE_RANKER_SYSTEM`, `build_re_ranker_messages`; update `build_summarizer_messages` and `_SUMMARIZER_SYSTEM` |
| `src/clinical_codes/graph/nodes.py` | Remove `consolidator`; add `_re_ranker_chain` and `re_ranker`; update imports |
| `src/clinical_codes/graph/builder.py` | Update imports; rename node to `re_ranker`; update all `NODE_CONSOLIDATOR` → `NODE_RE_RANKER`; update `make_initial_state` |
| `src/clinical_codes/app/streamlit_app.py` | Update results display from grouped dict to flat ranked table |
| `tests/graph/test_nodes.py` | Remove 8 `test_consolidator_*` tests; update `_make_state`; update `test_summarizer_writes_summary`; add 6 `test_re_ranker_*` tests |
| `tests/graph/test_prompts.py` | Update 4 summarizer tests (dict → list); add 2 `test_build_re_ranker_messages*` tests |
| `tests/graph/test_builder.py` | Update `NODE_CONSOLIDATOR` → `NODE_RE_RANKER` import; update 3 route assertions; update `consolidated == []` check |

---

### Task 1: Config — add flat_results and re_ranker_temperature, rename NODE_CONSOLIDATOR, remove display_results

**Files:**
- Modify: `src/clinical_codes/config.py`

- [ ] **Step 1: Verify current tests pass**

```
uv run pytest --tb=short -q
```

Expected: all tests pass (baseline).

- [ ] **Step 2: Update `config.py`**

Replace the entire `Settings` class and module-level constants with:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""

    # LLM model — all LLM nodes use this model
    llm_model: str = "claude-sonnet-4-6"
    # Temperatures per scope.md: 0 for deterministic planning/evaluation, 0.3 for prose
    planner_temperature: float = 0.0
    evaluator_temperature: float = 0.0
    re_ranker_temperature: float = 0.0
    summarizer_temperature: float = 0.3

    # NLM Clinical Tables API — trailing slash so httpx joins paths correctly
    nlm_api_base: str = "https://clinicaltables.nlm.nih.gov/api/"
    api_timeout: float = 10.0
    api_max_retries: int = 2      # retries after initial failure (3 total attempts)
    api_backoff_base: float = 1.0  # first retry delay in seconds; doubles each attempt

    fetch_results: int = 10   # results fetched per system per executor call
    flat_results: int = 5     # max codes in the flat re-ranked output

    confidence_threshold: float = 0.5  # reserved — evaluator uses semantic judgment, not this threshold


settings = Settings()

MAX_ITERATIONS = 2  # cap enforced in route_after_evaluator, not in LLM prompts
NODE_PLANNER = "planner"
NODE_RE_RANKER = "re_ranker"
```

- [ ] **Step 3: Run tests — expect failures from files that still reference NODE_CONSOLIDATOR or display_results**

```
uv run pytest --tb=short -q 2>&1 | head -40
```

Expected: `ImportError` or `AttributeError` in `builder.py` and `nodes.py` (they still import `NODE_CONSOLIDATOR` and reference `settings.display_results`). This is correct — the downstream files will be fixed in subsequent tasks.

---

### Task 2: State — add RankedCode and ReRankerOutput; change consolidated type

**Files:**
- Modify: `src/clinical_codes/graph/state.py`

- [ ] **Step 1: Update `state.py`**

Replace the full file:

```python
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
```

- [ ] **Step 2: Verify RankedCode and ReRankerOutput are importable**

```
uv run python -c "from clinical_codes.graph.state import RankedCode, ReRankerOutput; print('OK')"
```

Expected: `OK`

---

### Task 3: Prompts — add re_ranker message builder; update summarizer for flat consolidated

**Files:**
- Modify: `src/clinical_codes/graph/prompts.py`
- Modify: `tests/graph/test_prompts.py`

- [ ] **Step 1: Write failing tests for new re_ranker prompt builder**

Append to `tests/graph/test_prompts.py`:

```python
def test_build_re_ranker_messages() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from clinical_codes.graph.prompts import build_re_ranker_messages

    pool = [
        CodeResult(system=SystemName.ICD10CM, code="I10", display="Essential hypertension", score=1.0, raw={}),
        CodeResult(system=SystemName.RXNORM, code="854901", display="lisinopril 20 MG Oral Tablet", score=0.9, raw={}),
    ]
    messages = build_re_ranker_messages("hypertension", pool, flat_results=5)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    human = messages[1].content
    assert "hypertension" in human
    assert "[ICD10CM:I10]" in human
    assert "Essential hypertension" in human
    assert "[RXNORM:854901]" in human
    assert "lisinopril 20 MG Oral Tablet" in human


def test_build_re_ranker_messages_includes_flat_results_count() -> None:
    from clinical_codes.graph.prompts import build_re_ranker_messages

    pool = [
        CodeResult(system=SystemName.ICD10CM, code="I10", display="Essential hypertension", score=1.0, raw={}),
    ]
    messages = build_re_ranker_messages("test", pool, flat_results=3)
    human = messages[1].content
    assert "3" in human  # top-N count must appear in the human message
```

- [ ] **Step 2: Run new tests — expect failure (function does not exist yet)**

```
uv run pytest tests/graph/test_prompts.py::test_build_re_ranker_messages tests/graph/test_prompts.py::test_build_re_ranker_messages_includes_flat_results_count -v
```

Expected: `ImportError` or `AttributeError` — `build_re_ranker_messages` does not exist.

- [ ] **Step 3: Update summarizer tests to use flat `list[CodeResult]`**

In `tests/graph/test_prompts.py`, update four tests. Replace them in full:

```python
def test_build_summarizer_messages() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from clinical_codes.graph.prompts import build_summarizer_messages

    consolidated = [_make_result(SystemName.ICD10CM, "Essential (primary) hypertension", "I10")]
    attempt = _attempt()
    messages = build_summarizer_messages(
        "hypertension", consolidated, "ICD-10-CM covers diagnoses", [attempt]
    )

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "clinical information specialist" in messages[0].content
    human = messages[1].content
    assert "hypertension" in human
    assert "Essential (primary) hypertension" in human
    assert "ICD10CM I10" in human        # new flat format: [ICD10CM I10]
    assert "Reasoning trace" in human
    assert "Iteration" in human


def test_summarizer_truncates_to_five() -> None:
    from clinical_codes.graph.prompts import build_summarizer_messages

    results = [
        _make_result(SystemName.ICD10CM, f"Result {i}", f"X{i:02d}") for i in range(10)
    ]
    human = build_summarizer_messages(
        "test", results, "rationale", [_attempt()]
    )[1].content
    assert "Result 4" in human      # 5th result shown (index 4)
    assert "Result 5" not in human  # 6th result excluded


def test_build_summarizer_cap_hit_note_present() -> None:
    from clinical_codes.graph.prompts import build_summarizer_messages

    attempt = _attempt()
    human = build_summarizer_messages("hypertension", [], "rationale", [attempt])[1].content
    assert "Cap-hit" in human
    assert "LOINC returned no results for this drug query" in human


def test_build_summarizer_no_cap_hit_when_sufficient() -> None:
    from clinical_codes.graph.prompts import build_summarizer_messages

    attempt = Attempt(
        iteration=1,
        planner_output=_planner_output(),
        raw_results={},
        evaluator_output=_evaluator_output(decision="sufficient"),
    )
    human = build_summarizer_messages("hypertension", [], "rationale", [attempt])[1].content
    assert "Cap-hit" not in human
```

- [ ] **Step 4: Run updated summarizer tests — expect failure (implementation still uses dict)**

```
uv run pytest tests/graph/test_prompts.py -k "summarizer" -v
```

Expected: failures because `build_summarizer_messages` still expects a dict.

- [ ] **Step 5: Update `prompts.py`**

Make the following changes to `src/clinical_codes/graph/prompts.py`:

**a) Add import for CodeResult at the top (it's already imported via schemas):**

The existing imports already include `CodeResult` and `SystemName` — no change needed there.

**b) Add `build_re_ranker_messages` to the import in `nodes.py` later — for now just add the function and system string to `prompts.py`.**

**c) Add `_RE_RANKER_SYSTEM` constant after `_EVALUATOR_SYSTEM`:**

```python
_RE_RANKER_SYSTEM = """You are a clinical code relevance ranker. Given a user query and a pool of
candidate codes from multiple medical coding systems, select and return the
codes most relevant to the query.

Ranking criteria:
- Rank by how directly the code matches the specific clinical concept in the query.
- Prefer specificity: "lisinopril 20 MG Oral Tablet" ranks above "lisinopril (Oral Pill)"
  for the query "lisinopril 20 mg".
- Return fewer codes if fewer are relevant.
- Return only codes from the provided pool — do not invent codes."""
```

**d) Add `build_re_ranker_messages` function after `build_evaluator_messages`:**

```python
def build_re_ranker_messages(
    query: str, pool: list[CodeResult], flat_results: int
) -> list[BaseMessage]:
    code_lines = "\n".join(
        f"  [{r.system}:{r.code}] {r.display}" for r in pool
    )
    human = (
        f"Query: {query}\n\n"
        f"Candidate codes ({len(pool)} total):\n{code_lines}\n\n"
        f"Return the top {flat_results} codes most relevant to the query, "
        f"ranked most to least relevant."
    )
    return [SystemMessage(content=_RE_RANKER_SYSTEM), HumanMessage(content=human)]
```

**e) Update `_SUMMARIZER_SYSTEM`** — replace the line:

```python
"- Do not repeat individual codes or list results by system — those are shown separately above."
```

with:

```python
"- Do not repeat individual codes — they are shown above ranked by relevance to the query."
```

**f) Update `build_summarizer_messages` signature and result block:**

Replace the function signature and result-building loop:

```python
def build_summarizer_messages(
    query: str,
    consolidated: list[CodeResult],
    rationale: str,
    attempt_history: list[Attempt],
) -> list[BaseMessage]:
    result_lines: list[str] = []
    for i, r in enumerate(consolidated[:settings.flat_results], 1):
        result_lines.append(f"  {i}. [{r.system} {r.code}] {r.display}")
```

The rest of `build_summarizer_messages` (cap_hit detection, trace_block, human assembly) remains unchanged.

Also add `from clinical_codes.config import settings` at the top of `prompts.py` if not already imported. Check the current imports — currently prompts.py does NOT import settings. Add it:

```python
from clinical_codes.config import settings
```

- [ ] **Step 6: Run re_ranker prompt tests — expect pass**

```
uv run pytest tests/graph/test_prompts.py::test_build_re_ranker_messages tests/graph/test_prompts.py::test_build_re_ranker_messages_includes_flat_results_count -v
```

Expected: PASS.

- [ ] **Step 7: Run all prompts tests — expect pass**

```
uv run pytest tests/graph/test_prompts.py -v
```

Expected: all pass (some may still fail due to downstream import errors — that's OK for now).

---

### Task 4: Nodes — remove consolidator, add re_ranker

**Files:**
- Modify: `src/clinical_codes/graph/nodes.py`
- Modify: `tests/graph/test_nodes.py`

- [ ] **Step 1: Remove all 8 consolidator tests from `tests/graph/test_nodes.py`**

Delete the entire `# ── Consolidator ──` section (lines 68–218), which includes these tests:
- `test_consolidator_filters_to_selected_systems`
- `test_consolidator_deduplicates_by_code`
- `test_consolidator_sorts_by_score_descending`
- `test_consolidator_trims_to_display_results`
- `test_consolidator_dedup_keeps_highest_score`
- `test_consolidator_empty_results_system`
- `test_consolidator_filters_by_relevant_codes`
- `test_consolidator_no_filter_when_relevant_codes_empty`
- `test_consolidator_empty_list_removes_all_results`

Also update `_make_state` — change `"consolidated": {}` to `"consolidated": []`.

- [ ] **Step 2: Update `test_summarizer_writes_summary` in `tests/graph/test_nodes.py`**

Replace the test:

```python
async def test_summarizer_writes_summary() -> None:
    from clinical_codes.graph.nodes import summarizer

    po = _make_planner_output()
    consolidated = [_make_result(SystemName.ICD10CM, "I10", "Hypertension")]

    fake_response = MagicMock()
    fake_response.content = "Hypertension is a condition..."

    with patch("clinical_codes.graph.nodes._summarizer_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)
        with patch("clinical_codes.graph.nodes.build_summarizer_messages") as mock_build:
            mock_build.return_value = []
            result = await summarizer(
                _make_state(planner_output=po, consolidated=consolidated)
            )
            mock_build.assert_called_once_with("hypertension", consolidated, po.rationale, [])

    assert result["summary"] == "Hypertension is a condition..."
```

- [ ] **Step 3: Add 6 new re_ranker tests to `tests/graph/test_nodes.py`**

Add a new `# ── Re_ranker ──` section after the helpers and before `# ── Planner ──`:

```python
# ── Re_ranker ─────────────────────────────────────────────────────────────────

async def test_re_ranker_empty_pool_returns_empty() -> None:
    from clinical_codes.graph.nodes import re_ranker

    # No raw results for the selected system → empty pool, no LLM call
    state = _make_state(raw_results={})
    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        result = await re_ranker(state)
        mock_chain.ainvoke.assert_not_called()
    assert result["consolidated"] == []


async def test_re_ranker_small_pool_skips_llm() -> None:
    from clinical_codes.graph.nodes import re_ranker

    # Pool has 3 codes ≤ flat_results (5) — returns as-is, no LLM call
    results = [_make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}") for i in range(3)]
    state = _make_state(raw_results={SystemName.ICD10CM: results})

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        result = await re_ranker(state)
        mock_chain.ainvoke.assert_not_called()

    assert result["consolidated"] == results


async def test_re_ranker_calls_llm_for_large_pool() -> None:
    from clinical_codes.graph.nodes import re_ranker
    from clinical_codes.graph.state import RankedCode, ReRankerOutput

    # Pool has 7 codes > flat_results (5) — LLM called, top 5 returned in ranked order
    results = [_make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}") for i in range(7)]
    state = _make_state(raw_results={SystemName.ICD10CM: results})

    top_codes = [RankedCode(system=SystemName.ICD10CM, code=f"I{i:02d}") for i in range(5)]
    mock_output = ReRankerOutput(ranked_codes=top_codes)

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=mock_output)
        result = await re_ranker(state)

    assert len(result["consolidated"]) == 5
    assert [r.code for r in result["consolidated"]] == [f"I{i:02d}" for i in range(5)]


async def test_re_ranker_applies_domain_filter() -> None:
    from clinical_codes.graph.nodes import re_ranker

    # evaluator keeps only I10 — I51 is filtered before pool, so pool size is 1 (≤ 5)
    state = _make_state(
        evaluator_output=_make_evaluator_output(
            relevant_codes={SystemName.ICD10CM: ["I10"]}
        ),
        raw_results={
            SystemName.ICD10CM: [
                _make_result(SystemName.ICD10CM, "I10", "Essential hypertension"),
                _make_result(SystemName.ICD10CM, "I51", "Unspecified heart disease"),
            ]
        },
    )
    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        result = await re_ranker(state)
        mock_chain.ainvoke.assert_not_called()  # pool ≤ flat_results after filter

    codes = [r.code for r in result["consolidated"]]
    assert codes == ["I10"]
    assert "I51" not in codes


async def test_re_ranker_drops_invalid_llm_codes() -> None:
    from clinical_codes.graph.nodes import re_ranker
    from clinical_codes.graph.state import RankedCode, ReRankerOutput

    # Pool has 7 codes — LLM path triggered. LLM returns a code not in pool → dropped.
    results = [_make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}") for i in range(7)]
    state = _make_state(raw_results={SystemName.ICD10CM: results})

    ranked = [
        RankedCode(system=SystemName.ICD10CM, code="I00"),    # in pool
        RankedCode(system=SystemName.ICD10CM, code="FAKE99"), # not in pool — must be dropped
        RankedCode(system=SystemName.ICD10CM, code="I01"),    # in pool
    ]

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=ReRankerOutput(ranked_codes=ranked))
        result = await re_ranker(state)

    codes = [r.code for r in result["consolidated"]]
    assert "FAKE99" not in codes
    assert "I00" in codes
    assert "I01" in codes


async def test_re_ranker_deduplicates_llm_output() -> None:
    from clinical_codes.graph.nodes import re_ranker
    from clinical_codes.graph.state import RankedCode, ReRankerOutput

    # Pool has 7 codes — LLM path triggered. LLM returns I00 twice — second occurrence dropped.
    results = [_make_result(SystemName.ICD10CM, f"I{i:02d}", f"Result {i}") for i in range(7)]
    state = _make_state(raw_results={SystemName.ICD10CM: results})

    ranked = [
        RankedCode(system=SystemName.ICD10CM, code="I00"),
        RankedCode(system=SystemName.ICD10CM, code="I00"),  # duplicate
        RankedCode(system=SystemName.ICD10CM, code="I01"),
    ]

    with patch("clinical_codes.graph.nodes._re_ranker_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value=ReRankerOutput(ranked_codes=ranked))
        result = await re_ranker(state)

    codes = [r.code for r in result["consolidated"]]
    assert codes.count("I00") == 1
    assert "I01" in codes
```

- [ ] **Step 4: Run new re_ranker tests — expect failure (re_ranker not implemented yet)**

```
uv run pytest tests/graph/test_nodes.py -k "re_ranker" -v
```

Expected: `ImportError` — `re_ranker` not yet defined in `nodes.py`.

- [ ] **Step 5: Update `nodes.py` — remove consolidator, add re_ranker**

Replace the full file:

```python
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
    if len(pool) <= settings.flat_results:
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

    return {"consolidated": ranked[:settings.flat_results]}


async def summarizer(state: GraphState) -> dict:
    messages = build_summarizer_messages(
        state["query"],
        state["consolidated"],
        state["planner_output"].rationale,
        state["attempt_history"],
    )
    response = await _summarizer_llm.ainvoke(messages)
    return {"summary": response.content}
```

- [ ] **Step 6: Run all node tests — expect pass**

```
uv run pytest tests/graph/test_nodes.py -v
```

Expected: all pass (builder.py still references old names — that's OK, builder tests haven't been updated yet).

---

### Task 5: Builder — rewire for re_ranker node

**Files:**
- Modify: `src/clinical_codes/graph/builder.py`
- Modify: `tests/graph/test_builder.py`

- [ ] **Step 1: Update `tests/graph/test_builder.py`**

Replace the import line at the top:

```python
from clinical_codes.config import MAX_ITERATIONS, NODE_RE_RANKER, NODE_PLANNER
```

Remove the `NODE_CONSOLIDATOR` import entirely.

Rename the test and update its assertion:

```python
def test_route_after_planner_empty_selection_skips_to_re_ranker() -> None:
    from clinical_codes.graph.builder import route_after_planner
    from clinical_codes.graph.state import PlannerOutput

    state = _base_state(
        planner_output=PlannerOutput(
            selected_systems=[],
            search_terms={},
            rationale="Query is gibberish — no clinical term detected.",
        )
    )
    assert route_after_planner(state) == NODE_RE_RANKER
```

Update `test_route_after_planner_with_systems_goes_to_executor` — no change needed (it asserts `"executor"`).

Update `test_route_sufficient_under_cap`:

```python
def test_route_sufficient_under_cap() -> None:
    from clinical_codes.graph.builder import route_after_evaluator
    from clinical_codes.graph.state import EvaluatorOutput

    state = _base_state(
        iteration=1,
        evaluator_output=EvaluatorOutput(
            decision="sufficient", weak_systems=[], feedback="Good."
        ),
    )
    assert route_after_evaluator(state) == NODE_RE_RANKER
```

Update `test_route_cap_forces_consolidate` — rename and update assertion:

```python
def test_route_cap_forces_re_ranker() -> None:
    from clinical_codes.graph.builder import route_after_evaluator
    from clinical_codes.graph.state import EvaluatorOutput

    state = _base_state(
        iteration=MAX_ITERATIONS,
        evaluator_output=EvaluatorOutput(
            decision="refine",
            weak_systems=[SystemName.ICD10CM],
            feedback="Still weak.",
        ),
    )
    assert route_after_evaluator(state) == NODE_RE_RANKER
```

Update `test_make_initial_state_defaults`:

```python
def test_make_initial_state_defaults() -> None:
    from clinical_codes.graph.builder import make_initial_state

    state = make_initial_state("hypertension")
    assert state["iteration"] == 0
    assert state["planner_output"] is None
    assert state["raw_results"] == {}
    assert state["evaluator_output"] is None
    assert state["attempt_history"] == []
    assert state["consolidated"] == []
    assert state["summary"] == ""
```

- [ ] **Step 2: Run updated builder tests — expect failures (builder.py still uses old names)**

```
uv run pytest tests/graph/test_builder.py -v
```

Expected: failures — `NODE_RE_RANKER` does not exist yet in builder scope OR builder imports fail.

- [ ] **Step 3: Update `builder.py`**

Replace the full file:

```python
from langgraph.graph import END, StateGraph

from clinical_codes.config import MAX_ITERATIONS, NODE_RE_RANKER, NODE_PLANNER
from clinical_codes.graph.nodes import (
    evaluator,
    executor,
    planner,
    re_ranker,
    summarizer,
)
from clinical_codes.graph.state import GraphState


def route_after_planner(state: GraphState) -> str:
    if not state["planner_output"].selected_systems:
        return NODE_RE_RANKER
    return "executor"


def route_after_evaluator(state: GraphState) -> str:
    # `iteration` is post-increment (planner writes iteration + 1 at the start of each pass).
    # At MAX_ITERATIONS the cap fires regardless of the evaluator's decision.
    if state["iteration"] >= MAX_ITERATIONS:
        return NODE_RE_RANKER
    if state["evaluator_output"].decision == "refine":
        return NODE_PLANNER
    return NODE_RE_RANKER


def make_initial_state(query: str) -> GraphState:
    return {
        "query": query,
        "iteration": 0,
        "planner_output": None,
        "raw_results": {},
        "evaluator_output": None,
        "attempt_history": [],
        "consolidated": [],
        "summary": "",
    }


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node(NODE_PLANNER, planner)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator)
    graph.add_node(NODE_RE_RANKER, re_ranker)
    graph.add_node("summarizer", summarizer)

    graph.set_entry_point(NODE_PLANNER)

    graph.add_conditional_edges(
        NODE_PLANNER,
        route_after_planner,
        {"executor": "executor", NODE_RE_RANKER: NODE_RE_RANKER},
    )
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {NODE_PLANNER: NODE_PLANNER, NODE_RE_RANKER: NODE_RE_RANKER},
    )
    graph.add_edge(NODE_RE_RANKER, "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
```

- [ ] **Step 4: Run all tests — expect full suite to pass**

```
uv run pytest --tb=short -q
```

Expected: all tests pass. If any fail, diagnose from the `--tb=short` output.

- [ ] **Step 5: Commit**

```
git add src/clinical_codes/config.py src/clinical_codes/graph/state.py src/clinical_codes/graph/prompts.py src/clinical_codes/graph/nodes.py src/clinical_codes/graph/builder.py tests/graph/test_nodes.py tests/graph/test_prompts.py tests/graph/test_builder.py
git commit -m "feat: replace consolidator with re_ranker node for query-relevance ranking"
```

---

### Task 6: Streamlit UI — flat ranked results table

**Files:**
- Modify: `src/clinical_codes/app/streamlit_app.py`

- [ ] **Step 1: Update the results display block in `streamlit_app.py`**

Replace the results section (lines 53–69 in the current file) from:

```python
    if not consolidated:
        st.info("No results found.")
    else:
        rows = []
        for system, results in consolidated.items():
            term = search_terms.get(system, "")
            for r in results:
                row: dict = {
                    "System": system.value,
                    "Code": r.code,
                    "Display": r.display,
                    "Searched as": term,
                }
                if system.value == "RXNORM" and "row" in r.raw and len(r.raw["row"]) > 2:
                    row["Strengths"] = r.raw["row"][2]
                rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)
```

to:

```python
    if not consolidated:
        st.info("No results found.")
    else:
        rows = []
        for i, r in enumerate(consolidated, 1):
            term = search_terms.get(r.system, "")
            row: dict = {
                "Rank": i,
                "System": r.system.value,
                "Code": r.code,
                "Display": r.display,
                "Searched as": term,
            }
            if r.system.value == "RXNORM" and "row" in r.raw and len(r.raw["row"]) > 2:
                row["Strengths"] = r.raw["row"][2]
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)
```

- [ ] **Step 2: Verify the app starts without error**

```
uv run streamlit run src/clinical_codes/app/streamlit_app.py --server.headless true &
```

Wait 5 seconds, then kill the process. No import errors should appear.

- [ ] **Step 3: Run full test suite — confirm nothing regressed**

```
uv run pytest --tb=short -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```
git add src/clinical_codes/app/streamlit_app.py
git commit -m "feat: update Streamlit results display to flat ranked table"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Covered by task |
|---|---|
| `NODE_CONSOLIDATOR` → `NODE_RE_RANKER` in config | Task 1 |
| Add `flat_results: int = 5` | Task 1 |
| Add `re_ranker_temperature: float = 0.0` | Task 1 |
| Remove `display_results` | Task 1 |
| Add `RankedCode`, `ReRankerOutput` to state | Task 2 |
| `consolidated` type → `list[CodeResult]` | Task 2, Task 5 |
| `_RE_RANKER_SYSTEM` static prompt | Task 3 |
| `build_re_ranker_messages(query, pool, flat_results)` | Task 3 |
| `[SYSTEM:CODE]` format in candidate codes | Task 3 |
| `_SUMMARIZER_SYSTEM` line update | Task 3 |
| `build_summarizer_messages` flat enumerated format | Task 3 |
| `re_ranker` node: domain filter → pool → short-circuit → LLM → assemble | Task 4 |
| `_re_ranker_chain` with structured output | Task 4 |
| Pool ≤ flat_results → return as-is (no LLM) | Task 4 + test |
| Empty pool → return `[]` (no LLM) | Task 4 + test |
| LLM invalid code → silently dropped | Task 4 + test |
| LLM duplicate code → first occurrence kept | Task 4 + test |
| builder: `consolidator` node → `re_ranker` | Task 5 |
| `route_after_planner` miss-query → `NODE_RE_RANKER` | Task 5 |
| `route_after_evaluator` → `NODE_RE_RANKER` | Task 5 |
| `make_initial_state` consolidated `[]` | Task 5 |
| `test_re_ranker_*` 6 tests | Task 4 |
| `test_build_re_ranker_messages` 2 tests | Task 3 |
| Summarizer tests: `consolidated` dict → list | Task 3 |
| Builder tests: `NODE_CONSOLIDATOR` → `NODE_RE_RANKER` | Task 5 |
| Streamlit flat table with Rank column | Task 6 |

### Type consistency check

- `build_re_ranker_messages(query: str, pool: list[CodeResult], flat_results: int)` — `flat_results` is an `int`, passed as `settings.flat_results` in the node. ✓
- `ReRankerOutput.ranked_codes: list[RankedCode]` — consumed in `re_ranker` node via `output.ranked_codes`. ✓
- `RankedCode(system: SystemName, code: str)` — pool_index key is `(r.system, r.code)` matching `(rc.system, rc.code)`. ✓
- `consolidated: list[CodeResult]` in `GraphState` and `make_initial_state` both use `[]`. ✓
- `build_summarizer_messages(... consolidated: list[CodeResult] ...)` — called from `summarizer` node with `state["consolidated"]` (now a list). ✓
- Streamlit iterates `consolidated` (list) with `enumerate`. ✓

### Placeholder scan

No TBD, TODO, or incomplete sections found.
