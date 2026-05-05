# Result Re-Ranking Design

**Date:** 2026-05-05
**Status:** Approved

---

## Problem

The current system keeps the top 5 codes *per system* ranked by API position. The API rank is a positional score — the top result always scores 1.0 regardless of how well it actually matches the user's query. There is no mechanism to compare codes against the query itself, and results are grouped by system rather than by relevance.

This means a code that is marginally related to the query but happened to rank first in the API's response appears in the final output, while a highly relevant code from a different system that ranked third may not. The user gets what the API returned in order, not what is most relevant to their query.

---

## Goal

Pool all codes from all selected systems, score each independently against the original user query using an LLM, and return a flat ordered list of the top N most relevant codes. System becomes metadata on each result, not an organizing structure.

---

## Section 1: Pipeline Structure

The `consolidator` node is **replaced** by a `re_ranker` node. Node count stays at 5.

```
planner → executor → evaluator → re_ranker → summarizer
```

The miss-query short-circuit becomes `planner → re_ranker` (empty pool → empty list, no LLM call).

**Re_ranker responsibilities (in order):**
1. Apply evaluator's `relevant_codes` domain filter per system (same logic as old consolidator)
2. Flatten filtered results from all systems into one pool
3. If pool is empty → return `[]` immediately, no LLM call
4. If pool ≤ `flat_results` → return pool as-is (API order), no LLM call
5. Otherwise → call LLM with query + pool, receive top-N ranked codes
6. Look up each ranked code in the pool, assemble final `list[CodeResult]`

---

## Section 2: State and Config Changes

### `GraphState.consolidated` type

```python
# Before
consolidated: dict[SystemName, list[CodeResult]]

# After
consolidated: list[CodeResult]   # flat, ordered by query relevance, max flat_results entries
```

`CodeResult` is unchanged — `system` is already a field on it, so it works naturally as a flat list element.

### `config.py`

```python
# Add
flat_results: int = 5   # max codes in the flat re-ranked output

# Remove
display_results: int = 5   # no longer needed; was the per-system cap in old consolidator
```

### `config.py` — node name constant

```python
# Before
NODE_CONSOLIDATOR = "consolidator"

# After
NODE_RE_RANKER = "re_ranker"
```

### `EvaluatorOutput` — no change

The evaluator still produces `relevant_codes: dict[SystemName, list[str]]` for domain filtering. Re_ranker consumes this as its pool-building input.

---

## Section 3: Re_ranker Node

### New Pydantic models (in `state.py`)

```python
class RankedCode(BaseModel):
    system: SystemName
    code: str

class ReRankerOutput(BaseModel):
    ranked_codes: list[RankedCode]   # ordered most → least relevant, max flat_results entries
```

### System prompt (in `prompts.py`)

Static string — no substitution needed:

```
You are a clinical code relevance ranker. Given a user query and a pool of
candidate codes from multiple medical coding systems, select and return the
codes most relevant to the query.

Ranking criteria:
- Rank by how directly the code matches the specific clinical concept in the query.
- Prefer specificity: "lisinopril 20 MG Oral Tablet" ranks above "lisinopril (Oral Pill)"
  for the query "lisinopril 20 mg".
- Return fewer codes if fewer are relevant.
- Return only codes from the provided pool — do not invent codes.
```

### Human message format (in `prompts.py`)

Built dynamically by `build_re_ranker_messages(query, pool, flat_results)`. The count instruction goes here (not in the static system prompt):

```
Query: lisinopril 20 mg

Candidate codes (7 total):
  [RXNORM:854901] lisinopril 20 MG Oral Tablet
  [RXNORM:314076] lisinopril 10 MG Oral Tablet
  [ICD10CM:I10] Essential (primary) hypertension
  ...

Return the top 5 codes most relevant to the query, ranked most to least relevant.
```

### Node implementation sketch (in `nodes.py`)

```python
_re_ranker_chain = (
    ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.anthropic_api_key or "placeholder-for-tests",
    )
    .with_structured_output(ReRankerOutput)
)

async def re_ranker(state: GraphState) -> dict:
    ev = state["evaluator_output"]
    relevant = ev.relevant_codes if ev else {}
    raw = state["raw_results"]
    selected = state["planner_output"].selected_systems if state["planner_output"] else []

    # Build pool: apply domain filter per system, then flatten
    pool: list[CodeResult] = []
    for system in selected:
        results = raw.get(system, [])
        keep = relevant.get(system, None)
        if keep is not None:
            keep_set = set(keep)
            results = [r for r in results if r.code in keep_set]
        pool.extend(results)

    # Short-circuit: empty pool or small pool
    if not pool:
        return {"consolidated": []}
    if len(pool) <= settings.flat_results:
        return {"consolidated": pool}

    # LLM ranking
    messages = build_re_ranker_messages(state["query"], pool, settings.flat_results)
    output: ReRankerOutput = await _re_ranker_chain.ainvoke(messages)

    # Assemble results, validating against pool
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
```

### LLM configuration

| Setting | Value | Reason |
|---|---|---|
| Model | `settings.llm_model` (claude-sonnet-4-6) | Requires clinical vocabulary reasoning to rank by query relevance |
| Temperature | `settings.re_ranker_temperature` (default `0.0`) | Deterministic ranking for reproducible evaluation |

Add to `config.py`:
```python
re_ranker_temperature: float = 0.0
```

---

## Section 4: Summarizer and UI Changes

### `build_summarizer_messages` in `prompts.py`

Result block changes from grouped-by-system to a flat enumerated list:

```python
# Before
for system, results in consolidated.items():
    result_lines.append(f"  {system}:")
    for r in results[:5]:
        result_lines.append(f"    - {r.display} [{r.code}]")

# After
for i, r in enumerate(consolidated[:settings.flat_results], 1):
    result_lines.append(f"  {i}. [{r.system} {r.code}] {r.display}")
```

Human message results block example:
```
Final results:
  1. [ICD10CM I10] Essential (primary) hypertension
  2. [RXNORM 854901] lisinopril 20 MG Oral Tablet
  3. [LOINC 2823-3] Potassium in Serum or Plasma
```

### Summarizer system prompt update

```python
# Before
"- Do not repeat individual codes or list results by system — those are shown separately above."

# After
"- Do not repeat individual codes — they are shown above ranked by relevance to the query."
```

### Streamlit UI (`streamlit_app.py`)

Results display changes from one table per system to a single flat table:

| Rank | System | Code | Display |
|---|---|---|---|
| 1 | ICD10CM | I10 | Essential (primary) hypertension |
| 2 | RXNORM | 854901 | lisinopril 20 MG Oral Tablet |

The reasoning trace expander is **unchanged** — it iterates over `attempt_history`, which is unaffected.

---

## Section 5: Testing

### New tests — `tests/graph/test_nodes.py`

Replace all 8 `test_consolidator_*` tests with:

| Test | What it verifies |
|---|---|
| `test_re_ranker_empty_pool_returns_empty` | No systems selected or all filtered out → `[]`, no LLM call |
| `test_re_ranker_small_pool_skips_llm` | Pool ≤ `flat_results` → returns pool as-is, no LLM call |
| `test_re_ranker_calls_llm_for_large_pool` | Pool > `flat_results` → LLM called, top-N returned in order |
| `test_re_ranker_applies_domain_filter` | `relevant_codes` filter applied per system before pooling |
| `test_re_ranker_drops_invalid_llm_codes` | LLM returns code not in pool → silently dropped |
| `test_re_ranker_deduplicates_llm_output` | LLM returns duplicate code → first occurrence kept |

### New tests — `tests/graph/test_prompts.py`

| Test | What it verifies |
|---|---|
| `test_build_re_ranker_messages` | Query present, all candidate codes present with `[SYSTEM:CODE]` prefix, count shown |
| `test_build_re_ranker_messages_includes_flat_results_count` | Top-N count mentioned in human message |

### Existing tests to update

| File | Change |
|---|---|
| `tests/graph/test_nodes.py` | Remove all `test_consolidator_*`; add re_ranker tests above |
| `tests/graph/test_prompts.py` | `consolidated` arg in summarizer tests changes from `dict` to `list[CodeResult]` |
| `tests/graph/test_builder.py` | `NODE_CONSOLIDATOR` → `NODE_RE_RANKER` in all route assertions |

---

## Files Changed

| File | Change |
|---|---|
| `src/clinical_codes/config.py` | Add `flat_results`, `re_ranker_temperature`; remove `display_results`; rename `NODE_CONSOLIDATOR` → `NODE_RE_RANKER` |
| `src/clinical_codes/graph/state.py` | Add `RankedCode`, `ReRankerOutput`; change `consolidated` type to `list[CodeResult]` |
| `src/clinical_codes/graph/prompts.py` | Add `build_re_ranker_messages`, `_RE_RANKER_SYSTEM`; update `build_summarizer_messages` and `_SUMMARIZER_SYSTEM` |
| `src/clinical_codes/graph/nodes.py` | Replace `consolidator` with `re_ranker`; add `_re_ranker_chain` |
| `src/clinical_codes/graph/builder.py` | Replace `consolidator` node with `re_ranker`; update `route_after_planner` target |
| `src/clinical_codes/app/streamlit_app.py` | Update results display to flat table |
| `tests/graph/test_nodes.py` | Replace consolidator tests with re_ranker tests |
| `tests/graph/test_prompts.py` | Update summarizer tests; add re_ranker prompt tests |
| `tests/graph/test_builder.py` | Update `NODE_CONSOLIDATOR` → `NODE_RE_RANKER` references |

---

## Out of Scope

- Cross-system score normalization (codes are not compared against each other, only against the query)
- Storing the LLM's relevance score as a numeric field on `CodeResult` (ranking order is the signal)
- Streaming re_ranker output to the UI (same pattern as other LLM nodes)
