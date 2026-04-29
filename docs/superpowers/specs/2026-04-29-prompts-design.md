# Prompts Design

**Date:** 2026-04-29
**Scope:** `src/clinical_codes/graph/prompts.py` — all LLM prompt templates and message-building functions

---

## Context

Three nodes in the LangGraph pipeline make LLM calls: `planner`, `evaluator`, and `summarizer`. All prompt logic is centralized in `prompts.py`. Node bodies import a formatting function, call it with state fields, and pipe the result directly to the model chain.

LLM calls use `langchain-anthropic`'s `ChatAnthropic` with `.with_structured_output()` for the planner and evaluator (Pydantic models out directly), and plain string output for the summarizer. Model assignments and temperatures live in `config.py` and are consumed by the node files — `prompts.py` has no knowledge of model configuration.

---

## File layout

Everything in `src/clinical_codes/graph/prompts.py`:

- `SYSTEM_CATALOG: dict[SystemName, str]` — module-level constant, one description string per system
- `build_planner_messages(query: str, attempt_history: list[Attempt]) -> list[BaseMessage]`
- `build_evaluator_messages(query: str, planner_output: PlannerOutput, raw_results: dict[SystemName, list[CodeResult]]) -> list[BaseMessage]`
- `build_summarizer_messages(query: str, consolidated: dict[SystemName, list[CodeResult]], rationale: str) -> list[BaseMessage]`

No other public symbols. All helper formatting is private to the module.

---

## `SYSTEM_CATALOG`

A `dict[SystemName, str]` mapping each of the six systems to a one-line description. Descriptions are intentionally terse in v1 — they will be manually elaborated over time without code changes.

```python
SYSTEM_CATALOG: dict[SystemName, str] = {
    SystemName.ICD10CM:  "Diagnosis and condition codes. Use for diseases, symptoms, injuries, and clinical conditions.",
    SystemName.LOINC:    "Lab tests and clinical observations. Use for measurements, panels, and diagnostic procedures.",
    SystemName.RXNORM:   "Drug names and medications. Use for drugs, dosage forms, and active ingredients.",
    SystemName.HCPCS:    "Procedures, devices, and supplies billed to Medicare/Medicaid. Use for equipment, therapies, and clinical services.",
    SystemName.UCUM:     "Units of measure. Use for measurement units such as mg/dL, mmol/L, or beats per minute.",
    SystemName.HPO:      "Human phenotype terms. Use for genetic traits, rare disease features, and clinical phenotypes.",
}
```

---

## `build_planner_messages`

### Signature

```python
def build_planner_messages(query: str, attempt_history: list[Attempt]) -> list[BaseMessage]:
```

Returns `[SystemMessage, HumanMessage]`. Branches on `attempt_history`: empty → first pass, non-empty → refinement.

### System message content

```
You are a clinical coding specialist. Given a natural-language clinical query, select the most relevant
medical coding systems and generate a precise search term for each.

Available systems:
{SYSTEM_CATALOG serialized as "  SYSTEM_NAME: description" lines}

Selection rules:
- Select 1–3 systems. Select more only when the query genuinely spans multiple clinical domains.
- Generate exactly one search term per selected system.
- Use standardized clinical vocabulary — prefer terms the NLM Clinical Tables API recognizes
  over colloquial or abbreviated forms.

On refinement:
- You will receive the prior attempt's search terms, weak systems, and the evaluator's diagnosis.
- Based on the diagnosis, you may: retry a weak system with a different search term, drop a weak
  system that does not cover this query type, or add a system not in the original selection if the
  diagnosis suggests the query spans a different domain.
- Systems that returned strong results do not need to be re-queried; omit them from search_terms.
```

### Human message — first pass (`attempt_history == []`)

```
Query: {query}
```

### Human message — refinement (`attempt_history` non-empty, read `attempt_history[-1]`)

```
Query: {query}

Prior attempt:
  Systems queried: {search_terms}
  Weak systems: {weak_systems}
  Evaluator feedback: {feedback}

Revise your system selection and/or search terms based on the evaluator's feedback.
Systems that returned strong results do not need to be re-queried.
```

Where:
- `search_terms` — `attempt_history[-1].planner_output.search_terms` serialized as `SYSTEM: "term"` pairs
- `weak_systems` — `attempt_history[-1].evaluator_output.weak_systems` as a comma-separated list
- `feedback` — `attempt_history[-1].evaluator_output.feedback`

### Output type

`.with_structured_output(PlannerOutput)` — returns `PlannerOutput` directly.

### Note on `selected_systems` mutability

`selected_systems` in `PlannerOutput` is the planner's per-iteration selection. It may differ between iterations — the planner may add or drop systems on refinement. This supersedes the original state spec's "stable across iterations" note. The `Attempt` model preserves each iteration's full `planner_output` in `attempt_history`.

---

## `build_evaluator_messages`

### Signature

```python
def build_evaluator_messages(
    query: str,
    planner_output: PlannerOutput,
    raw_results: dict[SystemName, list[CodeResult]],
) -> list[BaseMessage]:
```

Returns `[SystemMessage, HumanMessage]`.

### System message content

```
You are a clinical code quality evaluator. Given a clinical query and the results returned for each
selected coding system, decide whether the results are sufficient or require refinement.

Evaluation criteria:
- sufficient: every selected system returned at least one result that appears semantically relevant
  to the query.
- refine: any selected system returned no results, or its results do not appear relevant to the
  query (e.g., a drug query against LOINC returns imaging codes).

For each weak system, provide a plain-English diagnosis explaining why the results are weak.
Do NOT prescribe remediation — do not suggest alternative search terms or systems.
Describe what went wrong; the planner will decide how to address it.

If decision is "sufficient", weak_systems must be empty and feedback must be an empty string.
```

### Human message

The top 5 display names per selected system are included. Score and code values are omitted — the evaluator reasons from display names, not numeric scores.

```
Query: {query}
Selected systems and search terms: {search_terms}

Results:
  {SYSTEM_NAME} (searched: "{term}"):
    1. {display}
    2. {display}
    ...   ← up to 5 results; "(no results)" if empty
```

Serialization: iterate over `planner_output.search_terms` in insertion order; for each system, take `raw_results.get(system, [])[:5]`.

### Output type

`.with_structured_output(EvaluatorOutput)` — returns `EvaluatorOutput` directly.

---

## `build_summarizer_messages`

### Signature

```python
def build_summarizer_messages(
    query: str,
    consolidated: dict[SystemName, list[CodeResult]],
    rationale: str,
) -> list[BaseMessage]:
```

Returns `[SystemMessage, HumanMessage]`.

### System message content

```
You are a clinical information specialist. Write a clear, plain-English summary of the medical codes
found for the given query. Your audience may be non-technical — a patient, student, or general
clinician.

Guidelines:
- For each system, write one short paragraph: what the system covers, what was found, and why it
  was included.
- Refer to results by display name. Include the code in brackets where clinically meaningful
  (e.g., ICD-10-CM and RxNorm codes are commonly referenced; UCUM units speak for themselves).
- Define any medical term you use.
- Do not mention systems that were not selected.
- If no results were found across any system, return a polite refusal and suggest the user rephrase
  using a recognized clinical term.
```

### Human message

```
Query: {query}

Why these systems were selected: {rationale}

Results:
  {SYSTEM_NAME}:
    - {display} [{code}]
    - {display} [{code}]
    ...   ← top 5 per system from consolidated
```

Serialization: iterate over `consolidated` items; for each system, list `result.display` and `result.code` for up to 5 results.

### Output type

Plain string — no `.with_structured_output()`. The summarizer returns prose.

---

## Testing

Test file: `tests/graph/test_prompts.py`

| Test | What it checks |
|---|---|
| `test_system_catalog_complete` | All 6 `SystemName` values present in `SYSTEM_CATALOG` |
| `test_build_planner_first_pass` | Empty `attempt_history` → 2 messages; human message contains query; no "Prior attempt" block |
| `test_build_planner_refinement` | Non-empty `attempt_history` → human message contains search terms, weak systems, feedback |
| `test_build_evaluator_messages` | Returns 2 messages; human message contains query, search terms, display names |
| `test_evaluator_empty_results` | System with no results renders as "(no results)" |
| `test_evaluator_truncates_to_five` | Systems with >5 results show only 5 display names |
| `test_build_summarizer_messages` | Returns 2 messages; human message contains rationale and result display names |

---

## Dependencies

Add to `pyproject.toml`:
```
langchain-anthropic>=0.3
langchain-core>=0.3
```

`langchain-core` provides `BaseMessage`, `SystemMessage`, `HumanMessage`. `langchain-anthropic` provides `ChatAnthropic`. Both are needed by node files (not by `prompts.py` itself, which only uses `langchain-core` types).

---

## Files changed by this spec

| File | Change |
|---|---|
| `src/clinical_codes/graph/prompts.py` | Create |
| `tests/graph/test_prompts.py` | Create |
| `pyproject.toml` | Add `langchain-anthropic`, `langchain-core` |
| `src/clinical_codes/graph/state.py` | Update `selected_systems` docstring (mutable, not stable) |
| `src/clinical_codes/config.py` | Add comment: `confidence_threshold` is vestigial for the evaluator |
