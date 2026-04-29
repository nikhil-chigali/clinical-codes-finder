# Design Decisions

Captures the key architectural decisions made during design sessions, with rationale and trade-offs. Intended as a reference for demos and future contributors.

---

## 1. LLM framework — `langchain-anthropic` with `.with_structured_output()`

**Decision:** Use `langchain-anthropic`'s `ChatAnthropic` with `.with_structured_output(PlannerOutput)` / `.with_structured_output(EvaluatorOutput)` rather than the bare Anthropic SDK or LangGraph's built-in model bindings.

**Why:**
- Eliminates manual JSON parsing for structured outputs — typed Pydantic models come back directly.
- Integrates naturally with LangGraph node return types.
- The bare SDK requires tool-use boilerplate or manual JSON extraction; `with_structured_output` handles this transparently.

**Trade-off:** Adds `langchain-anthropic` as a dependency. Accepted — LangGraph already presupposes this ecosystem.

---

## 2. Prompt module design — formatting functions, not template objects

**Decision:** `graph/prompts.py` exports three formatting functions (`build_planner_messages`, `build_evaluator_messages`, `build_summarizer_messages`), each returning `list[BaseMessage]`. No `ChatPromptTemplate` objects, no raw string constants.

**Why:**
- The planner's message shape differs between first pass and refinement (different human message content). A function handles this branching cleanly; a template object requires two templates and caller-side selection logic.
- Node bodies stay clean — they call a function and pipe the result to the model chain.
- Functions are straightforward to unit test (call with args, assert on returned messages).

**Trade-off:** Loses LangChain's template validation. Acceptable — the functions are simple Python, easier to reason about than template DSL.

---

## 3. Confidence score — rank-derived, not semantic similarity

**Decision:** The `score` field on `CodeResult` is derived from the NLM API's own result ordering: `score = (total - rank) / total`, mapping position in the result list to [0, 1]. Rank 0 (top result) always gets 1.0; rank n-1 gets 1/n.

**Why this matters:** The score is **not** a semantic similarity score. If a system returns any results, its top result always gets 1.0 regardless of how well it matches the query. This means threshold-based quality checks ("all results below 0.5") are near-meaningless for non-empty result sets.

**What the score is used for:**
- **Consolidator** — ranks results within each system to select the top `display_results` (5) for the final response. The API's ordering is a reasonable proxy for term relevance.
- **Not used by the evaluator** — the evaluator performs semantic relevance judgment instead (see §4).

**`confidence_threshold` in `config.py`:** Currently vestigial. It was designed for a threshold-based evaluator check that was superseded by semantic evaluation. Retained in config for potential future use (e.g., a consolidator filter).

---

## 4. Evaluator design — diagnose, don't prescribe

**Decision:** The evaluator identifies weak systems and explains *why* results are weak, but does **not** prescribe remediation (different search terms, alternate systems). Remediation is the planner's responsibility.

**Why:**
- The evaluator doesn't need clinical domain knowledge to assess quality; it only needs to recognize when returned results don't match the query domain.
- Prescribing remediation requires knowing the clinical vocabulary of alternative terms and which systems cover which domains — exactly what the planner system prompt is designed for.
- Cleaner separation of concerns: evaluator = quality judge, planner = clinical reasoning.

**What "weak" means in the evaluator:**
1. A selected system returned zero results.
2. A selected system returned results that, on semantic inspection, don't appear relevant to the query (e.g., a drug query against LOINC returns imaging panel codes).

**Evaluator human message format:** Top 5 result display names per system (not counts, not scores). The LLM reads display names to assess semantic relevance — this is where it adds value that a threshold check cannot.

**Evaluator `feedback` field:** Plain-English diagnosis per weak system. Examples:
- "LOINC returned no results for 'metformin' — this system covers lab tests, not drug names."
- "ICD-10-CM results are all hypertensive complications, not the primary hypertension condition the query describes."

---

## 5. Planner refinement — options and `selected_systems` mutability

**Decision:** On refinement, the planner may:
1. Retry a weak system with a different search term.
2. Drop a weak system that the evaluator diagnosed as wrong for this query type.
3. Add a new system not in the original selection, if the evaluator's diagnosis suggests the query maps to a different domain.

**Why `selected_systems` must be mutable:** Allowing drops already makes `selected_systems` mutable. Consistency requires also allowing additions — otherwise the planner can correct a wrong exclusion but not a wrong inclusion. A planner that can fully reconsider both selections and terms is more useful than one that can only narrow.

**State design implication:** `selected_systems` in `PlannerOutput` and `GraphState` is the planner's per-iteration selection, not a stable field. The `Attempt` model captures each iteration's full `planner_output`, so the selection history is preserved in `attempt_history`. The executor and evaluator read from the current `planner_output`; the summarizer reads from the final one.

**Note:** This supersedes the original state spec's claim that `selected_systems` "does not change between iterations." The state.py docstring will be updated accordingly.

---

## 6. Refinement context — summary, not full raw results

**Decision:** The planner's refinement human message includes prior search terms, weak system names, and evaluator feedback prose — but **not** the full raw results from the prior iteration.

**Why:** Raw results (up to 10 codes per system) are noisy and token-expensive. The evaluator's feedback already distills what went wrong. The planner needs to know which systems were tried and what the diagnosis was — not the full code list.

**Planner refinement human message shape:**
```
Query: {query}

Prior attempt:
  Systems queried: {search_terms}   ← dict of system → search term
  Weak systems: {weak_systems}      ← list of system names
  Evaluator feedback: {feedback}    ← prose diagnosis

Revise your system selection and/or search terms based on the evaluator's feedback.
Systems that returned strong results do not need to be re-queried.
```

---

## 7. Summarizer — audience and guidelines

**Decision:** The summarizer writes for a **non-technical audience** (patient, general clinician, or student). It does not surface which systems were excluded by the planner.

**Content per selected system:**
- System label and one-line description of what it covers.
- Top results by display name (not raw code strings unless clinically meaningful).
- Why this system was selected (drawn from planner's `rationale` field).

**Tone:** Plain English. Explain medical terms when they appear. No jargon without definition.

**Source:** Per `scope.md` — "plain-English explanation" and "for a potentially non-technical audience."
