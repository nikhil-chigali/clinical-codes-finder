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

## 3. API rank score — position in result list, not semantic similarity

**Decision:** The `score` field on `CodeResult` is derived from the NLM API's own result ordering: `score = (total - rank) / total`, mapping position in the result list to [0, 1]. Rank 0 (top result) always gets 1.0; rank n-1 gets 1/n.

**Why this matters:** The score is **not** a semantic similarity score — it is purely a position score. If a system returns any results, its top result always gets 1.0 regardless of how well it matches the query. This means threshold-based quality checks ("all results below 0.5") are near-meaningless for non-empty result sets.

**What the score is used for:**
- **Consolidator** — orders results within each system to select the top `display_results` (5) for the final response. The API's ordering is a reasonable proxy for term relevance.
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

---

## 8. Metrics — `MetricsSummary` aggregate fields are non-optional (`float`, not `float | None`)

**Decision:** `MetricsSummary.top3_recall` and `must_include_hit_rate` are typed `float`, while the equivalent fields on `QueryTypeMetrics` are `float | None`.

**Why:** `MetricsSummary` is the top-level eval result reported to the caller. Allowing `None` there would require reporter.py to handle a case that is nearly impossible in practice (a run with zero non-miss queries and zero queries with `must_include`). The `float` type keeps the reporter simple.

**Convention:** `compute_metrics` substitutes `0.0` for `None` using `if x is not None else 0.0` at the overall level only. `QueryTypeMetrics` preserves `None` so the reporter can render `n/a` for the miss-type slice without special-casing.

**Caveat for reporter.py:** If all queries in a run are miss-type, `MetricsSummary.top3_recall` will be `0.0` rather than `n/a`. Reporter should display the overall top-3 recall row only when `n_total - count_of_miss_queries > 0`, which it can derive from `by_type`.

---

## 10. RxNorm two-step dose-string fallback

**Decision:** Override `RxNormClient.search()` to detect dose strings in the query, strip the dose to extract the drug name, retry the API with just the drug name, and expand the response into per-strength `CodeResult`s ranked so the matching dose appears first.

**Background:** The RxTerms v3 API does a prefix match on `DISPLAY_NAME`. Drug display names look like `"lisinopril (Oral Pill)"` — they never start with a dose string. Queries like `"lisinopril 20 mg"` or `"metformin 500 mg"` return zero results from the primary search. This is the root cause of the top-3 recall failure on dose-qualified drug queries.

The API is designed as a two-step UI flow: search by drug name → pick a strength from a dropdown. The strength data (`RXCUIS`, `STRENGTHS_AND_FORMS`) is already present in the first response as comma-separated values per row — it just isn't used by the original `_parse_response`.

**Why override `search()` rather than modifying `_parse_response`:**
- `_parse_response` is called after a successful fetch; the zero-result problem occurs before parsing — there is nothing to parse.
- Overriding `search()` allows a second conditional API call, which is beyond `_parse_response`'s single-fetch contract.
- The base class retry/error-isolation infrastructure (`_fetch_with_retry`) is reused directly — no new HTTP logic.

**Why per-strength expansion (`_parse_strengths`):**
- The API returns rows where each row represents one drug+form group with comma-separated CUIs and strengths (parallel arrays).
- The expected code for `"lisinopril 20 mg"` is a specific-strength CUI, not the generic group CUI returned by `_parse_response`.
- Expanding per-strength gives the evaluator and consolidator the right granularity to surface the exact dose match. Dose-matching entries are ranked first via a `matching` / `others` bucket sort keyed on `dose_norm in strength.lower()`.

**Display format:** `"lisinopril (Oral Pill) — 20 MG Tablet"` — drug group name plus specific strength, separated by an em dash. This is fully qualified and human-readable in the Streamlit UI and summarizer output.

**Trade-off:** A dose-string query now potentially makes two API calls (primary → empty, fallback). Acceptable — the fallback fires only when the primary returns empty AND a dose pattern is detected. Plain drug-name queries (`"lisinopril"`) take the unchanged one-call path.

---

## 11. Planner conservative selection defaults and miss-query catch

**Decision:** Replace the vague `"Select 1-3 systems"` rule in `_PLANNER_SYSTEM` with (a) a conservative default of 1 system, (b) per-domain anchors for common single-system query types, and (c) an explicit instruction to return empty selection for clearly non-clinical queries.

**Background:** Eval run `20260502_175037` showed system precision ≈ 0.33 on simple condition queries — the planner consistently selected 3 systems for queries like `"diabetes"`, `"hypertension"`, and `"CPAP machine"` where only one was expected. Separately, the query `"asdfghjkl"` (keyboard mash) caused the planner to select a coding system and consume two full refinement iterations. Prose non-clinical queries (`"weather forecast"`, `"how do I make pasta"`) already worked correctly.

**The changes:**

| Before | After |
|---|---|
| `"Select 1-3 systems. Select more only when the query genuinely spans multiple clinical domains."` | `"Default to 1 system. Add a second only when the query explicitly spans two distinct clinical domains; add a third only for genuinely complex multi-domain queries."` |
| *(no domain anchors)* | Bare disease/symptom → ICD-10CM; drug → RxNorm; lab test → LOINC; device → HCPCS; unit → UCUM |
| *(no miss-query instruction)* | `"If the query is clearly not a clinical term — random characters, keyboard mash, or non-medical questions — return an empty system selection."` |

**Why not change the evaluator or graph:**
- The graph handles `selected_systems = []` correctly today (proven by q030/q031 in the eval — both return 1.0 system F1 with 0 API calls).
- Modifying the evaluator for the empty-selection case adds constraints that could have side effects on normal query paths.
- The planner prompt is the correct intervention point: it's where selection decisions are made.

**Why no HPO domain anchor:**
The HPO vs. ICD-10CM distinction (e.g., `"ataxia"` as a rare-disease phenotype vs. a billable ICD-10 condition) is too subtle for a fixed rule. A blanket HPO anchor would risk under-selection for ambiguous phenotype queries. HPO selection is left to LLM judgment.

**Trade-off:** Domain anchors reduce flexibility — `"diabetes"` will always route to ICD-10CM only, even in a context where RxNorm diabetes drugs are more relevant. This is acceptable: the system is designed for natural-language queries, and bare disease names almost universally map to diagnostic codes. Multi-domain queries like `"diabetes medication"` still route freely because they contain explicit signals from two domains.

---

## 12. Scaling beyond 6 systems

The current implementation embeds system descriptions directly in the planner prompt — appropriate for 6 fixed systems. Beyond ~15–20 systems this would become untenable: context cost grows linearly per query, the planner's attention across many options degrades, and onboarding a new system requires editing a central prompt.

The natural evolution is embedding-based pre-routing: each system ships with a self-contained manifest (description, when-to-use, examples), embedded once at startup. The planner LLM sees only the top-k candidates relevant to a given query, with the full catalog available but not always loaded. Adding a system becomes a file drop, not a prompt edit.

Past ~30–50 systems the routing itself is worth replacing with a learned classifier trained on production query-system pairs, which removes the LLM from the routing path entirely and reserves it for query-term generation within already-selected systems. Hierarchical routing (category → system) becomes worth considering at the scale of UMLS-style integration with hundreds of vocabularies.

None of this is built in the prototype because the constraints lock the system count at 6. But the codebase is structured to support the migration: tools already live in per-system modules, and the planner's catalog block is the single place that would be replaced by a registry lookup.
