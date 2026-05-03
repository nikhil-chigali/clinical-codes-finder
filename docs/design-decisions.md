# Design Decisions

Captures the key architectural decisions made during design sessions, with rationale and trade-offs. Intended as a reference for demos and future contributors.

---

## 1. API rank score — position in result list, not semantic similarity

**Decision:** The `score` field on `CodeResult` is derived from the NLM API's own result ordering: `score = (total - rank) / total`, mapping position in the result list to [0, 1]. Rank 0 (top result) always gets 1.0; rank n-1 gets 1/n.

**Why this matters:** The score is **not** a semantic similarity score — it is purely a position score. If a system returns any results, its top result always gets 1.0 regardless of how well it matches the query. This means threshold-based quality checks ("all results below 0.5") are near-meaningless for non-empty result sets.

**What the score is used for:**
- **Consolidator** — orders results within each system to select the top `display_results` (5) for the final response. The API's ordering is a reasonable proxy for term relevance.
- **Not used by the evaluator** — the evaluator performs semantic relevance judgment instead (see §2).

**`confidence_threshold` in `config.py`:** Currently vestigial. It was designed for a threshold-based evaluator check that was superseded by semantic evaluation. Retained in config for potential future use (e.g., a consolidator filter).

---

## 2. Evaluator design — diagnose, don't prescribe

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

## 3. Planner refinement — options and `selected_systems` mutability

**Decision:** On refinement, the planner may:
1. Retry a weak system with a different search term.
2. Drop a weak system that the evaluator diagnosed as wrong for this query type.
3. Add a new system not in the original selection, if the evaluator's diagnosis suggests the query maps to a different domain.

**Why `selected_systems` must be mutable:** Allowing drops already makes `selected_systems` mutable. Consistency requires also allowing additions — otherwise the planner can correct a wrong exclusion but not a wrong inclusion. A planner that can fully reconsider both selections and terms is more useful than one that can only narrow.

**State design implication:** `selected_systems` in `PlannerOutput` and `GraphState` is the planner's per-iteration selection, not a stable field. The `Attempt` model captures each iteration's full `planner_output`, so the selection history is preserved in `attempt_history`. The executor and evaluator read from the current `planner_output`; the summarizer reads from the final one.

---

## 4. Refinement context — summary, not full raw results

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

## 5. Summarizer — audience and guidelines

**Decision:** The summarizer writes for a **non-technical audience** (patient, general clinician, or student). It does not surface which systems were excluded by the planner.

**Content per selected system:**
- System label and one-line description of what it covers.
- Top results by display name (not raw code strings unless clinically meaningful).
- Why this system was selected (drawn from planner's `rationale` field).

**Tone:** Plain English. Explain medical terms when they appear. No jargon without definition.

---

## 6. RxNorm two-step dose-string fallback

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

## 7. Planner selection defaults, domain anchors, and multi-domain trigger

**Decision:** Replaced the vague `"Select 1-3 systems"` rule with a conservative default, per-domain anchors for all 6 systems, an explicit miss-query catch, and a softened multi-domain trigger with concrete examples.

**Background:** Two failures drove this:
1. System precision ≈ 0.33 on simple queries — the planner consistently selected 3 systems for `"diabetes"`, `"hypertension"`, and `"CPAP machine"` where only 1 was expected.
2. Keyboard mash (`"asdfghjkl"`) caused the planner to select a coding system and consume 2 full refinement iterations instead of returning empty.

**The changes:**

| Rule | Before | After |
|---|---|---|
| System count | `"Select 1-3 systems. Select more only when the query genuinely spans multiple clinical domains."` | `"Default to 1 system. Add a second when the query spans two distinct clinical domains (e.g. 'diabetes medication' → ICD-10CM + RxNorm); add a third only when three distinct domains are clearly involved."` |
| Domain anchors | None | Bare disease → ICD-10CM; phenotypic trait → HPO; drug → RxNorm; lab test → LOINC; device → HCPCS; unit → UCUM |
| Non-clinical queries | No instruction | Return empty system selection and state this in the rationale |

**Why HPO is included as a domain anchor:** The original ICD-10CM anchor included the word "symptom," causing phenotypic terms like `"ataxia"` to route to ICD-10CM instead of HPO. Scoping ICD-10CM strictly to disease names and adding an explicit HPO anchor for phenotypic traits and rare-disease characteristics resolved the routing.

**Why the multi-domain trigger was softened:** An earlier version used `"explicitly spans"` as the gating phrase, which caused real-world over-conservatism — the planner treated most queries as single-domain even when they weren't. Removing `"explicitly"` and adding concrete examples (e.g., `"diabetes medication"` → ICD-10CM + RxNorm) gave the LLM a pattern to match rather than a vague threshold to interpret.

**Trade-off:** Domain anchors reduce routing flexibility for bare single-domain queries — `"diabetes"` always routes to ICD-10CM only. This is acceptable: bare disease names almost universally map to diagnostic codes. Multi-domain queries like `"diabetes medication"` still route to multiple systems because they contain signals from two domains.

---

## 8. Scaling beyond 6 systems

The current implementation embeds system descriptions directly in the planner prompt. This is appropriate for a fixed catalog of 6, but breaks down at scale.

**~15–20 systems — embedding-based pre-routing:**
- Each system ships with a self-contained manifest (description, when-to-use, examples), embedded once at startup.
- The planner sees only the top-k candidates relevant to the query; the full catalog is available but not always in context.
- Adding a new system becomes a file drop, not a prompt edit.

**~30–50 systems — learned classifier:**
- Replace the LLM router with a classifier trained on production query-system pairs.
- Removes the LLM from the routing path entirely; it is reserved for query-term generation within already-selected systems.
- More consistent and cheaper at scale than prompting.

**Beyond ~50 systems — hierarchical routing:**
- Category-level routing (e.g., clinical → diagnostic, administrative, pharmaceutical) before system-level routing.
- Worth considering at UMLS-style integration scale with hundreds of vocabularies.

**Codebase readiness:** Tools already live in per-system modules, and the planner's catalog block is the single place that would be replaced by a registry lookup. None of this is built in the prototype — the system count is fixed at 6.
