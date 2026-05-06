# Design Decisions

Captures the key architectural decisions made during design sessions, with rationale and trade-offs. Intended as a reference for demos and future contributors.

---

## 1. API rank score — position in result list, not semantic similarity

**Decision:** The `score` field on `CodeResult` is derived from the NLM API's own result ordering: `score = (total - rank) / total`, mapping position in the result list to [0, 1]. Rank 0 (top result) always gets 1.0; rank n-1 gets 1/n.

**Why this matters:** The score is **not** a semantic similarity score — it is purely a position score. If a system returns any results, its top result always gets 1.0 regardless of how well it matches the query. This means threshold-based quality checks ("all results below 0.5") are near-meaningless for non-empty result sets.

**What the score is used for:**
- **Re_ranker pool ordering** — within a single system's contribution to the pool, API order is preserved as a tiebreak. The LLM then re-ranks the full cross-system pool by query relevance, superseding positional score entirely.
- **Not used by the evaluator** — the evaluator performs semantic relevance judgment instead (see §2).

**`confidence_threshold` in `config.py`:** Currently vestigial. It was designed for a threshold-based evaluator check that was superseded by semantic evaluation. Retained in config for potential future use.

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

**Evaluator human message format:** Top 5 results (code + display) per system. The LLM reads display names to assess semantic relevance — this is where it adds value that a threshold check cannot. On iteration 2+, the evaluator also receives a "carried over" block showing accumulated results for systems that returned strong results in a prior iteration and were not re-queried. Carried-over results are shown so the evaluator can populate `relevant_codes` for those systems and form an accurate overall picture — they are not re-evaluated for quality (see §12).

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

## 5. Summarizer — audience, guidelines, and cap-hit callout

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
- Expanding per-strength gives the evaluator and re_ranker the right granularity to surface the exact dose match. Dose-matching entries are ranked first via a `matching` / `others` bucket sort keyed on `dose_norm in strength.lower()`.

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

## 8. Evaluator semantic filtering — clinical domain, not sub-type specificity

**Decision:** The evaluator populates `relevant_codes: dict[SystemName, list[str]]` on every pass (sufficient and refine), listing only codes that belong to the correct clinical domain for the query. The re_ranker applies this filter before pooling.

**The clinical domain standard:** Filter results that are from a fundamentally different clinical category — not results that represent the same entity through a different method, specimen type, or sub-classification. The API's own ranking handles relevance within a domain; the evaluator's job is to catch cross-domain mismatches.

Cross-domain mismatches (filter):
- Query `"metformin 500 mg"` → a LOINC plasma metformin level panel does **not** match — it is a lab measurement, not a drug formulation. RxNorm drug formulation codes **do** match.
- Query `"hypertension"` → ICD-10-CM I10 (primary hypertension) **matches**; I51.9 (unspecified heart disease, a different condition) does **not**.

Within-domain variation (keep — trust the API):
- Query `"ecoli"` against LOINC → FISH assays, blood culture assays, and urine culture assays **all match** — they are all E. coli lab tests. The evaluator does not choose between specimen types or test methods; that sub-type distinction is beyond its role.

**Why the earlier "urine culture only" interpretation was wrong:** A prior version of the evaluator prompt used `"ecoli 10000"` as an example, treating only urine culture codes as relevant and discarding FISH assays. This was over-interpretation — the query is ambiguous about specimen type, and the API legitimately returns any E. coli lab test for the search term "ecoli". Filtering within a correctly matched domain assumes clinical specificity the system cannot reliably infer.

**Why filter in the evaluator, not a separate node:** The evaluator already reads every result display name to make its sufficiency decision. Filtering is a natural extension of the same judgment — no additional LLM call, no new node. The evaluator's `relevant_codes` output costs a handful of extra tokens in the structured response.

**Why populate on refine, not only on sufficient:** If the iteration cap fires and the pipeline is forced forward on a "refine" decision, `relevant_codes` is already populated with the best available filtered set. Without this, all raw results — including those the evaluator judged as domain-mismatches — would flow through unfiltered.

**Empty list vs absent key in `relevant_codes`:**
- Key absent → system had no raw results; re_ranker skips filter (nothing to filter).
- Empty list `[]` → system had results but all were domain-mismatches; re_ranker excludes all results for that system from the pool.
- Non-empty list → re_ranker keeps only those codes in the pool, discards the rest.

This distinction is enforced with `if keep is not None` in the re_ranker (not `if keep`), so an empty list correctly removes all results rather than being treated as "no filter."

---

## 9. Scaling beyond 6 systems

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

---

## 10. Miss-query short-circuit — planner → re_ranker when no systems selected

**Decision:** Add a conditional edge from the planner: if `selected_systems` is empty, route directly to the re_ranker, bypassing the executor and evaluator entirely.

**Background:** The planner prompt already instructs the model to return an empty system selection for gibberish, keyboard mash, or clearly non-clinical inputs, and state this in the rationale. Before this change, the graph ignored the empty selection and still ran the executor (no-op, since there are no search terms) and the evaluator. The evaluator, faced with zero results across zero systems, had nothing to evaluate — yet consistently returned `"refine"` with feedback like "there is nothing to evaluate." This triggered a second planner call, which again returned empty, burning the iteration cap before the pipeline could exit cleanly.

**Why the evaluator returned "refine":** The evaluator is prompted to return "sufficient" only when every selected system returned at least one result that matches clinical intent. With zero selected systems and zero results, neither condition is satisfied — so "refine" is technically correct by its own prompt logic. Fixing the evaluator prompt to handle this edge case is fragile; short-circuiting in the graph is the robust solution.

**Implementation:** `route_after_planner` in `builder.py` checks `state["planner_output"].selected_systems`. Empty → `NODE_RE_RANKER`; non-empty → `"executor"`. The re_ranker builds an empty pool (no selected systems), returning `consolidated = []`. The summarizer receives an empty result set and its "no results" prompt guideline handles the rest.

**Cost:** A miss query now costs exactly one LLM call (planner only), down from three (planner × 2 + evaluator × 2 under the old cap). No executor HTTP calls, no evaluator calls.

**Trade-off:** `attempt_history` is empty for miss queries (the evaluator is what appends `Attempt` records). The reasoning trace in the UI shows "0 iterations" — acceptable, since there was no search to trace.

---

## 11. Re_ranker — LLM cross-system relevance ranking replaces deterministic consolidator

**Decision:** Replace the deterministic `consolidator` (dedup, group by system, rank by API position) with an LLM-based `re_ranker` that pools domain-filtered codes from all systems and returns a flat top-N list ordered by query relevance.

**The problem with API rank position:** The API's positional score (rank 0 always = 1.0) is a tiebreak within one system, not a semantic similarity score. A marginally relevant code that ranked first in LOINC would always appear before a highly specific match that ranked third in RxNorm. There was no mechanism to compare codes against the original user query, and results were grouped by system rather than by relevance.

**Why flat output over grouped-by-system:** Grouping by system was an organizing convenience, not a clinical requirement. A user querying "lisinopril 20 mg" cares whether the top result is the specific-strength RxNorm code — not which system the second and third results come from. A flat relevance-ordered list answers the actual question: "what is most relevant to my query?"

**Why an LLM for ranking (not score normalization):** Cross-system score normalization would be false precision — a 1.0 from LOINC and a 1.0 from UCUM both mean "top result in that API's response" with no common semantic scale. An LLM can apply the same clinical reasoning the user would: it prefers "lisinopril 20 MG Oral Tablet" over "lisinopril (Oral Pill)" for the query "lisinopril 20 mg" because specificity matters here.

**Short-circuit paths (no LLM cost):**
- Empty pool (all systems filtered out, or no systems selected) → return `[]` immediately.
- Pool ≤ `flat_results` (5) → return pool as-is in API order, no ranking needed.

**`consolidated` state field:** type changed from `dict[SystemName, list[CodeResult]]` to `list[CodeResult]`. System is still a field on each `CodeResult`, so all downstream consumers (summarizer, UI) access it via `r.system` — no grouped dict needed.

**Trade-off:** The re_ranker adds one LLM call per query when the pool is larger than `flat_results`. This is acceptable: it fires only after the evaluator has already made a sufficient/refine decision (1–2 LLM calls), and it replaces a deterministic step that provided no query-relevance signal. The short-circuit paths keep miss queries and simple single-result queries at zero additional cost.

---

## 12. Evaluator context — carried-over systems on refinement iterations

**Decision:** On iteration 2+, `build_evaluator_messages` appends a "Already sufficient — not re-queried this iteration" block to the human message, showing the top-5 accumulated results for every system in `selected_systems` that was omitted from the current `search_terms`. The evaluator prompt instructs the model to count these results as established coverage without re-evaluating their quality.

**The problem:** When the planner omits a strong system from `search_terms` on iteration 2 (because its results were sufficient in iteration 1), `build_evaluator_messages` previously only iterated over `search_terms` — leaving the evaluator with no visibility into that system's accumulated results. The evaluator then returned "refine" because it had an incomplete picture of what was available. Concrete cases: "tuberculosis treatment" (ICD-10-CM results already strong, omitted from iter 2's `search_terms` → evaluator wrongly concluded the disease component was unaddressed), "anemia workup" (same pattern).

**Why this is a prompt-context problem, not a state problem:** `state["raw_results"]` already contains accumulated results from all iterations (the executor merges with `dict(state["raw_results"])` each pass). The data was present; the evaluator prompt simply wasn't showing it.

**Why not just pass all `raw_results` without the "carried over" label:** The evaluator needs to distinguish current-iteration results (which it must evaluate for quality) from prior-iteration results (which are established and should only count for coverage). Merging them without a label would cause the evaluator to re-evaluate the quality of carried-over results, potentially downgrading them and triggering another false refine.

**Effect:** Queries like "tuberculosis treatment" and "anemia workup" now exit iteration 2 with "sufficient" instead of triggering an unnecessary refinement. Mean API calls across the gold set: 1.68 → 1.55.

**The `relevant_codes` output for carried-over systems:** The evaluator populates `relevant_codes` for carried-over systems using the same domain standard as current-iteration systems. The re_ranker gates its pool on `relevant_codes` (not `selected_systems`), so carried-over systems' results are included in the final ranking even when the planner omits them from `selected_systems` on the second iteration.

**Coverage check removal:** A coverage check was initially part of the evaluator — it would trigger refinement if any meaningful query component was unrepresented across all selected systems. This was removed after it caused false-positive refines on queries where the planner made a deliberate, correct system selection (e.g., ICD-10-CM only for "type 2 diabetes" — the planner intentionally chose one system, but the evaluator flagged missing drug and lab coverage). The planner already owns system selection; the evaluator's role is to judge result quality only. Cases where the planner genuinely missed a domain are caught by the result quality check — empty results or off-domain results trigger refinement naturally.
