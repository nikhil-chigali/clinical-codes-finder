# Scope — Phase 0

Defines what is built, what is not, and every decision that would otherwise require re-discussion mid-build.

---

## Objective

Accept a single natural-language clinical query, infer which of six medical coding systems are relevant, search them in parallel, refine once if results are weak, and return matching codes with a plain-English explanation.

---

## Architecture constraints (locked)

| Constraint | Value |
|---|---|
| Graph nodes | 5 (planner → executor → evaluator → re_ranker → summarizer). Short-circuit: planner → re_ranker when `selected_systems` is empty (miss queries skip executor + evaluator entirely). |
| Query turns | Single-turn only. No conversational follow-up; state resets per query. |
| Coding systems | ICD-10-CM, LOINC, RxNorm, HCPCS, UCUM, HPO |
| Refinement cap | 2 iterations maximum, enforced in graph state (not prompted) |
| Data source | NLM Clinical Tables API (public, no auth) |
| Systems selected per query | 1–3 as soft guidance in the planner prompt, not a hard schema constraint. Planner may return 4 when genuinely necessary (e.g. "diabetes management" can defensibly touch ICD-10-CM + LOINC + RxNorm + HCPCS). Cap will be revisited after running the gold set — if the planner rarely exceeds 3, keep it; if it regularly wants 4, adjust. |
| Queries per selected system | 1 per system per planner call. Trades first-pass recall for simplicity; the refinement loop is the recovery path for a bad initial term. Watch "mean iterations / query" sliced by query type: if `ambiguous` queries average ≥ 1.8 iterations, single-query-per-system is the likely cause and two queries per system becomes worth the added complexity. |
| Language | English only |

---

## LLM configuration

### Model assignments

| Node | Model | Justification |
|---|---|---|
| `planner` | claude-sonnet-4-6 | Jointly selects systems and generates search terms in one call — both decisions require clinical vocabulary reasoning (e.g. knowing "metformin" maps to RxNorm drug names but "glucose lab" maps to LOINC panel names). Term quality directly gates result quality, and on refinement the model must re-evaluate its own prior system selection, not just retry terms. Haiku is insufficient for this combined reasoning task. |
| `evaluator` | claude-sonnet-4-6 | Makes a quality judgment that determines whether refinement fires. A wrong call is costly: false "sufficient" exits with bad results; false "refine" wastes an iteration. The 2-iteration cap means there is no safety net for a weak evaluator. |
| `re_ranker` | claude-sonnet-4-6 | Requires clinical vocabulary reasoning to rank by query relevance — e.g. knowing "lisinopril 20 MG Oral Tablet" is more specific than "lisinopril (Oral Pill)" for the query "lisinopril 20 mg". |
| `summarizer` | claude-sonnet-4-6 | Synthesizes codes across systems into coherent prose for a potentially non-technical audience. Quality of language matters here. |

### Temperature

| Node(s) | Temperature | Justification |
|---|---|---|
| `planner`, `evaluator`, `re_ranker` | `0` | Deterministic output required for reproducible evaluation runs. Variation in system selection, quality judgments, or ranking order would make eval metrics meaningless across runs. |
| `summarizer` | `0.3` | Slightly warmer for more natural prose. Still constrained enough to stay factual; loose enough to avoid robotic repetition when summarizing similar result sets. |

---

## API layer

### Results per system

| Parameter | Value | Justification |
|---|---|---|
| Fetch (per system, per call) | 10 results | Gives the evaluator enough signal to assess result quality. Beyond 10, marginal results are unlikely to change the refine/sufficient decision. |
| Display (flat ranked list, final response) | Top 5 by query relevance | Enough for clinical utility without overwhelming output. Re_ranker pools all systems and ranks by LLM relevance score. |

### Reliability

| Parameter | Value |
|---|---|
| Timeout per API call | 10 seconds |
| Max retries | 2 (3 total attempts) |
| Retry backoff | Exponential: 1 s, 2 s |
| On total failure for a system | Mark system as failed, continue with remaining systems. Failure is isolated — does not abort the pipeline. |

---

## Evaluator rules

The evaluator applies a **clinical domain** standard — filter results that belong to a fundamentally different clinical category than what the query requires. Within-domain variation (different specimen type, test method, or sub-classification for the same organism/drug/condition) is **not** filtered; the API's own ranking handles relevance within a system.

Refinement is triggered if **any** planner-selected system meets either condition:

1. **Empty results** — the API returned 0 results for that system.
2. **Domain mismatch** — the returned results are clearly from the wrong clinical category (e.g., a drug query against LOINC returns only lab measurement codes; a condition query returns procedure codes).

Within-domain variation is kept: a query for "ecoli" against LOINC accepts FISH assays, blood culture assays, and urine culture assays equally — they are all E. coli lab tests. Sub-type distinctions are for the user, not the evaluator.

**Coverage check:** if any meaningful component of the query is unrepresented by any selected system, that is always a refine decision — even if other systems returned strong results.

**Semantic filter (`relevant_codes`):** on every pass (sufficient *and* refine), the evaluator lists per system which specific codes belong to the correct clinical domain. The re_ranker applies this filter before pooling. An empty list for a system removes all its results. This ensures that if the iteration cap fires and the pipeline is forced forward on a "refine" decision, the best available filtered set is used rather than all raw results.

**Cap-hit summarizer behavior:** when the iteration cap fires and the decision is still "refine", the summarizer receives an explicit note naming the evaluator's final feedback. It is instructed to surface this in the summary — stating that the search was incomplete, naming the specific gap, and suggesting the user rephrase or narrow the query.

**Refinement loop:** loops back to `planner` carrying the original query plus the evaluator's diagnosis of what went wrong. The planner receives the prior attempt's results as context and may revise both system selection and search terms — not just search terms. Systems that returned strong results are not re-queried. If a system returned no results, the planner is guided to shorten its search term (the API is autocomplete-style; concise phrases find results where full descriptions do not).

---

## Output contract

The final response includes:

- A flat ranked list of up to 5 codes, ordered by query relevance. Each entry has: system label, code, display name.
- Plain-English summary explaining what was found and why each system was included.

The response does **not** surface which systems were excluded by the planner. The UI displays all 6 systems in a fixed sidebar so the user always knows what the system covers.

**Out-of-scope / miss queries:** the planner returns an empty `selected_systems` and states this in its rationale. The graph short-circuits directly to re_ranker (executor and evaluator are not called), producing an empty result set. The summarizer issues a single-sentence polite refusal explaining that the query does not map to any supported coding system, with a suggestion to rephrase using a clinical term.

*Output structure is subject to iteration after examining real API responses.*

---

## Streamlit UI

- Text input for the query
- Fixed sidebar listing all 6 coding systems and their descriptions (always visible)
- Streaming response (results appear as they arrive)
- Reasoning trace visible: which systems the planner selected, per-system API results, evaluator decision, refinement iterations (if any)

**Deployment target:** local only for Phase 0. Streamlit Cloud deployment is a post-Phase-0 decision.

---

## Observability

LangSmith tracing enabled. Every graph run produces a traceable execution with node-level inputs/outputs.

---

## Evaluation — gold set

| Parameter | Value |
|---|---|
| Gold set size | 30 queries |
| Version | `gold_v0.1.1.json` — never overwritten, version bumped on additions |
| Curation | Done by Nikhil (next step, after Phase 0 implementation) |
| Query type distribution | `simple`, `multi_system`, `ambiguous`, `miss` |

Target metrics (thresholds TBD after curation):

- **System-selection F1** — did the planner select the right systems?
- **Top-3 code recall** — did expected codes appear in the top 3?
- **Mean iterations / query** — refinement loop frequency
- **Mean API calls / query** — cost proxy

---

## Out of scope — Phase 0

- Multi-turn / conversational follow-up
- Multilingual queries
- Response caching (TTL cache is a post-Phase-0 improvement)
- Learned re-ranking (confidence relies solely on Clinical Tables' built-in scoring)
- LangSmith evaluation datasets or automated regression testing against LangSmith (tracing only)
- Streamlit Cloud deployment
- Authentication, rate limiting, or any production hardening
