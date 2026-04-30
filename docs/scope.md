# Scope — Phase 0

Defines what is built, what is not, and every decision that would otherwise require re-discussion mid-build.

---

## Objective

Accept a single natural-language clinical query, infer which of six medical coding systems are relevant, search them in parallel, refine once if results are weak, and return matching codes with a plain-English explanation.

---

## Architecture constraints (locked)

| Constraint | Value |
|---|---|
| Graph nodes | 5 (planner → executor → evaluator → consolidator → summarizer) |
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
| `summarizer` | claude-sonnet-4-6 | Synthesizes codes across systems into coherent prose for a potentially non-technical audience. Quality of language matters here. |

### Temperature

| Node(s) | Temperature | Justification |
|---|---|---|
| `planner`, `evaluator` | `0` | Deterministic output required for reproducible evaluation runs. Variation in system selection or quality judgments would make eval metrics meaningless across runs. |
| `summarizer` | `0.3` | Slightly warmer for more natural prose. Still constrained enough to stay factual; loose enough to avoid robotic repetition when summarizing similar result sets. |

---

## API layer

### Results per system

| Parameter | Value | Justification |
|---|---|---|
| Fetch (per system, per call) | 10 results | Gives the evaluator enough signal to assess result quality. Beyond 10, marginal results are unlikely to change the refine/sufficient decision. |
| Display (per system, in final response) | Top 5 by confidence score | Enough for clinical utility without overwhelming output. Filtered after consolidation. |

### Reliability

| Parameter | Value |
|---|---|
| Timeout per API call | 10 seconds |
| Max retries | 2 (3 total attempts) |
| Retry backoff | Exponential: 1 s, 2 s |
| On total failure for a system | Mark system as failed, continue with remaining systems. Failure is isolated — does not abort the pipeline. |

---

## Evaluator rules

Refinement is triggered if **any** planner-selected system meets either condition:

1. **Empty results** — the API returned 0 results for that system.
2. **Low-confidence results** — all returned results for that system fall below a confidence threshold.

**Confidence threshold:** `0.5` on a normalized 0–1 scale (initial value; to be refined after examining real API responses across all six systems).

**Refinement loop:** loops back to `planner` carrying the original query plus the evaluator's diagnosis of what went wrong. The planner receives the prior attempt's results as context and may revise both system selection and search terms — not just search terms. Systems that returned strong results are not re-queried.

---

## Output contract

The final response includes, per system selected by the planner:

- System label (e.g., `LOINC`, `RxNorm`)
- Up to 5 results, each with: code, display name, confidence score
- Plain-English summary explaining what was found and why each system was included

The response does **not** surface which systems were excluded by the planner. The UI displays all 6 systems in a fixed sidebar so the user always knows what the system covers.

**Out-of-scope / miss queries:** polite refusal — a brief message explaining that the query does not map to any of the supported coding systems, with a suggestion to rephrase using a clinical term.

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
