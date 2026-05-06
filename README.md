# Clinical Codes Finder

An agentic system that takes a natural-language clinical term and returns relevant codes across six major medical coding systems — **ICD-10-CM**, **LOINC**, **RxNorm**, **HCPCS**, **UCUM**, and **HPO** — with a plain-English explanation of what was found and why.

![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

> 🎬 **Demo video:** [Watch on Loom](https://www.loom.com/share/14e85cc12fed4ae4b001382ee37fdb90)
> 🚀 **Live demo:** [clinical-codes-finder.streamlit.app](https://clinical-codes-finder-igkg6rozbv48bl4t9fktjm.streamlit.app/)

---

## The problem

Clinical data lives across half a dozen incompatible coding systems. A clinician searching for *"blood sugar test"* needs LOINC; a billing team handling *"wheelchair"* needs HCPCS; a pharmacy looking up *"metformin 500 mg"* needs RxNorm. Existing search tools force users to know which system to query before they search.

This project demonstrates an agent that **infers intent**, **routes to the relevant systems**, **executes searches in parallel**, **refines on weak results**, and **explains its reasoning** — all from a single natural-language query.

---

## Architecture

![architecture](docs/images/architecture.svg)

The pipeline is a LangGraph state machine. At its core is a tight **Planner → Executor → Evaluator** loop:

1. **`planner`** — LLM. In a single call, picks which of the 6 coding systems are relevant **and** generates per-system search terms. On refinement, it's re-entered with the prior attempt's results as context, so it can revise both decisions jointly. If no systems are selected (gibberish or non-clinical query), the graph short-circuits directly to re_ranker — executor and evaluator are never called.
2. **`executor`** — Async fan-out. Calls only the selected Clinical Tables APIs concurrently. Per-system failures are isolated.
3. **`evaluator`** — LLM. Inspects results and decides: *sufficient* → forward to re-ranking; *weak* → loop back to planner with feedback. Capped at 2 iterations.
4. **`re_ranker`** — LLM. Pools domain-filtered results from all systems, then ranks them by query relevance using an LLM. Returns a flat ordered list (max 5). Short-circuits without an LLM call for empty or small pools.
5. **`summarizer`** — LLM. Plain-English explanation with reasoning trace.

### Step-by-step

**Planner** (LLM)
- Decomposes the query into meaningful clinical components before selecting systems — `"ecoli 10000"` → organism name (`"ecoli"` → LOINC) + numeric quantity (`"10000"` → UCUM); `"diabetes"` → single component → ICD-10-CM only.
- Maps each component to a domain anchor and selects the union of systems needed to cover all of them.
- Generates one short, keyword-optimized search term per system (abbreviated phrases work; verbose descriptions don't).
- On refinement: receives prior search terms, weak systems, and the evaluator's diagnosis; can revise both system selection and search terms in the same call.

**Executor** (async fan-out)
- Calls all selected APIs concurrently via `asyncio.gather` — a timeout or error on one system does not block the others.
- Normalizes results to a common `{code, display, score, raw}` shape.
- `score` is a rank-position value: `(total - rank) / total`. The top API result always scores 1.0 — this is **not** a semantic similarity score.

**Evaluator** (LLM)
- Applies a *clinical domain* standard — filters results that belong to a fundamentally different clinical category than what the query requires. Within-domain variation (different specimen type, test method, sub-classification) is kept; the API's own ranking handles relevance within a system. Example: a query for "ecoli" against LOINC accepts FISH assays, blood cultures, and urine cultures equally — all are E. coli lab tests. Only a cross-domain result (a drug code appearing in lab results) is filtered.
- Runs a *result quality check*: does each system's results belong to the correct clinical domain for the query? A system is sufficient if it returned at least one on-domain result — stray off-domain codes are excluded via `relevant_codes`, not used to trigger refinement.
- Runs a *semantic filter*: populates `relevant_codes` — a per-system list of codes that match the clinical intent. The re_ranker applies this filter before pooling. Populated on every pass (sufficient and refine), so if the iteration cap forces the pipeline forward, the best available filtered set is used.
- On iteration 2+, receives a "carried over" block showing accumulated results for systems not re-queried this iteration — so it can populate `relevant_codes` for those systems and form an accurate picture of what's available.
- On refinement: provides a plain-English diagnosis of what's missing or mismatched, but does not prescribe what to do — the planner decides the remediation.
- Outputs a binary decision (sufficient / refine), not a numeric score — LLMs are poor calibrators; a score of 0.7 carries no stable meaning across runs. The prose diagnosis in `feedback` is more actionable than any number would be.
- Hard-capped at 2 iterations in graph state (not prompted into the LLM).

**Re-ranker** (LLM)
- Applies the evaluator's semantic filter first: keeps only codes listed in `relevant_codes` for each system. If a system's list is empty (all results irrelevant), that system contributes nothing to the pool.
- Flattens all domain-filtered results from every selected system into a single pool.
- If the pool is empty or small (≤ 5), returns it immediately without an LLM call.
- Otherwise, calls an LLM with the query and the full pool; the LLM returns the top-5 most relevant codes ranked most-to-least relevant.
- Output is a **flat ordered list** — system is metadata on each result, not an organizing structure. Codes are ordered by query relevance, not grouped by system.

**Summarizer** (LLM)
- Writes a plain-English explanation for a non-technical reader — patient, student, or general clinician.
- Covers what each selected system is, what was found, and why that system was relevant to the query.
- Defines medical terms inline.

### Why this architecture, and not ReAct?

The instinct on agent assignments is to reach for ReAct (think → act → observe in a loop with one LLM). I deliberately chose **Plan-and-Execute with parallel fan-out and a bounded refinement loop** instead:

- **The 6 coding systems are independent.** A search in LOINC has no bearing on a search in HCPCS. ReAct would needlessly serialize them, wasting latency and tokens on coordination the task doesn't need.
- **Most queries resolve in one pass.** The evaluator only triggers refinement when results are empty or don't match the clinical intent of the query — not on every query. This keeps the median path cheap.
- **Per-system fan-out gives clean traces.** Each tool call is a separate observable unit, easier to debug and evaluate than a single agent juggling 6 tools through one prompt.
- **The loop is bounded (max 2 iterations).** Unbounded refinement is where agents go to die. The cap is enforced in graph state, not prompted into the LLM.

Full trade-off analysis in [`docs/design-decisions.md`](docs/design-decisions.md).

---

## Setup

```bash
git clone <repo-url> && cd clinical-codes-finder
uv sync                    # or: pip install -e .
cp .env.example .env       # add ANTHROPIC_API_KEY
uv run pytest              # confirm 112 tests pass
```

## Usage

**CLI:**
```bash
uv run python -m scripts.run_query "metformin 500 mg"
uv run python -m scripts.run_query "metformin 500 mg" --output json | jq .
uv run python -m scripts.run_query "metformin 500 mg" --verbose
```

**Streamlit UI:**
```bash
uv run streamlit run src/clinical_codes/app/streamlit_app.py
```

**Run the eval:**
```bash
uv run python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json
```

---

## Evaluation

The system is evaluated against a hand-curated gold set (`data/gold/gold_v0.1.1.json`) of 31 queries spanning five difficulty types: **simple** (one system, unambiguous), **multi-system** (legitimately spans 2+ systems), **ambiguous** (planner judgment call), **refinement** (designed to fail on first pass), and **miss** (out-of-scope / gibberish — agent should return empty).

Results from eval run `20260506_123053` (gold v0.1.1, 0 errors):

| Metric | Value | What it measures |
|---|---|---|
| System-selection F1 | 0.92 | Did the planner pick the right systems? |
| Top-3 code recall | 0.51 | Did expected codes appear in the top 3? |
| Must-include hit rate | 0.75 | Did canonically required codes appear? |
| Mean iterations / query | 1.23 | How often does refinement actually fire? |
| Mean API calls / query | 1.65 | Cost proxy. Lower with better planning. |

Sliced by query type:

| Query type | N | System-selection F1 | Top-3 recall |
|---|---|---|---|
| simple | 11 | 1.00 | 0.82 |
| multi_system | 8 | 0.79 | 0.25 |
| ambiguous | 8 | 0.92 | 0.42 |
| refinement | 1 | 1.00 | 0.00 |
| miss | 3 | 1.00 | n/a |

**What improved:** System-selection F1 is now 0.92 and top-3 recall 0.51, up from 0.85 / 0.42 in the prior run. Simple queries hit system_f1 = 1.00 across the board. The evaluator semantic filter now correctly populates `relevant_codes` for carried-over systems on refinement iterations, preventing ICD-10-CM results from prior iterations from being dropped when the re-ranker pools results. The evaluator was also made less strict — it no longer triggers refinement when a system returns mostly on-domain results with a few stray codes (those are excluded via `relevant_codes`, not re-queried). Must-include hit rate recovered from 0.50 → 0.75 as a result.

**Remaining gaps:** Multi-system top-3 recall (0.25) is the main weakness. Queries like `"hypertension management"` and `"diabetes management"` span 3+ systems and the planner's conservative defaults sometimes under-select, leaving expected codes unrepresented in the pool. Full results in `results/`.

---

## Project structure

```
clinical-codes-finder/
├── pyproject.toml                     # deps + project metadata (uv)
├── .env.example                       # ANTHROPIC_API_KEY
│
├── src/clinical_codes/
│   ├── config.py                      # settings, env vars, model name, timeouts
│   ├── schemas.py                     # shared types: SystemName, CodeResult
│   │
│   ├── tools/                         # per-system Clinical Tables API wrappers
│   │   ├── base.py                    # http client, retry, timeout, normalize → {code,display,score,raw}
│   │   ├── icd10cm.py
│   │   ├── loinc.py
│   │   ├── rxnorm.py                  # includes dose-string fallback
│   │   ├── hcpcs.py
│   │   ├── ucum.py
│   │   └── hpo.py
│   │
│   ├── graph/                         # LangGraph state machine
│   │   ├── state.py                   # GraphState TypedDict, PlannerOutput, EvaluatorOutput
│   │   ├── prompts.py                 # all prompt templates in one place
│   │   ├── nodes.py                   # planner, executor, evaluator, re_ranker, summarizer
│   │   └── builder.py                 # build_graph() — wires nodes + conditional edges
│   │
│   ├── evaluation/
│   │   ├── schema.py                  # GoldQuery, GoldSet
│   │   ├── runner.py                  # runs gold set through the graph
│   │   ├── metrics.py                 # system-selection F1, recall@k, mean iters, mean API calls
│   │   └── reporter.py                # results table + markdown summary
│   │
│   ├── cli/
│   │   └── display.py                 # Rich display helpers (used by run_query)
│   │
│   └── app/
│       └── streamlit_app.py
│
├── data/gold/                         # versioned gold eval sets
│   ├── gold_v0.1.1.json               # current — API-verified (31 queries)
│   └── README.md                      # curation notes, query-type distribution
│
├── results/                           # eval run outputs
├── scripts/
│   ├── run_query.py                   # python -m scripts.run_query "diabetes"
│   └── run_eval.py                    # python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json
│
├── tests/                             # mirrors src/ layout
│   ├── tools/
│   ├── graph/
│   └── evaluation/
│
└── docs/
    ├── design-decisions.md            # architecture trade-offs and key decisions
    └── images/architecture.svg
```

---

## Limitations

- **English only.** Planner prompt and gold set are English-only; multilingual support would need re-evaluation.
- **Single-turn.** No conversational follow-ups (e.g. "now narrow to type 2"). State is reset per query.
- **Refinement capped at 2 iterations.** Long-tail ambiguous queries may not converge; this is by design — unbounded loops are worse than honest failure.
- **No caching.** Every query hits Clinical Tables fresh. A simple TTL cache would meaningfully cut API calls in production.
- **Re-ranker is only as good as the LLM's clinical vocabulary.** It ranks by query relevance using an LLM, which handles specificity well but may not perfectly distinguish sub-type variants (e.g., different specimen types for the same lab test).

---

## What I'd do with more time

**Completed improvements:**
- **RxNorm dose-string fallback** — Queries like `"metformin 500 mg"` previously returned zero results because the RxTerms API prefix-matches on drug display names. The fix detects dose patterns, retries with just the drug name, and ranks results so the matching strength surfaces first. Top-3 recall improved from 0.43 → 0.51.
- **Planner conservative defaults** — Replaced `"select 1–3 systems"` with explicit domain anchors (bare disease → ICD-10-CM, drug → RxNorm, lab test → LOINC, etc.) and a conservative default of 1 system. System-selection F1 improved from 0.69 → 0.85 (+23%); simple query precision went from ~0.33 to near-perfect.
- **Miss-query catch** — Added an instruction to return empty system selection for clearly non-clinical inputs (keyboard mash, non-medical questions). All 3 miss-type queries now score system_f1 = 1.0 (was 0.67 average).
- **Evaluator semantic filtering** — The evaluator populates `relevant_codes` on every pass (sufficient and refine), listing only codes that belong to the correct clinical domain for the query. The re_ranker applies this filter before pooling. The standard is cross-domain mismatch (drug query → lab codes = filter) not sub-type specificity (E. coli FISH assay vs urine culture = same domain, keep both).
- **Result re-ranking** — Replaced the deterministic consolidator with an LLM-based re_ranker that pools domain-filtered results from all systems and ranks them by query relevance. Output is a flat ordered list (max 5) instead of grouped-by-system tables. The LLM prefers specificity ("lisinopril 20 MG Oral Tablet" over "lisinopril (Oral Pill)" for the query "lisinopril 20 mg").
- **Refinement autocomplete guidance** — On refinement, if a system returned no results, the planner is now guided to shorten its search term (the Clinical Tables API is autocomplete-style; concise 1–3 word phrases find results where full descriptions do not).
- **Miss-query short-circuit** — When the planner returns an empty system selection (gibberish, keyboard mash, non-clinical input), the graph now routes directly to the re_ranker, skipping executor and evaluator entirely. Previously these queries burned 2 full iterations as the evaluator wrongly voted "refine" on nothing. Now they resolve in a single planner call.
- **Summarizer cap-hit callout** — When the refinement cap fires and the pipeline is forced forward on a "refine" decision, the summarizer now explicitly flags the incomplete search, names the evaluator's identified gap(s), and suggests rephrasing — rather than presenting partial results as a complete answer.

**Longer-term:**
- **Split the planner into a deterministic router + LLM planner** for cost efficiency. Cheap rules or a small classifier handles 80% of unambiguous queries (e.g. "mg/dL" → UCUM); LLM only fires on the ambiguous tail.
- **Replace the LLM evaluator with a deterministic policy** for clear-cut cases (zero results, single high-score match) and reserve the LLM call for genuinely ambiguous outcomes.
- **LangSmith tracing** for production observability.
- **Expand the gold set** to 100+ queries with inter-rater agreement on the ambiguous slice.
- **SME-guided prompt and gold set refinement** — the precision/conservatism balance in the planner prompt is ultimately a business decision: a billing team wants high precision (only route when confident), a research team wants high recall (cast wide). Collaborating with clinical SMEs to annotate edge cases and tune the domain anchors against a larger, domain-validated gold set would ground these trade-offs in real use-case requirements rather than heuristics.
- **Cache layer** with TTL keyed on `(system, normalized_query)`.

---

## Stack

LangGraph · Claude Anthropic · Pydantic · httpx · Rich · Typer · Streamlit · pytest · uv