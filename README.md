# Clinical Codes Finder

An agentic system that takes a natural-language clinical term and returns relevant codes across six major medical coding systems — **ICD-10-CM**, **LOINC**, **RxNorm**, **HCPCS**, **UCUM**, and **HPO** — with a plain-English explanation of what was found and why.

![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

> 🎬 **Demo video:** coming soon
> 🚀 **Live demo:** [clinical-codes-finder.streamlit.app](https://clinical-codes-finder-igkg6rozbv48bl4t9fktjm.streamlit.app/)

---

## The problem

Clinical data lives across half a dozen incompatible coding systems. A clinician searching for *"blood sugar test"* needs LOINC; a billing team handling *"wheelchair"* needs HCPCS; a pharmacy looking up *"metformin 500 mg"* needs RxNorm. Existing search tools force users to know which system to query before they search.

This project demonstrates an agent that **infers intent**, **routes to the relevant systems**, **executes searches in parallel**, **refines on weak results**, and **explains its reasoning** — all from a single natural-language query.

---

## Architecture

![architecture](docs/images/architecture.svg)

The pipeline is a LangGraph state machine. At its core is a tight **Planner → Executor → Evaluator** loop:

1. **`planner`** — LLM. In a single call, picks which of the 6 coding systems are relevant **and** generates per-system search terms. On refinement, it's re-entered with the prior attempt's results as context, so it can revise both decisions jointly.
2. **`executor`** — Async fan-out. Calls only the selected Clinical Tables APIs concurrently. Per-system failures are isolated.
3. **`evaluator`** — LLM. Inspects results and decides: *sufficient* → forward to consolidation; *weak* → loop back to planner with feedback. Capped at 2 iterations.
4. **`consolidator`** — Deterministic. Dedups, groups by system, ranks by API result order.
5. **`summarizer`** — LLM. Plain-English explanation with reasoning trace.

### Step-by-step

**Planner** receives the raw query and first decomposes it into meaningful clinical components before selecting systems. `"ecoli 10000"` decomposes into an organism name (`"ecoli"` → LOINC) and a numeric quantity (`"10000"` → UCUM); `"diabetes"` is a single component (bare disease name → ICD-10-CM only). Each component is mapped to a domain anchor and the planner selects the union of systems needed to cover all of them. It then generates one search term per system — short, keyword-optimized phrases rather than full descriptions, since the Clinical Tables APIs work best with abbreviated terms. On refinement, the planner receives the prior search terms, the list of weak systems, and the evaluator's diagnosis; it can revise both system selection and search terms in the same call.

**Executor** fans out concurrently to all selected APIs using `asyncio.gather`. Each system is called independently — a timeout or error on one does not block the others. Results are normalized to a common `{code, display, score, raw}` shape. The `score` field is a rank-position value: `(total - rank) / total`, mapping the API's result ordering to [0, 1]. The top API result always scores 1.0 regardless of how well it matches the query — this is **not** a semantic similarity score.

**Evaluator** runs two checks before deciding whether to continue or refine. First, a *result quality check*: are the returned display names semantically relevant to the query? Second, a *coverage check*: is every meaningful component of the original query addressed by at least one selected system? A numeric quantity with no unit system selected, or a drug name with no RxNorm results, both trigger refinement even if the other systems returned strong results. When refinement fires, the evaluator provides a plain-English diagnosis — what's missing or mismatched — but does not prescribe what to do next. The planner, which has full domain context in its system prompt, decides the remediation. The loop is hard-capped at 2 iterations in graph state (not prompted).

**Consolidator** is the only fully deterministic node. It deduplicates within each system by code, sorts by API rank score descending, and keeps the top 5 per system. Results stay **grouped by system** — there is intentionally no cross-system ranking or score normalization. The rank-position score is meaningful within a single system's response (it reflects that API's own relevance ordering) but not across systems: a score of 1.0 from LOINC and 1.0 from UCUM both mean "top result in that API's response" — they carry no comparable semantic weight. Normalizing them into a flat ranked list would be false precision. The evaluator is the semantic quality gate; the consolidator's job is to organize what made it through, preserving the per-system grouping that makes results interpretable to different clinical audiences.

**Summarizer** takes the consolidated results and the planner's rationale and writes a plain-English explanation for a non-technical reader — patient, student, or general clinician. It explains what each selected system covers, what was found, and why that system was relevant to this query. Medical terms are defined inline.

### Per-system nuances

- **ICD-10-CM** — Diseases, injuries, and clinical conditions. Bare disease or condition names route here by default. Does not handle phenotypic traits (HPO) or drug names (RxNorm).
- **LOINC** — Lab tests and clinical observations. The Clinical Tables API matches on abbreviated panel labels ("Bacteria Ur Cult", "Glucose Ser Qn") rather than full descriptions — the planner generates short 1–3 word terms for LOINC specifically. Verbose phrases like "Escherichia coli colony count" return nothing.
- **RxNorm** — Drug names and dosage forms. The API prefix-matches on display names that include route ("metFORMIN (Oral Pill)"). Dose-qualified queries ("metformin 500 mg") use a two-step fallback: if the full term returns nothing, the client strips the dose, retries with just the drug name, and expands results per-strength so the matching dose surfaces first.
- **HCPCS** — Procedures, devices, and durable medical equipment billed to Medicare/Medicaid. Device and equipment queries route here ("wheelchair", "CPAP machine").
- **UCUM** — Units of measure. Handles standalone unit queries ("mg/dL", "CFU/mL") and the unit component of multi-part queries ("ecoli 10000" → LOINC + UCUM). A unit embedded in a drug dosage string is not treated as a UCUM query — "mg" in "metformin 500 mg" is part of the RxNorm drug concept, not a measurement unit query.
- **HPO** — Human Phenotype Ontology. Phenotypic traits, observable clinical features, and rare-disease characteristics ("ataxia", "brachydactyly", "photophobia"). Distinct from ICD-10-CM: HPO describes what a patient looks like clinically; ICD-10-CM is for diagnostic billing codes.

### Why this architecture, and not ReAct?

The instinct on agent assignments is to reach for ReAct (think → act → observe in a loop with one LLM). I deliberately chose **Plan-and-Execute with parallel fan-out and a bounded refinement loop** instead:

- **The 6 coding systems are independent.** A search in LOINC has no bearing on a search in HCPCS. ReAct would needlessly serialize them, wasting latency and tokens on coordination the task doesn't need.
- **Most queries resolve in one pass.** The evaluator only triggers refinement when results are empty or semantically irrelevant to the query — not on every query. This keeps the median path cheap.
- **Per-system fan-out gives clean traces.** Each tool call is a separate observable unit, easier to debug and evaluate than a single agent juggling 6 tools through one prompt.
- **The loop is bounded (max 2 iterations).** Unbounded refinement is where agents go to die. The cap is enforced in graph state, not prompted into the LLM.

Full trade-off analysis in [`docs/design-decisions.md`](docs/design-decisions.md).

---

## Key design decisions

**1. Merged planner — system selection and search terms in one LLM call**

An earlier design split these into a separate router and planner. The problem: with separate nodes, the refinement loop could only revise search terms — it could never reconsider which systems were selected. If the router picked LOINC for "blood sugar test" and got weak results, the planner could only retry with different LOINC terms; it could never escalate to ICD-10-CM. Merging the two decisions into one call means refinement can correct both — the most impactful change for ambiguous queries.

**2. Evaluator diagnoses, planner decides**

The evaluator tells the planner *what went wrong* ("LOINC returned imaging codes for a drug query"), but never prescribes *what to do next*. That separation keeps the evaluator lightweight — it only needs to recognize a mismatch, not reason about clinical vocabulary. The planner, which already has full domain context in its system prompt, handles remediation. This makes both nodes easier to test and reason about independently.

**3. Domain anchors with a tunable precision/recall dial**

The planner prompt encodes explicit per-domain routing rules (bare disease → ICD-10-CM, drug → RxNorm, phenotypic trait → HPO, etc.) with a default of 1 system. This default is a deliberate business-level choice: a billing team needs high precision (only route when confident); a research team needs high recall (cast wide). The anchors and the multi-domain trigger threshold are the single place to tune that trade-off — no graph or code changes required.

---

## Setup

```bash
git clone <repo-url> && cd clinical-codes-finder
uv sync                    # or: pip install -e .
cp .env.example .env       # add ANTHROPIC_API_KEY
uv run pytest              # confirm 105 tests pass
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

Results from eval run `20260503_104339` (gold v0.1.1, 0 errors):

| Metric | Value | What it measures |
|---|---|---|
| System-selection F1 | 0.85 | Did the planner pick the right systems? |
| Top-3 code recall | 0.42 | Did expected codes appear in the top 3? |
| Must-include hit rate | 0.50 | Did canonically required codes appear? |
| Mean iterations / query | 1.23 | How often does refinement actually fire? |
| Mean API calls / query | 1.23 | Cost proxy. Lower with better planning. |

Sliced by query type:

| Query type | N | System-selection F1 | Top-3 recall |
|---|---|---|---|
| simple | 11 | 0.91 | 0.64 |
| multi_system | 8 | 0.72 | 0.21 |
| ambiguous | 8 | 0.83 | 0.40 |
| refinement | 1 | 1.00 | 0.00 |
| miss | 3 | 1.00 | n/a |

**What improved:** System-selection F1 jumped from 0.69 → 0.85 (+23%) after two planner prompt improvements. First, the planner was given a conservative default of 1 system and explicit domain anchors (bare disease → ICD-10-CM, drug → RxNorm, lab test → LOINC, etc.), which fixed over-selection on simple queries: `"diabetes"`, `"hypertension"`, `"asthma"`, and `"CPAP machine"` all went from system_f1 0.50 → 1.0. Second, an instruction was added to return an empty selection for clearly non-clinical inputs, which fixed gibberish queries like `"asdfghjkl"` (system_f1 0.00 → 1.0). Mean API calls dropped from 3.10 → 1.23. (A prior iteration also added an RxNorm dose-string fallback — enabling queries like `"lisinopril 20 mg"` to match on drug+strength rather than returning zero results — which improved top-3 recall from 0.43 → 0.51. The current run's top-3 recall of 0.42 reflects the cost of conservative system selection, not a loss of the fallback.)

**Remaining gaps:** Top-3 recall (0.42) and must-include hit rate (0.50) regressed from the prior run (0.51 and 0.75 respectively) due to the conservative system selection — the planner now defaults to fewer systems, which helps precision but hurts recall on multi-system queries. The planner correctly selects systems for simple queries, but the specific expected codes rarely surface in the top 3, most pronounced in multi-system queries (top-3 recall 0.21). Multi-system cases where gold expects 3+ systems (q017 `"hypertension management"`, q018 `"diabetes management"`) remain under-recalled. Full results in `results/`.

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
│   │   ├── nodes.py                   # planner, executor, evaluator, consolidator, summarizer
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
- **Ranking relies on Clinical Tables' built-in result order.** No learned re-ranker; result ordering is only as good as the API's own scoring.

---

## What I'd do with more time

**Completed improvements:**
- **RxNorm dose-string fallback** — Queries like `"metformin 500 mg"` previously returned zero results because the RxTerms API prefix-matches on drug display names. The fix detects dose patterns, retries with just the drug name, and ranks results so the matching strength surfaces first. Top-3 recall improved from 0.43 → 0.51.
- **Planner conservative defaults** — Replaced `"select 1–3 systems"` with explicit domain anchors (bare disease → ICD-10-CM, drug → RxNorm, lab test → LOINC, etc.) and a conservative default of 1 system. System-selection F1 improved from 0.69 → 0.85 (+23%); simple query precision went from ~0.33 to near-perfect.
- **Miss-query catch** — Added an instruction to return empty system selection for clearly non-clinical inputs (keyboard mash, non-medical questions). All 3 miss-type queries now score system_f1 = 1.0 (was 0.67 average).

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