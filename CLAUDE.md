# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An agentic system that takes a natural-language clinical term and returns relevant codes across six medical coding systems — **ICD-10-CM**, **LOINC**, **RxNorm**, **HCPCS**, **UCUM**, and **HPO** — with a plain-English explanation. External data source is the NLM Clinical Tables API (no auth required).

## Commands

```bash
uv sync                    # install deps (use uv, not pip)
cp .env.example .env       # add ANTHROPIC_API_KEY

uv run pytest              # all tests
uv run pytest tests/graph/test_nodes.py  # single test file

# Not yet implemented (pending builder, evaluation, app, scripts phases):
# uv run python -m scripts.run_query "metformin 500 mg"
# uv run streamlit run src/clinical_codes/app/streamlit_app.py
# uv run python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json
```

## Architecture

LangGraph state machine with 5 nodes in `src/clinical_codes/graph/`:

1. **`planner`** (LLM) — in one call, selects 1–3 relevant coding systems **and** generates one search term per selected system. On refinement, re-entered with the prior attempt's results as context, so it can revise both system selection and search terms jointly.
2. **`executor`** (async fan-out) — calls only the selected Clinical Tables APIs concurrently; per-system failures are isolated.
3. **`evaluator`** (LLM) — decides *sufficient* (forward) or *refine* (loop back to planner). **Capped at 2 iterations** — enforced in graph state, not in the prompt.
4. **`consolidator`** (deterministic) — dedup, group by system, rank by API confidence score.
5. **`summarizer`** (LLM) — plain-English explanation with reasoning trace.

Graph is assembled in `graph/builder.py`. State shape lives in `graph/state.py`. All prompt templates are centralized in `graph/prompts.py`.

## Key design decisions

- **Merged planner (not separate router + planner).** A separate router could only revise search terms on refinement — never reconsider system selection. The merged planner can correct both decisions jointly when looping back.
- **Plan-and-Execute with parallel fan-out, not ReAct.** The 6 systems are independent — ReAct would serialize them unnecessarily.
- **Refinement only fires on weak results** (empty results for a planner-selected system, or all results below confidence floor). Not on every query.
- **Gold set is versioned** (`data/gold/gold_v0.1.1.json`). Never overwrite — bump the version when adding queries.
- Tools in `src/clinical_codes/tools/` wrap the Clinical Tables API and normalize to a common shape `{code, display, score, raw}`. Base client with retry/timeout lives in `tools/base.py`.

## Implementation status

| Component | Status |
|---|---|
| `tools/` — 6 Clinical Tables API wrappers + base client | ✅ Done |
| `graph/state.py` — TypedDict, Pydantic models, `operator.add` reducer | ✅ Done |
| `graph/prompts.py` — all prompt templates | ✅ Done |
| `graph/nodes.py` — all 5 nodes | ✅ Done |
| `graph/builder.py` — graph assembly | 🔲 Pending |
| `evaluation/` — gold set runner, metrics, reporter | 🔲 Pending |
| `app/streamlit_app.py` — Streamlit UI | 🔲 Pending |
| `scripts/run_query.py`, `scripts/run_eval.py` | 🔲 Pending |

## Open obligations for `graph/builder.py`

These constraints are enforced by convention, not the type system — the builder author must honour them:

1. **Iteration contract.** The `planner` node must write `{"iteration": state["iteration"] + 1}` as part of its return value at the start of each pass. `route_after_evaluator` reads `iteration` post-increment; with `MAX_ITERATIONS=2`, the second evaluator call sees `iteration=2`, which triggers the cap. If the planner does not increment, the cap never fires.

2. **Node-name constants.** `NODE_PLANNER = "planner"` and `NODE_CONSOLIDATOR = "consolidator"` are defined as module-level constants in `config.py` (alongside `MAX_ITERATIONS`). `route_after_evaluator` lives in `builder.py` — not `state.py` — and returns these constants. Register nodes with `add_node(NODE_PLANNER, planner)` etc. A name mismatch silently routes to a nonexistent node and surfaces only at runtime.

## Project layout

```
src/clinical_codes/
├── config.py          # settings, env vars, model name, timeouts
├── schemas.py         # shared types: SystemName, normalized result shape
├── tools/             # per-system Clinical Tables API wrappers
├── graph/             # LangGraph nodes, state, prompts, builder
├── evaluation/        # gold set schema, runner, metrics, reporter
└── app/               # Streamlit UI

data/gold/             # versioned gold eval sets (do not overwrite)
scripts/               # CLI entry points: run_query.py, run_eval.py
tests/                 # mirrors src/ layout
```

## Evaluation metrics

- Router F1 (did the router select the right systems?)
- Top-3 code recall (did expected codes appear in the top 3?)
- Mean iterations/query (refinement loop frequency)
- Mean API calls/query (cost proxy)

Sliced by query type: `simple`, `multi_system`, `ambiguous`, `miss`.

## Notes

- `langsmith` appears in pytest plugin output — it's a transitive dep of `langchain-anthropic`, not explicitly installed
