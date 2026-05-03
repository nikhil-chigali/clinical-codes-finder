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

uv run python -m scripts.run_query "metformin 500 mg"  # CLI query runner

# Streamlit UI:
uv run streamlit run src/clinical_codes/app/streamlit_app.py

uv run python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json
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
| `graph/builder.py` — graph assembly | ✅ Done |
| `evaluation/schema.py` — GoldQuery, GoldSet, RunResult | ✅ Done |
| `evaluation/runner.py` — run_query, run_gold_set, lazy graph singleton | ✅ Done |
| `evaluation/metrics.py` — QueryMetrics, MetricsSummary, compute_metrics | ✅ Done |
| `evaluation/reporter.py` — results table + markdown summary | ✅ Done |
| `scripts/run_query.py` — CLI query runner (Rich + Typer) | ✅ Done |
| `app/streamlit_app.py` — Streamlit UI | ✅ Done |
| `scripts/run_eval.py` — evaluation runner CLI | ✅ Done |

## Project layout

```
src/clinical_codes/
├── config.py          # settings, env vars, model name, timeouts
├── schemas.py         # shared types: SystemName, normalized result shape
├── tools/             # per-system Clinical Tables API wrappers
├── graph/             # LangGraph nodes, state, prompts, builder
├── evaluation/        # gold set schema, runner, metrics, reporter
├── cli/               # Rich/Typer display helpers (used by scripts/run_query.py)
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

## To-do (pre-share checklist)

- [ ] **Planner prompt fixes (Fix B + Fix D)** — spec approved at `docs/superpowers/specs/2026-05-02-planner-prompt-fixes-design.md`. Next step: invoke `writing-plans` to generate the implementation plan, then implement. Target: system_f1 improves for `q001`–`q003`, `q009`; `q029` stops selecting systems.
- [ ] **Review `docs/design-decisions.md`** — 12 sections now. Decide before sharing externally which to keep vs. cut (§6 Refinement context and §7 Summarizer are candidates to condense or drop; §8 Metrics types is an implementation detail that may not interest external readers).
- [ ] **README final check** — confirm eval table reflects the post-Fix-B+D run, all setup instructions work on a clean clone, and demo links are filled in once deployed.
- [ ] **Deploy Streamlit app** — push repo is already at `nikhil-chigali/clinical-codes-finder`. Steps: connect to [share.streamlit.io](https://share.streamlit.io), set main file to `src/clinical_codes/app/streamlit_app.py`, add `ANTHROPIC_API_KEY` under App Settings → Secrets (Streamlit Cloud injects it as an env var — no code changes needed). Verify with a test query after deploy.
- [ ] **Share with interviewer** — GitHub repo URL + live Streamlit URL once deployed. Final `git status` + `git push` before sending.

---

## Notes

- `langsmith` appears in pytest plugin output — it's a transitive dep of `langchain-anthropic`, not explicitly installed
- Graph nodes (`planner`, `executor`, `evaluator`, `summarizer`) are `async def`. Calling the compiled graph from synchronous code requires `asyncio.run(graph.ainvoke(...))` — `.invoke()` raises `TypeError: No synchronous function provided` in LangGraph 1.1.10
- Integration tests (`@pytest.mark.integration`) are excluded from `uv run pytest` by default. Run them explicitly: `uv run pytest -m integration -v`
