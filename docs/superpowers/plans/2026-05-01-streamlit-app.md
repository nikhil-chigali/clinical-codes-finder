# `app/streamlit_app.py` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-page Streamlit app that accepts a clinical term, runs the LangGraph pipeline, and renders the summary, per-system results, and full reasoning trace.

**Architecture:** One file — `src/clinical_codes/app/streamlit_app.py`. All rendering is inline `st.*` calls. A `@st.cache_resource` singleton builds the graph once per server process. No unit tests (Streamlit components can't be isolated); verified by running the app manually.

**Tech Stack:** Streamlit ≥ 1.35, asyncio, LangGraph `ainvoke`, Pydantic models from `graph/state.py`

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `pyproject.toml` | Add `streamlit>=1.35` dependency |
| Create | `src/clinical_codes/app/streamlit_app.py` | Full Streamlit app |
| Modify | `CLAUDE.md` | Mark streamlit_app.py ✅ Done; uncomment run command |
| Modify | `README.md` | Mark streamlit_app.py ✅ Done; remove "pending" note |

`src/clinical_codes/app/__init__.py` already exists (empty) — do not create.

---

## Task 1: Add `streamlit` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `streamlit>=1.35` to `pyproject.toml`**

Edit the `dependencies` list (insert alphabetically, after `rich>=13.0`):

```toml
[project]
name = "clinical-codes-finder"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28",
    "langchain-anthropic>=0.3",
    "langchain-core>=0.3",
    "langgraph>=0.2",
    "pydantic>=2.13.3",
    "pydantic-settings>=2.14.0",
    "rich>=13.0",
    "streamlit>=1.35",
    "tenacity>=8.0",
    "typer>=0.12",
]
```

- [ ] **Step 2: Sync dependencies**

```bash
uv sync
```

Expected: resolves and installs `streamlit` and its dependencies (click, tornado, etc.).

- [ ] **Step 3: Verify existing tests still pass**

```bash
uv run pytest
```

Expected: all existing tests pass (102 passing, 1 deselected).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add streamlit dependency"
```

---

## Task 2: Create `streamlit_app.py`

**Files:**
- Create: `src/clinical_codes/app/streamlit_app.py`

No unit tests — Streamlit components can't be isolated. Verification is a syntax check + manual run.

**Key imports and types to know:**
- `SYSTEM_CATALOG: dict[SystemName, str]` — imported from `clinical_codes.graph.prompts`. Contains the description string for each of the 6 coding systems. Used to populate the sidebar.
- `build_graph()` — returns a compiled LangGraph graph. Decorated with `@st.cache_resource` so it's built once.
- `make_initial_state(query: str) -> GraphState` — creates the initial graph state dict.
- `state["consolidated"]: dict[SystemName, list[CodeResult]]` — per-system results after the pipeline.
- `state["summary"]: str` — plain-English summary from the summarizer node.
- `state["attempt_history"]: list[Attempt]` — each `Attempt` has:
  - `attempt.iteration: int`
  - `attempt.planner_output.selected_systems: list[SystemName]`
  - `attempt.planner_output.rationale: str`
  - `attempt.evaluator_output.decision: Literal["sufficient", "refine"]`
  - `attempt.evaluator_output.feedback: str` (empty string when decision is "sufficient")

- [ ] **Step 1: Create `src/clinical_codes/app/streamlit_app.py`**

```python
from __future__ import annotations

import asyncio

import streamlit as st

from clinical_codes.graph.builder import build_graph, make_initial_state
from clinical_codes.graph.prompts import SYSTEM_CATALOG

st.set_page_config(page_title="Clinical Codes Finder", layout="wide")


@st.cache_resource
def _get_graph():
    return build_graph()


# ── Sidebar — system catalog ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"**SUPPORTED SYSTEMS ({len(SYSTEM_CATALOG)})**")
    for system, description in SYSTEM_CATALOG.items():
        with st.expander(system.value):
            st.caption(description)


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Clinical Codes Finder")
st.caption(
    "Enter a clinical term to search across ICD-10-CM, LOINC, RxNorm, HCPCS, UCUM, and HPO."
)

query = st.text_input("Clinical term", placeholder="e.g. metformin 500 mg")
search = st.button("Search", disabled=not bool(query.strip()))

if search and query.strip():
    try:
        with st.spinner("Running..."):
            state = asyncio.run(_get_graph().ainvoke(make_initial_state(query.strip())))
    except Exception as e:
        st.error(str(e))
        st.stop()

    consolidated = state["consolidated"]
    summary = state["summary"]
    attempt_history = state["attempt_history"]

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Summary**")
    st.markdown(summary)

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Results**")
    if not consolidated:
        st.info("No results found.")
    else:
        for system, results in consolidated.items():
            with st.expander(f"{system.value} · {len(results)} results", expanded=True):
                st.dataframe(
                    [
                        {
                            "Code": r.code,
                            "Display": r.display,
                            "Score": f"{r.score:.2f}",
                        }
                        for r in results
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    # ── Reasoning trace ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔍 Reasoning trace"):
        if len(attempt_history) > 1:
            n = len(attempt_history)
            st.markdown(
                f'<span style="background:#5a3a2a;color:#e8a87c;padding:2px 10px;'
                f'border-radius:12px;font-size:13px">🔁 {n} iterations</span>',
                unsafe_allow_html=True,
            )
            st.write("")
        for i, attempt in enumerate(attempt_history):
            systems = ", ".join(s.value for s in attempt.planner_output.selected_systems)
            st.markdown(f"**Iteration {attempt.iteration}** · {systems}")
            st.caption(attempt.planner_output.rationale)
            if attempt.evaluator_output.decision == "sufficient":
                st.success("✓ Sufficient")
            else:
                st.warning(f"↩ Refine — {attempt.evaluator_output.feedback}")
            if i < len(attempt_history) - 1:
                st.divider()
```

- [ ] **Step 2: Verify syntax**

```bash
uv run python -m py_compile src/clinical_codes/app/streamlit_app.py && echo "OK"
```

Expected: prints `OK` with no errors.

- [ ] **Step 3: Run the full test suite**

```bash
uv run pytest
```

Expected: all existing tests still pass (102 passing, 1 deselected). The app file is not imported by any test.

- [ ] **Step 4: Manual smoke test (requires `.env` with `ANTHROPIC_API_KEY`)**

```bash
uv run streamlit run src/clinical_codes/app/streamlit_app.py
```

Open http://localhost:8501. Verify:
- Sidebar shows "SUPPORTED SYSTEMS (6)" with 6 expandable entries (ICD10CM, LOINC, RXNORM, HCPCS, UCUM, HPO)
- Each sidebar entry shows its description text when expanded
- Title and subtitle visible in main area
- "Search" button is greyed out when input is empty
- Enter "metformin 500 mg", click Search: spinner appears, then summary + results + reasoning trace render
- Reasoning trace expander is collapsed by default; expand it to see iteration detail

- [ ] **Step 5: Commit**

```bash
git add src/clinical_codes/app/streamlit_app.py
git commit -m "feat: add streamlit_app.py — query UI with results and reasoning trace"
```

---

## Task 3: Update `CLAUDE.md` and `README.md`

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md`**

In the `## Commands` section, uncomment the streamlit run command (it currently reads `# uv run streamlit run ...`):

```
uv run streamlit run src/clinical_codes/app/streamlit_app.py
```

In the implementation status table, change the `app/streamlit_app.py` row from:

```
| `app/streamlit_app.py` — Streamlit UI | 🔲 Pending |
```

to:

```
| `app/streamlit_app.py` — Streamlit UI | ✅ Done |
```

- [ ] **Step 2: Update `README.md`**

In the implementation status table, change the `app/streamlit_app.py` row from:

```
| `app/streamlit_app.py` — Streamlit UI | 🔲 Pending |
```

to:

```
| `app/streamlit_app.py` — Streamlit UI | ✅ Done |
```

In the `## Usage` section, the Streamlit UI block currently reads:

```
**Streamlit UI** *(pending `app/streamlit_app.py`)*:
```bash
uv run streamlit run src/clinical_codes/app/streamlit_app.py
```

Change to:

```
**Streamlit UI:**
```bash
uv run streamlit run src/clinical_codes/app/streamlit_app.py
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: mark streamlit_app.py done; update usage instructions"
```
