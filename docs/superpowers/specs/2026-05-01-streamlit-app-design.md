# `app/streamlit_app.py` — Streamlit UI Design

## Goal

A single-page Streamlit app that accepts a natural-language clinical term, runs the LangGraph pipeline with a spinner, and renders the summary, per-system results, and a full reasoning trace with refinement history.

## Architecture

**One file:** `src/clinical_codes/app/streamlit_app.py`. All `st.*` calls are inline — no helper modules. Streamlit components can't be meaningfully unit-tested in isolation, so extraction adds indirection without benefit at this scale.

**Page layout:** Streamlit sidebar + main area.

---

## Public API

Run with:
```bash
uv run streamlit run src/clinical_codes/app/streamlit_app.py
```

No new Python modules or public functions — the app is a script, not a library.

---

## Sidebar

`st.sidebar` contains the system catalog:

- Header: **"SUPPORTED SYSTEMS (6)"**
- One `st.sidebar.expander(system.value)` per `SystemName` (all 6, always shown)
- Description text pulled from `SYSTEM_CATALOG` in `graph/prompts.py` — single source of truth, no duplication

`SYSTEM_CATALOG` is a `dict[SystemName, str]` already defined in `prompts.py`:
```python
from clinical_codes.graph.prompts import SYSTEM_CATALOG
```

---

## Main Area (top to bottom)

### 1. Title + subtitle
```
Clinical Codes Finder
Enter a clinical term to search across ICD-10-CM, LOINC, RxNorm, HCPCS, UCUM, and HPO.
```

### 2. Query input
`st.text_input` + `st.button("Search")` — plain form, no `st.form` wrapper.

Button is disabled (greyed) when the text input is empty.

### 3. Graph invocation
```python
@st.cache_resource
def get_graph():
    return build_graph()

with st.spinner("Running..."):
    state = asyncio.run(get_graph().ainvoke(make_initial_state(query)))
```

`@st.cache_resource` replaces the module-level singleton pattern — Streamlit's cache handles server restarts and thread safety.

### 4. Summary block
```python
st.markdown("**Summary**")
st.markdown(state["summary"])
```
Rendered with a visual separator (`st.divider()`) above it.

### 5. Results
Label: `st.markdown("**Results**")`

For each `(system, results)` in `state["consolidated"]`:
- `st.expander(f"{system.value} · {len(results)} results", expanded=True)`
- Inside: `st.dataframe` with columns **Code**, **Display**, **Score** (Score formatted `{score:.2f}`)

Systems with zero results are not shown (only systems in `consolidated` are iterated — the consolidator already filters empties).

If `consolidated` is empty: `st.info("No results found.")`.

### 6. Reasoning trace
`st.expander("🔍 Reasoning trace")` — collapsed by default.

**Refinement badge** (shown at top of expander if `len(attempt_history) > 1`):
```python
st.markdown(
    f'<span style="background:#5a3a2a;color:#e8a87c;padding:2px 10px;'
    f'border-radius:12px;font-size:13px">🔁 {n} iterations</span>',
    unsafe_allow_html=True,
)
```

**Per-iteration block** (for each `Attempt` in `state["attempt_history"]`):
```
Iteration N · RXNORM, ICD10CM
<rationale text in st.caption>
✓ Sufficient          ← st.success  (if evaluator_output.decision == "sufficient")
↩ Refine — <feedback> ← st.warning  (if evaluator_output.decision == "refine")
```
`st.divider()` between iterations.

---

## Data flow

```python
state = asyncio.run(get_graph().ainvoke(make_initial_state(query)))

consolidated   = state["consolidated"]     # dict[SystemName, list[CodeResult]]
summary        = state["summary"]          # str
attempt_history = state["attempt_history"] # list[Attempt]
```

`Attempt` fields used:
- `attempt.iteration` — int
- `attempt.planner_output.selected_systems` — list[SystemName]
- `attempt.planner_output.rationale` — str
- `attempt.evaluator_output.decision` — "sufficient" | "refine"
- `attempt.evaluator_output.feedback` — str (empty string when sufficient)

---

## Error handling

```python
try:
    state = asyncio.run(get_graph().ainvoke(make_initial_state(query)))
except Exception as e:
    st.error(str(e))
    st.stop()
```

No traceback shown — this is a UI. The error message from Anthropic's SDK or the graph is descriptive enough.

---

## Testing

No unit tests for `streamlit_app.py`. Streamlit components aren't isolatable. Verified by running the app manually:
```bash
uv run streamlit run src/clinical_codes/app/streamlit_app.py
```

The underlying graph, consolidator, and metrics are covered by the existing test suite (102 tests).

---

## Files

| Action | Path | Notes |
|---|---|---|
| Create | `src/clinical_codes/app/streamlit_app.py` | The app |
| Modify | `pyproject.toml` | Add `streamlit>=1.35` to dependencies |

`src/clinical_codes/app/__init__.py` already exists (empty).

---

## Dependencies to add

```toml
streamlit>=1.35
```

Pure-Python, no system deps. `streamlit>=1.35` is required for `st.cache_resource` (introduced in 1.18) and `st.dataframe` improvements used here. 1.35 is a safe minimum for the feature set used.
