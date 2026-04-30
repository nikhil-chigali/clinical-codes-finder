# Evaluation Runner Design

## Goal

Load the versioned gold set JSON, run each query through the compiled LangGraph graph sequentially, and return a list of `RunResult` objects capturing the metrics needed for system-selection F1, top-3 recall, mean iterations, and mean API calls — plus the plain-English summary for manual inspection.

---

## File map

| File | Action | Purpose |
|---|---|---|
| `src/clinical_codes/evaluation/__init__.py` | Create | Empty package marker |
| `src/clinical_codes/evaluation/schema.py` | Create | `GoldQuery`, `GoldSet`, `RunResult` Pydantic models |
| `src/clinical_codes/evaluation/runner.py` | Create | `run_query()`, `run_gold_set()`, lazy graph singleton |
| `tests/evaluation/__init__.py` | Create | Empty package marker |
| `tests/evaluation/test_runner.py` | Create | 3 unit tests + 1 integration smoke test |
| `pyproject.toml` | Modify | Register `integration` marker; add `addopts = "-m 'not integration'"` |

`schema.py` is kept separate from `runner.py` so `metrics.py` and `reporter.py` can import `GoldQuery`/`RunResult` without pulling in the graph or I/O dependencies.

---

## Data models (`schema.py`)

```python
class GoldQuery(BaseModel):
    id: str
    query: str
    query_type: str          # "simple" | "multi_system" | "ambiguous" | "refinement" | "miss"
    expected_systems: list[SystemName]
    expected_codes: dict[SystemName, list[str]]
    must_include: list[str]
    must_not_include: list[str]
    notes: str = ""

class GoldSet(BaseModel):
    version: str
    queries: list[GoldQuery]

class RunResult(BaseModel):
    query_id: str
    query: str
    query_type: str
    predicted_systems: list[SystemName]
    predicted_codes: dict[SystemName, list[str]]  # code strings only, top-5 per system
    iterations: int           # state["iteration"] — 1-based after first pass
    api_calls: int            # sum of selected_systems count across all attempts
    latency_s: float
    error: str | None         # str(exc) if graph raised, else None
    summary: str              # plain-English output from summarizer node
```

`api_calls` is computed as:
```python
sum(len(a.planner_output.selected_systems) for a in state["attempt_history"])
```
This counts distinct system-search pairs per iteration — the cost-proxy metric from scope.md.

For a `miss` query where the planner selects no systems: `predicted_systems=[]`, `predicted_codes={}`, `api_calls=0`. The graph still runs one pass so `iterations=1`.

---

## Runner logic (`runner.py`)

### Lazy graph singleton

Compiled once on first call, reused across all queries in the same process:

```python
_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
```

### `run_query(gold_query: GoldQuery) -> RunResult`

The unit function. Runs one query through the graph and extracts all `RunResult` fields from the final `GraphState`.

```python
def run_query(gold_query: GoldQuery) -> RunResult:
    start = time.monotonic()
    try:
        state = _get_graph().invoke(make_initial_state(gold_query.query))
        latency_s = time.monotonic() - start
        planner_out = state["planner_output"]
        predicted_systems = planner_out.selected_systems if planner_out else []
        predicted_codes = {
            sys: [r.code for r in results]
            for sys, results in state["consolidated"].items()
        }
        api_calls = sum(
            len(a.planner_output.selected_systems)
            for a in state["attempt_history"]
        )
        return RunResult(
            query_id=gold_query.id,
            query=gold_query.query,
            query_type=gold_query.query_type,
            predicted_systems=predicted_systems,
            predicted_codes=predicted_codes,
            iterations=state["iteration"],
            api_calls=api_calls,
            latency_s=latency_s,
            error=None,
            summary=state["summary"],
        )
    except Exception as exc:
        return RunResult(
            query_id=gold_query.id,
            query=gold_query.query,
            query_type=gold_query.query_type,
            predicted_systems=[],
            predicted_codes={},
            iterations=0,
            api_calls=0,
            latency_s=time.monotonic() - start,
            error=str(exc),
            summary="",
        )
```

### `run_gold_set(path: Path | str) -> list[RunResult]`

Thin loader + sequential loop over `run_query`. Prints one progress line per query to stdout. No file writing — that is reporter.py's responsibility.

```python
def run_gold_set(path: Path | str) -> list[RunResult]:
    data = json.loads(Path(path).read_text())
    gold_set = GoldSet.model_validate(data)
    results = []
    for gq in gold_set.queries:
        result = run_query(gq)
        status = f"ERROR: {result.error}" if result.error else f"{result.latency_s:.1f}s"
        print(f"  {gq.id} ({gq.query_type}): {status}")
        results.append(result)
    return results
```

---

## Testing (`tests/evaluation/test_runner.py`)

### Unit tests (3) — mock `_get_graph`, no real LLM or API calls

**Test 1 — happy path**

Mock graph returns a valid final `GraphState` with one system selected (ICD10CM), two `CodeResult` entries, one refinement iteration in `attempt_history`. Assert:
- `predicted_systems == [SystemName.ICD10CM]`
- `predicted_codes == {SystemName.ICD10CM: ["E11.9", "E10.9"]}`
- `iterations == 1`
- `api_calls == 1`
- `latency_s >= 0`
- `error is None`
- `summary == "some summary"`

**Test 2 — error handling**

Mock graph raises `RuntimeError("api error")`. Assert:
- `error == "api error"`
- `predicted_systems == []`
- `predicted_codes == {}`
- `iterations == 0`
- `summary == ""`
- `latency_s >= 0`

**Test 3 — `run_gold_set` loads and loops**

Write a minimal two-query gold JSON to a `tmp_path` file. Patch `run_query` to return a fixed `RunResult`. Call `run_gold_set(path)`. Assert the returned list has two entries with the correct `query_id` values (`"q001"`, `"q002"`). Isolates file-loading and loop logic from graph execution.

### Integration smoke test (1) — real graph, real API

Marked `@pytest.mark.integration`. Excluded from the default `uv run pytest` run by adding to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = ["-m", "not integration"]
markers = ["integration: marks tests that require a real ANTHROPIC_API_KEY"]
```

Runs `"hypertension"` through the real compiled graph (ANTHROPIC_API_KEY loaded from `.env` via `pydantic_settings`). Asserts:
- `result.error is None`
- `SystemName.ICD10CM in result.predicted_systems`
- `len(result.predicted_codes.get(SystemName.ICD10CM, [])) > 0`
- `result.summary != ""`

Does not assert specific code values (API responses can drift); just verifies the pipeline returns something sensible end-to-end.

To run:
```bash
uv run pytest tests/evaluation/test_runner.py -m integration -v
```

---

## Execution order

Sequential — one query at a time. No concurrency. Runtime is ~2–5 minutes for the full 31-query gold set. This is acceptable for a Phase 0 eval runner that is invoked manually, not in CI.

---

## Out of scope

- Concurrent query execution
- Saving results to disk (reporter.py's job)
- Metrics computation (metrics.py's job)
- Rate limiting or retry logic at the runner level (already handled inside the graph's tool layer)
