# `scripts/run_query.py` — CLI Entry Point Design

## Goal

A single Typer command that accepts a natural-language clinical term, runs the LangGraph pipeline, streams the final summary as it generates, and renders the results in a Rich table.

## Architecture

Two new modules with distinct responsibilities:

- **`scripts/run_query.py`** — thin Typer app: arg parsing, graph streaming loop, coordination between display functions
- **`src/clinical_codes/cli/display.py`** — all Rich rendering, independently testable

Supporting empties:
- `scripts/__init__.py` — makes `scripts/` a package so `python -m scripts.run_query` works
- `src/clinical_codes/cli/__init__.py`

---

## Public API

### `scripts/run_query.py`

One Typer command:

```python
@app.command()
def run(
    query: Annotated[str, typer.Argument(help="Clinical term to look up")],
    model: Annotated[str, typer.Option(help="Claude model ID")] = settings.llm_model,
    output: Annotated[str, typer.Option(help="Output format: table or json")] = "table",
    verbose: Annotated[bool, typer.Option(help="Show scores, iterations, tracebacks")] = False,
) -> None:
```

Invoked as: `uv run python -m scripts.run_query "metformin 500 mg"`

### `src/clinical_codes/cli/display.py`

```python
def update_status(status: Status, node_name: str) -> None
def render_results(console: Console, consolidated: dict[SystemName, list[CodeResult]], verbose: bool) -> None
def render_error(console: Console, message: str, tb: str | None = None) -> None
```

`Console` is passed in (not module-global) so tests can capture output with `Console(file=StringIO())`.

Summary streaming is handled inline in `run_query.py` — chunks printed directly as they arrive via `console.print(chunk.content, end="")`.

---

## UX Flow

```
$ uv run python -m scripts.run_query "metformin 500 mg"

⠋ Planning...
⠋ Searching...
⠋ Evaluating...
⠋ Consolidating...
⠋ Summarizing...

── Summary ──────────────────────────────────────────────────
Metformin 500 mg is a common oral diabetes medication...   ← streamed token by token

── Results ──────────────────────────────────────────────────
 System    Code        Display                    Score
 RXNORM    860975      metFORMIN 500 MG Tablet    1.00
 ...
```

### Graph streaming loop

The graph is invoked with `graph.astream(initial_state, stream_mode=["updates", "messages"])`. Each event is a `(mode, data)` tuple:

| Event | Action |
|---|---|
| `("updates", {"planner": ...})` | Spinner → "Searching..." |
| `("updates", {"executor": ...})` | Spinner → "Evaluating..." |
| `("updates", {"evaluator": ...})` | Spinner → "Consolidating..." (or "Refining... (2/2)" on loop) |
| `("updates", {"consolidator": ...})` | Spinner → "Summarizing..." |
| `("messages", (chunk, meta))` where `meta["langgraph_node"] == "summarizer"` | Stop spinner; print `chunk.content` inline |

After stream completes: call `render_results()`.

### JSON output (`--output json`)

Rich output (spinner, table) redirected to `stderr` via `Console(stderr=True)`. A JSON dict is printed to `stdout`:

```json
{
  "query": "metformin 500 mg",
  "summary": "...",
  "results": { "RXNORM": [{"code": "860975", "display": "...", "score": 1.0}] }
}
```

Makes the command pipeable to `jq`.

### Verbose mode (`--verbose`)

- Iteration count and systems selected per iteration printed after results table
- Confidence score column included in results table
- Traceback included in error panels

---

## Error Handling

| Failure | Detection | Display |
|---|---|---|
| Missing API key | Check `settings.anthropic_api_key` before graph invocation | Rich error panel, exit 1 |
| NLM / LLM error | `except Exception` in streaming loop | Rich error panel, exit 1 |
| `--verbose` | `traceback.format_exc()` inside error panel | Traceback visible |

`render_error(console, message, tb=None)` handles all three cases.

---

## Testing

File: `tests/cli/test_display.py`

`run_query.py` is not tested directly (Typer + asyncio + real graph = integration territory). Only `display.py` is unit-tested using `Console(file=StringIO())` to capture Rich output.

| Test | Assertion |
|---|---|
| `render_results` with non-empty `consolidated` | System name and code appear in output |
| `render_results` with empty `consolidated` | "No results" appears |
| `render_results` with `verbose=True` | Score column header appears |
| `render_error` with message only | Message appears; no traceback text |
| `render_error` with `tb="..."` | Traceback text appears |

Also needed: `tests/cli/__init__.py` (empty).

---

## Dependencies to add

```toml
typer>=0.12
rich>=13.0
```

Both are pure-Python, no system deps. `rich` is already a transitive dep of several packages in the tree; adding it explicitly pins the minimum.
