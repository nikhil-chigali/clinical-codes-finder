# Reporter Design

## Goal

Given a `MetricsSummary` and a caller-supplied `run_id` string, write two files to `results/`:

1. `eval_{run_id}.json` — full `MetricsSummary` dump (machine-readable, for downstream analysis)
2. `eval_{run_id}.md` — human-readable markdown with three sections: overall metrics table, per-type breakdown, and a failure list

The caller (e.g. `run_eval.py`) decides what `run_id` means — it might be a gold set version (`"v0.1.1"`), a timestamp, or any other meaningful label. `reporter.py` is a pure formatting + I/O module; it has no knowledge of the graph, runner, gold set schema, or naming policy.

---

## File map

| File | Action | Purpose |
|---|---|---|
| `src/clinical_codes/evaluation/reporter.py` | Create | `format_markdown` (pure) + `write_report` (I/O) |
| `tests/evaluation/test_reporter.py` | Create | Unit tests for both functions |

---

## Public API

```python
from pathlib import Path
from clinical_codes.evaluation.metrics import MetricsSummary

def format_markdown(summary: MetricsSummary) -> str:
    """Pure function. Returns the full markdown report as a string."""

def write_report(
    summary: MetricsSummary,
    run_id: str,
    output_dir: Path = Path("results"),
) -> tuple[Path, Path]:
    """Writes eval_{run_id}.json and eval_{run_id}.md to output_dir.
    Creates output_dir if it does not exist.
    Returns (json_path, md_path).
    """
```

`write_report` calls `format_markdown` internally. The JSON is written with `summary.model_dump_json(indent=2)`. `output_dir` is created via `output_dir.mkdir(parents=True, exist_ok=True)`. The caller passes whatever `run_id` is meaningful (e.g. `"v0.1.1"`, `"2026-04-30"`).

---

## Markdown structure

```
# Eval results — {run_id}

## Overall

| Metric | Value |
|---|---|
| Total queries | {n_total} |
| Errors | {n_errors} |
| System-selection F1 | {system_selection_f1:.2f} |
| Top-3 recall | {top3_recall} |
| Must-include hit rate | {must_include_hit_rate:.2f} |
| Mean iterations | {mean_iterations:.2f} |
| Mean API calls | {mean_api_calls:.2f} |

## By query type

| Type | N | System F1 | Top-3 recall | Must-include | Mean iter | Mean API calls |
|---|---|---|---|---|---|---|
| {query_type} | {n} | {system_selection_f1:.2f} | {top3_recall} | {must_include_hit_rate} | {mean_iterations:.2f} | {mean_api_calls:.2f} |
...

## Failures (system_f1 < 1.0 or error)

| Query ID | Query | Type | System F1 | Error |
|---|---|---|---|---|
| {query_id} | {query} | {query_type} | {system_f1:.2f} | {error or "—"} |
...
```

---

## `None` rendering rules

| Field | When `None` | Render as |
|---|---|---|
| `QueryTypeMetrics.top3_recall` | miss type (no expected codes) | `n/a` |
| `QueryTypeMetrics.must_include_hit_rate` | no must_include in this type | `n/a` |
| `MetricsSummary.top3_recall` (typed `float`) | all types have `top3_recall is None` | `n/a` |
| `MetricsSummary.must_include_hit_rate` (typed `float`) | always a float | `{x:.2f}` |

**All-miss-type edge case:** `MetricsSummary.top3_recall` is typed `float` (substitutes `0.0` for `None` at the overall level). `format_markdown` detects this case by checking whether every `QueryTypeMetrics` in `summary.by_type.values()` has `top3_recall is None`. If so, render `n/a` for the overall Top-3 recall row instead of `0.00`.

**Floats:** formatted to 2 decimal places (`f"{x:.2f}"`).

**Failures section:** if no query has `system_f1 < 1.0` and no query has `error is not None`, render `*(none)*` instead of a table.

---

## Failure list definition

A query appears in the Failures section if **either**:
- `query_metrics.error is not None` (runner-level failure), OR
- `query_metrics.system_f1 < 1.0` (imperfect system selection, including partial overlaps)

Both conditions are checked against `summary.per_query`. Queries are listed in input order (no additional sort).

---

## Testing (`tests/evaluation/test_reporter.py`)

All tests use a shared `make_summary()` helper that builds a minimal `MetricsSummary` with:
- One `simple` query (`query_id="q001"`): `system_f1=0.5`, no error, `recall_at_3=0.8`, `must_include_hit_rate=1.0`
- One `miss` query (`query_id="q002"`): `system_f1=1.0` (correct miss), no error, `recall_at_3=None`, `must_include_hit_rate=None`
- One `simple` query (`query_id="q003"`) with error: `system_f1=0.0`, `error="timeout"`, `recall_at_3=0.0`, `must_include_hit_rate=0.0`

### Tests for `format_markdown`

```python
def test_overall_table_values():
    md = format_markdown(make_summary())
    assert "## Overall" in md
    assert str(make_summary().n_total) in md

def test_by_type_miss_shows_n_a_for_recall():
    md = format_markdown(make_summary())
    # The miss row should show n/a for Top-3 recall
    assert "n/a" in md

def test_failures_includes_error_query():
    md = format_markdown(make_summary())
    assert "timeout" in md

def test_failures_includes_low_f1_query():
    md = format_markdown(make_summary())
    # q001 has system_f1=0.5 — must appear in the failures table
    assert "q001" in md

def test_no_failures_shows_none():
    # Build a summary where all queries have system_f1=1.0 and no errors
    summary = make_perfect_summary()
    md = format_markdown(summary)
    assert "*(none)*" in md

def test_all_miss_type_overall_recall_shows_n_a():
    # Build a summary where every QueryTypeMetrics has top3_recall=None
    summary = make_all_miss_summary()
    md = format_markdown(summary)
    lines = [l for l in md.splitlines() if "Top-3 recall" in l]
    assert any("n/a" in l for l in lines)
```

### Tests for `write_report`

```python
def test_write_report_creates_both_files(tmp_path):
    json_path, md_path = write_report(make_summary(), run_id="v0.1.1", output_dir=tmp_path)
    assert json_path.exists()
    assert md_path.exists()

def test_write_report_json_roundtrips(tmp_path):
    json_path, _ = write_report(make_summary(), run_id="v0.1.1", output_dir=tmp_path)
    loaded = MetricsSummary.model_validate_json(json_path.read_text())
    assert loaded.n_total == make_summary().n_total

def test_write_report_filenames_contain_version(tmp_path):
    json_path, md_path = write_report(make_summary(), run_id="v0.1.1", output_dir=tmp_path)
    assert "v0.1.1" in json_path.name
    assert "v0.1.1" in md_path.name
```

---

## Out of scope

- Printing to stdout (caller's responsibility — `run_eval.py` can print the returned paths)
- Sorting or re-ordering the failure list beyond input order
- Per-query detail table in the markdown (full data available in JSON)
- Latency percentiles or histograms (mean only, already in `MetricsSummary`)
