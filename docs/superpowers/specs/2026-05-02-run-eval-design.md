# `scripts/run_eval.py` — Eval Runner CLI Design

## Goal

A Typer CLI that runs the LangGraph pipeline against a versioned gold set, computes evaluation metrics, writes timestamped JSON + markdown reports, and renders a Rich summary table to the terminal.

## Architecture

**One file:** `scripts/run_eval.py`. Mirrors `scripts/run_query.py` — a Typer app and a Rich console, no new modules.

No changes to `evaluation/runner.py`, `evaluation/metrics.py`, or `evaluation/reporter.py`. All evaluation logic is already implemented there.

**Data flow:**
```
gold JSON  →  GoldSet.model_validate()
           →  loop: run_query(gq) per query  →  [per-query status line printed]
           →  compute_metrics(results, gold_set.queries)
           →  write_report(summary, run_id, output_dir)  →  results/eval_{run_id}.{json,md}
           →  render Rich summary tables to console
```

`run_id` is auto-generated: `datetime.now().strftime("%Y%m%d_%H%M%S")`.

---

## Public API

Run with:
```bash
uv run python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json
uv run python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json --output-dir results
```

---

## CLI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--gold` | `Path` | required | Path to gold set JSON |
| `--output-dir` | `Path` | `results/` | Directory for output files |

**Early exits (Rich error panel, exit code 1):**
- `ANTHROPIC_API_KEY` not set
- `--gold` file not found

---

## Console Output

### During the run

Header rule with run ID and query count, then one line per query as it completes:

```
─────────── Eval run 20260502_143022 — 10 queries ───────────
  q001 (simple): 4.2s
  q002 (simple): 3.8s
  q003 (multi_system): ERROR: Connection timeout
```

Latency is green; errors are red.

### After the run

Two Rich tables under a `Results` rule.

**Overall table** — columns: Metric, Value:
- Total queries
- Errors
- System-selection F1
- Top-3 recall
- Must-include hit rate
- Mean iterations
- Mean API calls

**By-type table** — columns: Type, N, System F1, Top-3 recall, Must-include, Mean iter, Mean API calls. `None` values render as `n/a`.

**File paths** printed after the tables:
```
Wrote: results/eval_20260502_143022.json
       results/eval_20260502_143022.md
```

No failures table in the console — that detail lives in the markdown report.

---

## Implementation

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from clinical_codes.cli.display import render_error
from clinical_codes.config import settings
from clinical_codes.evaluation.metrics import compute_metrics
from clinical_codes.evaluation.reporter import write_report
from clinical_codes.evaluation.runner import run_query
from clinical_codes.evaluation.schema import GoldSet

app = typer.Typer(add_completion=False)


@app.command()
def run(
    gold: Annotated[Path, typer.Option(help="Path to gold set JSON file")],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to write results")
    ] = Path("results"),
) -> None:
    console = Console()

    if not settings.anthropic_api_key:
        render_error(console, "ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        raise typer.Exit(code=1)

    if not gold.exists():
        render_error(console, f"Gold file not found: {gold}")
        raise typer.Exit(code=1)

    data = json.loads(gold.read_text(encoding="utf-8"))
    gold_set = GoldSet.model_validate(data)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    console.print(Rule(f"Eval run {run_id} — {len(gold_set.queries)} queries"))

    results = []
    for gq in gold_set.queries:
        result = run_query(gq)
        if result.error:
            status = f"[red]ERROR: {result.error}[/red]"
        else:
            status = f"[green]{result.latency_s:.1f}s[/green]"
        console.print(f"  {gq.id} ({gq.query_type}): {status}")
        results.append(result)

    summary = compute_metrics(results, gold_set.queries)
    json_path, md_path = write_report(summary, run_id, output_dir)

    console.print(Rule("Results"))

    overall = Table(show_header=True, header_style="bold")
    overall.add_column("Metric")
    overall.add_column("Value", justify="right")
    overall.add_row("Total queries", str(summary.n_total))
    overall.add_row("Errors", str(summary.n_errors))
    overall.add_row("System-selection F1", f"{summary.system_selection_f1:.2f}")
    overall.add_row("Top-3 recall", f"{summary.top3_recall:.2f}")
    overall.add_row("Must-include hit rate", f"{summary.must_include_hit_rate:.2f}")
    overall.add_row("Mean iterations", f"{summary.mean_iterations:.2f}")
    overall.add_row("Mean API calls", f"{summary.mean_api_calls:.2f}")
    console.print(overall)

    by_type = Table(show_header=True, header_style="bold")
    by_type.add_column("Type")
    by_type.add_column("N", justify="right")
    by_type.add_column("System F1", justify="right")
    by_type.add_column("Top-3 recall", justify="right")
    by_type.add_column("Must-include", justify="right")
    by_type.add_column("Mean iter", justify="right")
    by_type.add_column("Mean API calls", justify="right")
    for qt in summary.by_type.values():
        top3 = "n/a" if qt.top3_recall is None else f"{qt.top3_recall:.2f}"
        mi = "n/a" if qt.must_include_hit_rate is None else f"{qt.must_include_hit_rate:.2f}"
        by_type.add_row(
            qt.query_type, str(qt.n),
            f"{qt.system_selection_f1:.2f}", top3, mi,
            f"{qt.mean_iterations:.2f}", f"{qt.mean_api_calls:.2f}",
        )
    console.print(by_type)

    console.print(f"\nWrote: [bold]{json_path}[/bold]")
    console.print(f"       [bold]{md_path}[/bold]")


if __name__ == "__main__":
    app()
```

---

## Files

| Action | Path | Notes |
|---|---|---|
| Create | `scripts/run_eval.py` | The CLI |
| Modify | `CLAUDE.md` | Uncomment `run_eval` command; mark ✅ Done |

No new modules. No changes to `evaluation/`.

---

## Testing

No unit tests for the CLI layer — Typer+Rich output isn't isolatable at this scale. The underlying `runner`, `metrics`, and `reporter` modules are already covered by the existing test suite.

**Verification:**
```bash
uv run python -m py_compile scripts/run_eval.py && echo "OK"
uv run pytest
uv run python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json
```
